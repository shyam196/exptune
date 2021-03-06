import abc
import pickle
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from ray import ObjectID
from ray import tune as tune
from ray.tune import ExperimentAnalysis, ProgressReporter
from ray.tune.progress_reporter import CLIReporter
from ray.tune.schedulers import TrialScheduler

from exptune.hyperparams import HyperParam
from exptune.search_strategies import SearchStrategy
from exptune.utils import (
    BEST_HYPERPARAMS_FILE,
    EXP_CONF_KEY,
    HPARAMS_KEY,
    PINNED_OID_KEY,
    SEARCH_DF_FILE,
    SEARCH_DIR,
    SEARCH_SUMMARY_DIR,
    ComposeStopper,
    ExperimentSettings,
    FinalRunsSummarizer,
    Metric,
    SearchSummarizer,
    TrialResources,
    convert_experiment_analysis_to_df,
)

TModel = TypeVar("TModel")
TOpt = TypeVar("TOpt")
TExtra = TypeVar("TExtra")
TData = TypeVar("TData")


class ExperimentConfig(abc.ABC, Generic[TModel, TOpt, TExtra, TData]):
    def __init__(self, debug_mode=False) -> None:
        super().__init__()
        self.debug_mode = debug_mode

    @abc.abstractmethod
    def settings(self) -> ExperimentSettings:
        raise NotImplementedError

    def configure_seeds(self, seed: int) -> None:
        pass

    @abc.abstractmethod
    def resource_requirements(self) -> TrialResources:
        raise NotImplementedError

    @abc.abstractmethod
    def hyperparams(self) -> Dict[str, HyperParam]:
        raise NotImplementedError

    def fixed_hyperparams(self) -> Dict[str, HyperParam]:
        return {}

    @abc.abstractmethod
    def search_strategy(self) -> SearchStrategy:
        raise NotImplementedError

    @abc.abstractmethod
    def trial_scheduler(self) -> TrialScheduler:
        raise NotImplementedError

    @abc.abstractmethod
    def trial_metric(self) -> Metric:
        raise NotImplementedError

    def stoppers(self) -> List[tune.Stopper]:
        return []

    def dataset_pin(self) -> List[ObjectID]:
        return []

    def trial_init(self) -> None:
        pass

    @abc.abstractmethod
    def data(self, pinned_objs: List[ObjectID], hparams: Dict[str, Any]) -> TData:
        raise NotImplementedError

    @abc.abstractmethod
    def model(self, hparams: Dict[str, Any]) -> TModel:
        raise NotImplementedError

    @abc.abstractmethod
    def optimizer(self, model: Any, hparams: Dict[str, Any]) -> TOpt:
        raise NotImplementedError

    def extra_setup(
        self, model: TModel, optimizer: TOpt, hparams: Dict[str, Any]
    ) -> Optional[TExtra]:
        return None

    @abc.abstractmethod
    def persist_trial(
        self,
        checkpoint_dir: Path,
        model: TModel,
        optimizer: TOpt,
        hparams: Dict[str, Any],
        extra: TExtra,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def restore_trial(
        self, checkpoint_dir: Path
    ) -> Tuple[TModel, TOpt, Dict[str, Any], TExtra]:
        raise NotImplementedError

    @abc.abstractmethod
    def train(
        self, model: TModel, optimizer: TOpt, data: TData, extra: TExtra, iteration: int
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def val(
        self, model: TModel, data: TData, extra: TExtra, iteration: int
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def test(
        self, model: TModel, data: TData, extra: TExtra
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    def search_summaries(self) -> List[SearchSummarizer]:
        return []

    def final_runs_summaries(self) -> List[FinalRunsSummarizer]:
        return []

    def progress_reporter(self) -> ProgressReporter:
        metric = self.trial_metric()
        return CLIReporter(metric=metric.name, mode=metric.mode, infer_limit=10)

    def __repr__(self):
        return str(self.__class__.__name__)


class _ET(tune.Trainable):
    """Terse Name to reduce space taken up when Ray reports a trial"""

    def setup(self, config: Dict[str, Any]):
        self.exp_conf: ExperimentConfig = config[EXP_CONF_KEY]
        self.exp_conf.trial_init()

        self.hparams: Dict[str, Any] = config[HPARAMS_KEY]
        pinned_object_ids: List[ObjectID] = config[PINNED_OID_KEY]

        self.data: Any = self.exp_conf.data(pinned_object_ids, self.hparams)
        self.model: Any = self.exp_conf.model(self.hparams)
        self.optimizer: Any = self.exp_conf.optimizer(self.model, self.hparams)
        self.extra: Any = self.exp_conf.extra_setup(
            self.model, self.optimizer, self.hparams
        )

    def reset_config(self, new_config):
        new_hparams: Dict[str, Any] = new_config[HPARAMS_KEY]
        updated: bool = self.exp_conf.trial_update_hparams(
            self.model, self.optimizer, self.extra, new_hparams
        )
        if updated:
            self.hparams = new_hparams
        return updated

    def _log_extra_info(self, data: Any, name: str):
        if data is None:
            return
        with open(Path("extra_info") / f"{name}_{self.iteration}.pkl", "wb") as f:
            pickle.dump(data, f)

    def step(self):
        t_metrics: Dict[str, Any]
        t_extra: Any
        t_metrics, t_extra = self.exp_conf.train(
            self.model, self.optimizer, self.data, self.extra, self.iteration
        )

        v_metrics: Dict[str, Any]
        v_extra: Any
        v_metrics, v_extra = self.exp_conf.val(
            self.model, self.data, self.extra, self.iteration
        )

        self._log_extra_info(t_extra, "train")
        self._log_extra_info(v_extra, "val")

        return {**t_metrics, **v_metrics}

    def save_checkpoint(self, checkpoint_dir):
        self.exp_conf.persist_trial(
            Path(checkpoint_dir), self.model, self.optimizer, self.hparams, self.extra
        )
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        (
            self.model,
            self.optimizer,
            self.hparams,
            self.extra,
        ) = self.exp_conf.restore_trial(Path(checkpoint_dir))


def run_search(
    experiment_config: ExperimentConfig,
    exp_directory: Path,
    verbosity=2,
    np_seed=None,
) -> Dict[str, Any]:
    settings: ExperimentSettings = experiment_config.settings()
    # Make the experiment directory
    if exp_directory.exists():
        print(f"Experiment directory ({exp_directory}) already exists; continuing...")
    else:
        print(f"Creating experiment directory ({exp_directory})")
        exp_directory.mkdir(parents=True)

    search_strategy: SearchStrategy = experiment_config.search_strategy()

    config = {
        EXP_CONF_KEY: experiment_config,
        PINNED_OID_KEY: experiment_config.dataset_pin(),
    }

    # Set up hyperparameters
    if np_seed is not None:
        np.random.seed(np_seed)
    hparams = experiment_config.hyperparams()
    for k, v in experiment_config.fixed_hyperparams().items():
        hparams[k] = v
    for k, v in hparams.items():
        if isinstance(v, HyperParam):
            hparams[k] = search_strategy.process_hparam((k, v))

    config[HPARAMS_KEY] = hparams

    analysis: ExperimentAnalysis = tune.run(
        _ET,
        name=settings.name,
        stop=ComposeStopper(experiment_config.stoppers()),
        config=config,
        resources_per_trial=experiment_config.resource_requirements().as_dict(),
        num_samples=search_strategy.num_samples(),
        search_alg=search_strategy.search_algorithm(),
        scheduler=experiment_config.trial_scheduler(),
        checkpoint_freq=settings.checkpoint_freq,
        checkpoint_at_end=settings.checkpoint_at_end,
        keep_checkpoints_num=settings.keep_checkpoints_num,
        reuse_actors=settings.reuse_actors,
        raise_on_failed_trial=settings.raise_on_failed_trial,
        ray_auto_init=False,
        local_dir=str(exp_directory.expanduser() / SEARCH_DIR),
        verbose=verbosity,
        progress_reporter=experiment_config.progress_reporter(),
    )

    metric = experiment_config.trial_metric()
    analysis.default_metric = metric.name
    analysis.default_mode = metric.mode

    # used for the search summaries
    search_df: pd.DataFrame = convert_experiment_analysis_to_df(analysis)
    search_df.to_pickle(str(exp_directory / SEARCH_DF_FILE))

    best_config: Dict[str, Any] = analysis.get_best_trial(scope="all").config
    best_hparams: Dict[str, Any] = best_config[HPARAMS_KEY]

    with open(exp_directory / BEST_HYPERPARAMS_FILE, "w") as f:
        f.write(str(best_hparams) + "\n")

    print(search_df.to_string())
    summary_dir = exp_directory / SEARCH_SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    print("Summarizing search results to: ", summary_dir)
    for summarizer in experiment_config.search_summaries():
        try:
            summarizer(summary_dir, search_df)
        except Exception:
            print("Failed summariser:", summarizer)

    return best_hparams
