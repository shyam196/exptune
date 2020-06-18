import abc
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray.tune as tune
from ray import ObjectID
from ray.tune import ExperimentAnalysis
from ray.tune.schedulers import TrialScheduler

from .hyperparams import HyperParam
from .search_strategies import SearchStrategy
from .utils import (
    DEBUG_MODE_KEY,
    EXP_CONF_KEY,
    HPARAMS_KEY,
    PINNED_OID_KEY,
    convert_experiment_analysis_to_df,
)


@dataclass
class ExperimentSettings:
    exp_name: str
    timestamp_experiment_name: bool = True
    checkpoint_freq: int = 0
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 1
    reuse_actors: bool = False
    raise_on_failed_trial: bool = False
    max_retries: int = 3
    final_max_iterations: int = 100
    final_repeats: int = 5
    final_run_timeout: Optional[float] = None

    @property
    def name(self):
        if self.timestamp_experiment_name:
            return f"{self.exp_name}_{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}"
        else:
            return self.exp_name


@dataclass
class TrialResources:
    cpus: float
    gpus: float

    def as_dict(self) -> Dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus}

    def requests_gpu(self) -> bool:
        return self.gpus > 0.0


class ComposeStopper(tune.Stopper):
    def __init__(self, stoppers: List[tune.Stopper]):
        super().__init__()
        self.stoppers = stoppers

    def __call__(self, trial_id, result):
        stop = [s(trial_id, result) for s in self.stoppers]
        return any(stop)

    def stop_all(self):
        return any([s.stop_all() for s in self.stoppers])


class SearchSummarizer(abc.ABC):
    def __call__(self, search_df: pd.DataFrame) -> None:
        raise NotImplementedError


class FinalRunsSummarizer(abc.ABC):
    def __call__(self, training_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        raise NotImplementedError


class ExperimentConfig(abc.ABC):
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
    def trial_metric(self) -> Tuple[str, str]:
        raise NotImplementedError

    def stoppers(self) -> List[tune.Stopper]:
        return []

    def dataset_pin(self, debug_mode: bool) -> List[ObjectID]:
        return []

    @abc.abstractmethod
    def data(
        self, pinned_objs: List[ObjectID], hparams: Dict[str, Any], debug_mode: bool
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def model(self, hparams: Dict[str, Any], debug_mode: bool) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def optimizer(self, model: Any, hparams: Dict[str, Any], debug_mode: bool) -> Any:
        raise NotImplementedError

    def extra_setup(
        self, model: Any, optimizer: Any, hparams: Dict[str, Any], debug_mode: bool,
    ) -> Any:
        return None

    @abc.abstractmethod
    def persist_trial(
        self,
        checkpoint_dir: Path,
        model: Any,
        optimizer: Any,
        hparams: Dict[str, Any],
        extra: Any,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def restore_trial(
        self, checkpoint_dir: Path
    ) -> Tuple[Any, Any, Dict[str, Any], Any]:
        raise NotImplementedError

    def trial_update_hparams(
        self, model: Any, optimizer: Any, extra: Any, new_hparams: Dict[str, Any]
    ) -> bool:
        # used by PBT
        return False

    @abc.abstractmethod
    def train(
        self, model: Any, optimizer: Any, data: Any, extra: Any, debug_mode: bool
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def val(
        self, model: Any, data: Any, extra: Any, debug_mode: bool
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def test(
        self, model: Any, data: Any, extra: Any, debug_mode: bool
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError

    def search_summaries(self) -> List[SearchSummarizer]:
        return []

    def final_runs_summaries(self) -> List[FinalRunsSummarizer]:
        return []

    def __repr__(self):
        return str(self.__class__.__name__)


class _ExperimentTrainable(tune.Trainable):
    def _setup(self, config: Dict[str, Any]):
        self.exp_conf: ExperimentConfig = config[EXP_CONF_KEY]
        self.debug_mode: bool = config[DEBUG_MODE_KEY]
        self.hparams: Dict[str, Any] = config[HPARAMS_KEY]
        pinned_object_ids: List[ObjectID] = config[PINNED_OID_KEY]

        self.data: Any = self.exp_conf.data(
            pinned_object_ids, self.hparams, self.debug_mode
        )
        self.model: Any = self.exp_conf.model(self.hparams, self.debug_mode)
        self.optimizer: Any = self.exp_conf.optimizer(
            self.model, self.hparams, self.debug_mode
        )
        self.extra: Any = self.exp_conf.extra_setup(
            self.model, self.optimizer, self.hparams, self.debug_mode
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

    def _train(self):
        t_metrics: Dict[str, Any]
        t_extra: Any
        t_metrics, t_extra = self.exp_conf.train(
            self.model, self.optimizer, self.data, self.extra, self.debug_mode
        )

        v_metrics: Dict[str, Any]
        v_extra: Any
        v_metrics, v_extra = self.exp_conf.val(
            self.model, self.data, self.extra, self.debug_mode
        )

        self._log_extra_info(t_extra, "train")
        self._log_extra_info(v_extra, "val")

        return {**t_metrics, **v_metrics}

    def _save(self, checkpoint_dir):
        self.exp_conf.persist_trial(
            Path(checkpoint_dir), self.model, self.optimizer, self.hparams, self.extra
        )
        return checkpoint_dir

    def _restore(self, checkpoint_dir):
        (
            self.model,
            self.optimizer,
            self.hparams,
            self.extra,
        ) = self.exp_conf.restore_trial(Path(checkpoint_dir))


def run_search(
    experiment_config: ExperimentConfig, debug_mode=False
) -> tune.ExperimentAnalysis:
    search_strategy: SearchStrategy = experiment_config.search_strategy()
    settings: ExperimentSettings = experiment_config.settings()

    config = {
        EXP_CONF_KEY: experiment_config,
        DEBUG_MODE_KEY: debug_mode,
        PINNED_OID_KEY: experiment_config.dataset_pin(debug_mode),
    }

    # Set up hyperparameters
    hparams = experiment_config.hyperparams()
    for k, v in experiment_config.fixed_hyperparams().items():
        hparams[k] = v
    for k, v in hparams.items():
        if isinstance(v, HyperParam):
            hparams[k] = search_strategy.process_hparam(v)

    config[HPARAMS_KEY] = hparams

    analysis: ExperimentAnalysis = tune.run(
        _ExperimentTrainable,
        name=settings.name,
        stop=ComposeStopper(experiment_config.stoppers()),
        config=config,
        resources_per_trial=experiment_config.resource_requirements().as_dict(),
        num_samples=search_strategy.num_samples(),
        scheduler=experiment_config.trial_scheduler(),
        checkpoint_freq=settings.checkpoint_freq,
        checkpoint_at_end=settings.checkpoint_at_end,
        keep_checkpoints_num=settings.keep_checkpoints_num,
        reuse_actors=settings.reuse_actors,
        raise_on_failed_trial=settings.raise_on_failed_trial,
        ray_auto_init=False,
    )

    search_df: pd.DataFrame = convert_experiment_analysis_to_df(analysis)
    for summarizer in experiment_config.search_summaries():
        summarizer(search_df)

    return analysis
