import abc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ray import ObjectID
from ray.tune import Stopper, Trainable
from ray.tune.schedulers import TrialScheduler

from .hyperparams import HyperParam


@dataclass
class ExperimentSettings:
    exp_name: str
    timestamp_experiment: bool = True
    checkpoint_freq: int = 0
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 1
    reuse_actors: bool = False
    raise_on_failed_trial: bool = False


@dataclass
class TrialResources:
    cpus: float
    gpus: float


class SearchStrategy(abc.ABCMeta):
    pass


@dataclass
class Optimizer:
    optimizer: Any
    lr_scheduler: Any


class Summarizer(abc.ABCMeta):
    def __call__(self, train_df, val_df, test_df):
        raise NotImplementedError


class ExperimentConfig(abc.ABCMeta):
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
    def trial_metric(self) -> str:
        raise NotImplementedError

    def trial_stoppers(self) -> List[Stopper]:
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
    def optimizer(
        self, model: Any, hparams: Dict[str, Any], debug_mode: bool
    ) -> Optimizer:
        raise NotImplementedError

    def extra_setup(
        self,
        model: Any,
        optimizer: Optimizer,
        hparams: Dict[str, Any],
        debug_mode: bool,
    ) -> Any:
        return None

    @abc.abstractmethod
    def persist_trial(
        self,
        checkpoint_dir: Path,
        model: Any,
        optimizer: Optimizer,
        hparams: Dict[str, Any],
        extra: Any,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def restore_trial(
        self, checkpoint_dir: Path
    ) -> Tuple[Any, Optimizer, Dict[str, Any], Any]:
        raise NotImplementedError

    def trial_update_hparams(
        self, model: Any, optimizer: Optimizer, extra: Any, new_hparams: Dict[str, Any]
    ) -> bool:
        # used by PBT
        return False

    @abc.abstractmethod
    def train(
        self, model: Any, optimizer: Optimizer, data: Any, extra: Any, debug_mode: bool
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
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def summaries(self) -> List[Summarizer]:
        return []

    def __repr__(self):
        return str(self.__name__)


_EXP_CONF_KEY = "exp_conf_obj"
_HPARAMS_KEY = "hparams"
_PINNED_OID_KEY = "pinned_obj_ids"
_DEBUG_MODE_KEY = "debug_mode"


class _ExperimentTrainable(Trainable):
    def _setup(self, config: Dict[str, Any]):
        self.exp_conf: ExperimentConfig = config[_EXP_CONF_KEY]
        self.debug_mode: bool = config[_DEBUG_MODE_KEY]
        self.hparams: Dict[str, Any] = config[_HPARAMS_KEY]
        pinned_object_ids: List[ObjectID] = config[_PINNED_OID_KEY]

        self.data: Any = self.exp_conf.data(
            pinned_object_ids, self.hparams, self.debug_mode
        )
        self.model: Any = self.exp_conf.model(self.hparams, self.debug_mode)
        self.optimizer: Optimizer = self.exp_conf.optimizer(
            self.model, self.hparams, self.debug_mode
        )
        self.extra: Any = self.exp_conf.extra_setup(
            self.model, self.optimizer, self.hparams, self.debug_mode
        )

    def reset_config(self, new_config):
        new_hparams: Dict[str, Any] = new_config[_HPARAMS_KEY]
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

    def train(self):
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

        return {**t_extra, **v_extra}

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
