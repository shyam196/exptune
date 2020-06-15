import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from ray import ObjectID
from ray.tune import Stopper
from ray.tune.schedulers import TrialScheduler


class ExperimentSettings:
    def __init__(
        self,
        base_name: str,
        timestamp: bool = True,
        checkpoint_freq: int = 0,
        checkpoint_at_end: bool = True,
        keep_checkpoints_num: int = 1,
        reuse_actors: bool = True,
        raise_on_failed_trial: bool = False,
    ):
        self.base_name: str = base_name
        self.timestamp: bool = timestamp
        self.checkpoint_freq: int = checkpoint_freq
        self.checkpoint_at_end: bool = checkpoint_at_end
        self.keep_checkpoints_num: int = keep_checkpoints_num
        self.reuse_actors: bool = reuse_actors
        self.raise_on_failed_trial: bool = raise_on_failed_trial


class TrialResources:
    def __init__(self, cpus: float, gpus: float):
        self.cpus: float = cpus
        self.gpus: float = gpus


class SearchStrategy(abc.ABCMeta):
    pass


class HyperParam(abc.ABCMeta):
    pass


@dataclass
class TrialFunctions:
    data: Callable
    model: Callable
    optimizer: Callable
    lr_scheduler: Callable
    train: Callable
    val: Callable
    test: Callable


class Summarizer(abc.ABCMeta):
    def __call__(self, train_df, val_df, test_df):
        raise NotImplementedError


class ExperimentConfig(abc.ABCMeta):
    @abc.abstractmethod
    def settings(self) -> ExperimentSettings:
        raise NotImplementedError

    def configure_seeds(self) -> None:
        pass

    @abc.abstractmethod
    def resource_requirements(self) -> TrialResources:
        raise NotImplementedError

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

    def dataset_pin(self) -> List[ObjectID]:
        return []

    def data_config(self) -> Dict[str, Any]:
        return {}

    def model_config(self) -> Dict[str, Any]:
        return {}

    def optim_config(self) -> Dict[str, Any]:
        return {}

    def lr_scheduler_config(self) -> Dict[str, Any]:
        return {}

    def extra_config(self) -> Dict[str, Any]:
        return {}

    @abc.abstractmethod
    def trial_functions(self) -> TrialFunctions:
        raise NotImplementedError

    def summaries(self) -> List[Summarizer]:
        return []
