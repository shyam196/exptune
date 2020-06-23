import abc
from typing import Any

import numpy as np
from ray.tune import loguniform, sample_from, uniform

from .hyperparams import (
    ChoiceHyperParam,
    HyperParam,
    LogUniformHyperParam,
    UniformHyperParam,
)


class SearchStrategy(abc.ABC):
    @abc.abstractmethod
    def process_hparam(self, hparam: HyperParam) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError


def _type_preserving_choice(*args, **kwargs):
    # add .item to return a native python type so this will work with tensorboard logging
    return sample_from(lambda _: np.random.choice(*args, **kwargs).item())


class RandomSearchStrategy(SearchStrategy):
    def __init__(self, num_samples: int):
        super().__init__()
        self._num_samples = num_samples

    def num_samples(self) -> int:
        return self._num_samples

    def process_hparam(self, hparam: HyperParam) -> sample_from:
        if isinstance(hparam, UniformHyperParam):
            return uniform(hparam.low, hparam.high)
        elif isinstance(hparam, LogUniformHyperParam):
            return loguniform(hparam.low, hparam.high)
        elif isinstance(hparam, ChoiceHyperParam):
            return _type_preserving_choice(hparam.choices)
        else:
            raise ValueError("Unsupported hyperparameter for random search")
