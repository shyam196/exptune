import abc
from typing import Any, Dict, Tuple

import numpy as np
from ray.tune import grid_search, loguniform, sample_from, uniform

from .hyperparams import (
    ChoiceHyperParam,
    HyperParam,
    LogUniformHyperParam,
    UniformHyperParam,
)

# Attention: Tune can only serialise native types to Tensorboard for hparams
# Take care to convert from numpy to native types!


class SearchStrategy(abc.ABC):
    @abc.abstractmethod
    def process_hparam(self, hparam: Tuple[str, HyperParam]) -> Any:
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

    def process_hparam(self, hparam: Tuple[str, HyperParam]) -> sample_from:
        hparam_obj: HyperParam = hparam[1]
        if isinstance(hparam_obj, UniformHyperParam):
            return uniform(hparam_obj.low, hparam_obj.high)
        elif isinstance(hparam_obj, LogUniformHyperParam):
            return loguniform(hparam_obj.low, hparam_obj.high)
        elif isinstance(hparam_obj, ChoiceHyperParam):
            return _type_preserving_choice(hparam_obj.choices)
        else:
            raise ValueError("Unsupported hyperparameter for random search")


class GridSearchStrategy(SearchStrategy):
    def __init__(self, grid_spec: Dict[str, int], repeats=1):
        super().__init__()
        self.grid_spec: Dict[str, int] = grid_spec
        self.repeats: int = repeats

    def num_samples(self) -> int:
        return self.repeats

    def process_hparam(self, hparam: Tuple[str, HyperParam]) -> grid_search:
        name, hparam_obj = hparam
        if isinstance(hparam_obj, UniformHyperParam):
            if name not in self.grid_spec:
                raise ValueError(
                    f"Grid dimension for hyperparameter {name} not specified"
                )

            return grid_search(
                [
                    float(f)
                    for f in np.linspace(
                        hparam_obj.low, hparam_obj.high, num=self.grid_spec[name]
                    )
                ]
            )

        elif isinstance(hparam_obj, LogUniformHyperParam):
            if name not in self.grid_spec:
                raise ValueError(
                    f"Grid dimension for hyperparameter {name} not specified"
                )

            low = np.log10(hparam_obj.low)
            high = np.log10(hparam_obj.high)

            return grid_search(
                [
                    float(f)
                    for f in 10 ** np.linspace(low, high, num=self.grid_spec[name])
                ]
            )

        elif isinstance(hparam_obj, ChoiceHyperParam):
            if name in self.grid_spec:
                raise ValueError(
                    f"Grid dimension for hyperparameter {name} specified but "
                    "it is a choice parameter!"
                )

            return grid_search(hparam_obj.choices)

        else:
            raise ValueError("Unsupported hyperparameter for random search")
