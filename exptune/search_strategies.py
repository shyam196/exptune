import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ray.tune import grid_search, loguniform, sample_from, uniform
from ray.tune.suggest import Searcher
from ray.tune.suggest.bayesopt import BayesOptSearch

from exptune.hyperparams import (
    ChoiceHyperParam,
    HyperParam,
    LogUniformHyperParam,
    UniformHyperParam,
)
from exptune.utils import HPARAMS_KEY, Metric

# Attention: Tune can only serialise native types to Tensorboard for hparams
# Take care to convert from numpy to native types!


class SearchStrategy(abc.ABC):
    @abc.abstractmethod
    def process_hparam(self, hparam: Tuple[str, HyperParam]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def search_algorithm(self) -> Any:
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

    def search_algorithm(self):
        return None

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

    def search_algorithm(self):
        return None

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


class _BayesTrigger(abc.ABC):
    @abc.abstractmethod
    def __call__(self, val: float):
        raise NotImplementedError


class _LogTrigger(_BayesTrigger):
    def __call__(self, val):
        return 10 ** val


class _DiscreteTrigger(_BayesTrigger):
    def __init__(self, choices):
        super().__init__()
        self.choices: List[Any] = choices

    def __call__(self, val):
        return self.choices[int(val)]


class _BayesOptSearchWrapper(BayesOptSearch):
    def __init__(
        self,
        space: Dict[str, Tuple[float, float]],
        triggers: Dict[str, _BayesTrigger],
        metric="episode_reward_mean",
        mode="max",
        utility_kwargs=None,
        random_state=1,
        verbose=0,
        max_concurrent=None,
        use_early_stopped_trials=None,
    ):
        super().__init__(
            space,
            metric=metric,
            mode=mode,
            utility_kwargs=utility_kwargs,
            random_state=random_state,
            verbose=verbose,
            max_concurrent=max_concurrent,
            use_early_stopped_trials=use_early_stopped_trials,
        )
        self._triggers: Dict[str, _BayesTrigger] = triggers

    def suggest(self, trial_id):
        suggestions: Dict[str, float] = super().suggest(trial_id)
        return_dict = {HPARAMS_KEY: {}}

        for name, val in suggestions.items():
            if name in self._triggers and self._triggers[name] is not None:
                val = self._triggers[name](val)
            return_dict[HPARAMS_KEY][name] = val

        return return_dict


class _ExceptionHandlingConcurrencyLimiter(Searcher):
    def __init__(self, searcher, max_concurrent):
        assert type(max_concurrent) is int and max_concurrent > 0
        self.searcher = searcher
        self.max_concurrent = max_concurrent
        self.live_trials = set()
        super(_ExceptionHandlingConcurrencyLimiter, self).__init__(
            metric=self.searcher.metric, mode=self.searcher.mode
        )

    def suggest(self, trial_id):
        if len(self.live_trials) >= self.max_concurrent:
            return None

        try:
            suggestion = self.searcher.suggest(trial_id)
            self.live_trials.add(trial_id)
            return suggestion
        except Exception:
            return None

    def on_trial_complete(self, trial_id, result=None, error=False):
        if trial_id not in self.live_trials:
            return
        else:
            self.searcher.on_trial_complete(trial_id, result=result, error=error)
            self.live_trials.remove(trial_id)

    def save(self, checkpoint_dir):
        self.searcher.save(checkpoint_dir)

    def restore(self, checkpoint_dir):
        self.searcher.restore(checkpoint_dir)


class BayesOptSearchStrategy(SearchStrategy):
    def __init__(
        self,
        *,
        num_samples: int,
        max_concurrent: int,
        metric: Metric,
        acq_type="ucb",
        kappa=5.0,
        kappa_decay=0.99,
        xi=1.0,
    ):
        super().__init__()
        self.registered_hparams: Dict[
            str, Tuple[Tuple[float, float], Optional[_BayesTrigger]]
        ] = dict()
        self._num_samples: int = num_samples
        self.max_concurrent: int = max_concurrent
        self.metric: Metric = metric

        self.acq_type: str = acq_type
        self.kappa: float = kappa
        self.kappa_decay: float = kappa_decay
        self.xi: float = xi

    def num_samples(self):
        return self._num_samples

    def search_algorithm(self):
        space: Dict[str, Tuple[float, float]] = {}
        triggers: Dict[str, _BayesTrigger] = {}
        for name, (param_range, trigger) in self.registered_hparams.items():
            space[name] = param_range
            triggers[name] = trigger

        return _ExceptionHandlingConcurrencyLimiter(
            _BayesOptSearchWrapper(
                space=space,
                triggers=triggers,
                metric=self.metric.name,
                mode=self.metric.mode,
                utility_kwargs={
                    "kind": self.acq_type,
                    "kappa": self.kappa,
                    "kappa_decay": self.kappa_decay,
                    "xi": self.xi,
                },
            ),
            self.max_concurrent,
        )

    def process_hparam(self, hparam: Tuple[str, Any]):
        name, hparam_obj = hparam

        if isinstance(hparam_obj, UniformHyperParam):
            self.registered_hparams[name] = ((hparam_obj.low, hparam_obj.high), None)

        elif isinstance(hparam_obj, LogUniformHyperParam):
            self.registered_hparams[name] = (
                (np.log10(hparam_obj.low), np.log10(hparam_obj.high)),
                _LogTrigger(),
            )

        elif isinstance(hparam_obj, ChoiceHyperParam):
            # check if they're all numeric
            if all(isinstance(x, (int, float)) for x in hparam_obj.choices):
                choices = sorted(hparam_obj.choices)
                self.registered_hparams[name] = (
                    (-0.5, len(choices) - 0.5001),
                    _DiscreteTrigger(choices),
                )
            else:
                raise ValueError(
                    "Non-sorted discrete values are not supported with Bayesian Optimisation"
                )

        else:
            raise ValueError("Unsupported hyperparameter for Nevergrad-backed search")

        return None
