import abc
from typing import Any, List


class HyperParam(abc.ABC):
    @abc.abstractmethod
    def default(self) -> Any:
        raise NotImplementedError


class UniformHyperParam(HyperParam):
    def __init__(self, low: float, high: float, default: float):
        super().__init__()
        self.low: float = low
        self.high: float = high
        self._default: float = default

    def default(self) -> float:
        return self._default

    def __repr__(self):
        return f"Uniform[{self.low}, {self.high}  (default={self._default})]"


class LogUniformHyperParam(HyperParam):
    def __init__(self, low: float, high: float, default: float):
        super().__init__()
        self.low: float = low
        self.high: float = high
        self._default: float = default

    def default(self) -> float:
        return self._default

    def __repr__(self):
        return f"LogUniform[{self.low}, {self.high}  (default={self._default})]"


class ChoiceHyperParam(HyperParam):
    def __init__(self, choices: List[Any], default: Any):
        super().__init__()
        self.choices = choices
        self._default = default

    def default(self) -> Any:
        return self._default

    def __repr__(self):
        return f"Choice[{self.choices}  (default={self._default})]"
