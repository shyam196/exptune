import abc


class HyperParam(abc.ABCMeta):
    @property
    @abc.abstractmethod
    def default(self):
        raise NotImplementedError


class UniformHyperParam(HyperParam):
    def __init__(self, low: float, high: float, default: float):
        self.low: float = low
        self.high: float = high
        self._default: float = default

    def default(self):
        return self._default

    def __repr__(self):
        return f"Uniform[{self.low}, {self.high}  (default={self.default})]"


class LogUniformHyperParam(HyperParam):
    def __init__(self, low: float, high: float, default: float):
        self.low: float = low
        self.high: float = high
        self._default: float = default

    def default(self):
        return self._default

    def __repr__(self):
        return f"LogUniform[{self.low}, {self.high}  (default={self.default})]"
