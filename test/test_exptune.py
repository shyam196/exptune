# Check that files can be imported
from exptune.exptune import ExperimentConfig


def test_sane() -> None:
    print(ExperimentConfig.__name__)  # shut flake8 up
    assert 1 + 1 == 2
