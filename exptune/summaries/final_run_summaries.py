import itertools
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exptune.utils import FinalRunsSummarizer


def _trial_curve(train_df: pd.DataFrame, quantity: str, path: Path) -> None:
    plt.clf()
    sns.lineplot(x="training_iteration", y=quantity, data=train_df)
    plt.tight_layout()
    plt.savefig(str(path))


def _quantity_matrix(
    df: pd.DataFrame, quantities: List[str], path: Path, color_by_iteration: bool = True
) -> None:
    plt.clf()
    x, y = quantities[0], quantities[1]
    sns.scatterplot(
        x=x, y=y, hue="training_iteration" if color_by_iteration else None, data=df,
    )
    plt.tight_layout()
    plt.savefig(str(path))


def _violin(df: pd.DataFrame, quantity: str, path: Path) -> None:
    plt.clf()
    sns.violinplot(x=quantity, data=df)
    plt.tight_layout()
    plt.savefig(str(path))


class TrialCurvePlotter(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        out_dir = path / "trial_curves"
        out_dir.mkdir()
        for quant in self.quantities:
            _trial_curve(training_df, quant, out_dir / f"{quant}.png")


class TrainingQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        out_dir = path / "training_scatters"
        out_dir.mkdir()

        for x, y in itertools.combinations(self.quantities, r=2):
            _quantity_matrix(training_df, [x, y], out_dir / f"{x}_{y}.png")


class TestQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        out_dir = path / "test_scatters"
        out_dir.mkdir()

        for x, y in itertools.combinations(self.quantities, r=2):
            _quantity_matrix(
                test_df, [x, y], out_dir / f"{x}_{y}.png", color_by_iteration=False
            )


class ViolinPlotter(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        out_dir = path / "violins"
        out_dir.mkdir()

        for quant in self.quantities:
            if quant in training_df.columns:
                _violin(training_df, quant, out_dir / f"{quant}.png")

        for quant in self.quantities:
            if quant in test_df.columns:
                _violin(test_df, quant, out_dir / f"{quant}.png")


class TestMetricSummaries(FinalRunsSummarizer):
    def __call__(self, path, training_df, test_df):
        print(test_df.describe())
