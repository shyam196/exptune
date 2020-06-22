from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from exptune.exptune import FinalRunsSummarizer
from exptune.summaries.plotly_utils import write_figs

_THEME = "plotly_white"


def _trial_curve(train_df: pd.DataFrame, quantity: str) -> go.Figure:
    mean = train_df.groupby("training_iteration").mean()
    std = train_df.groupby("training_iteration").std()

    m = mean[quantity]
    s = std[quantity]

    upper = go.Scatter(
        x=train_df["training_iteration"],
        y=m + s,
        name="Upper (std)",
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor="rgba(255, 68, 68, 0.15)",
        fill="tonexty",
    )
    mean = go.Scatter(
        x=train_df["training_iteration"],
        y=m,
        name="Mean",
        mode="lines",
        line=dict(color="rgb(255, 10, 10)"),
        fillcolor="rgba(255, 68, 68, 0.15)",
        fill="tonexty",
    )
    lower = go.Scatter(
        x=train_df["training_iteration"],
        y=m - s,
        name="Lower (std)",
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
    )

    fig = go.Figure(data=[lower, mean, upper])
    fig.update_layout(
        xaxis_title="training_iteration", yaxis_title=quantity,
    )

    fig.layout.template = _THEME

    return fig


def _quantity_matrix(
    df: pd.DataFrame, quantities: List[str], color_by_iteration: bool = True
) -> go.Figure:
    fig = px.scatter_matrix(
        df,
        dimensions=quantities,
        color="training_iteration" if color_by_iteration else None,
        hover_data=df.columns,
    )

    fig.layout.template = _THEME
    return fig


def _violin(df: pd.DataFrame, quantity: str) -> go.Figure:
    fig = px.violin(df, x=quantity, box=True, points="all", hover_data=df.columns)
    fig.layout.template = _THEME
    return fig


class TrialCurvePlotter(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], out_path: Path):
        super().__init__()
        self.quantities: List[str] = quantities
        self.out_path: Path = out_path.expanduser()

    def __call__(self, training_df, test_df):
        figs: List[go.Figure] = []
        for quant in self.quantities:
            figs.append(_trial_curve(training_df, quant))

        write_figs(figs, self.out_path)


class TrainingQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], out_path: Path):
        super().__init__()
        self.quantities: List[str] = quantities
        self.out_path: Path = out_path.expanduser()

    def __call__(self, training_df, test_df):
        write_figs(
            [_quantity_matrix(training_df, self.quantities, color_by_iteration=True)],
            self.out_path,
        )


class TestQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], out_path: Path):
        super().__init__()
        self.quantities: List[str] = quantities
        self.out_path: Path = out_path.expanduser()

    def __call__(self, training_df, test_df):
        write_figs(
            [_quantity_matrix(test_df, self.quantities, color_by_iteration=False)],
            self.out_path,
        )


class ViolinPlotter(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], out_path: Path):
        super().__init__()
        self.quantities: List[str] = quantities
        self.out_path: Path = out_path.expanduser()

    def __call__(self, training_df, test_df):
        figs: List[go.Figure] = []

        for quant in self.quantities:
            if quant in training_df.columns:
                figs.append(_violin(training_df, quant))

        for quant in self.quantities:
            if quant in test_df.columns:
                figs.append(_violin(test_df, quant))

        write_figs(figs, self.out_path)


class TestMetricSummaries(FinalRunsSummarizer):
    def __call__(self, training_df, test_df):
        print(test_df.describe())
