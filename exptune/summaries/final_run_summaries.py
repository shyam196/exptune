from typing import List

import pandas as pd
from plotly import express as px
from plotly import graph_objects as go

from exptune.summaries.plotly_utils import write_figs
from exptune.utils import FinalRunsSummarizer

_THEME = "plotly_white"


def _trial_curve(train_df: pd.DataFrame, quantity: str) -> go.Figure:
    train_df.sort_values(by=["training_iteration", "trial_id"])

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
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        figs: List[go.Figure] = []
        for quant in self.quantities:
            figs.append(_trial_curve(training_df, quant))

        write_figs(figs, path.expanduser() / f"{self.name}.html")


class TrainingQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        write_figs(
            [_quantity_matrix(training_df, self.quantities, color_by_iteration=True)],
            path.expanduser() / f"{self.name}.html",
        )


class TestQuantityScatterMatrix(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        write_figs(
            [_quantity_matrix(test_df, self.quantities, color_by_iteration=False)],
            path.expanduser() / f"{self.name}.html",
        )


class ViolinPlotter(FinalRunsSummarizer):
    def __init__(self, quantities: List[str], name: str):
        super().__init__()
        self.quantities: List[str] = quantities
        self.name = name

    def __call__(self, path, training_df, test_df):
        figs: List[go.Figure] = []

        for quant in self.quantities:
            if quant in training_df.columns:
                figs.append(_violin(training_df, quant))

        for quant in self.quantities:
            if quant in test_df.columns:
                figs.append(_violin(test_df, quant))

        write_figs(figs, path.expanduser() / f"{self.name}.html")


class TestMetricSummaries(FinalRunsSummarizer):
    def __call__(self, path, training_df, test_df):
        print(test_df.describe())
