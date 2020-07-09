import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from exptune.hyperparams import ChoiceHyperParam, HyperParam, LogUniformHyperParam
from exptune.summaries.plotly_utils import write_figs
from exptune.utils import Metric, SearchSummarizer

_THEME = "plotly_white"


def _plot_surface(
    search_df: pd.DataFrame,
    metric: Metric,
    param1: Tuple[str, HyperParam],
    param2: Tuple[str, HyperParam],
):
    grouped = search_df.groupby("trial_id")[[metric.name, param1[0], param2[0]]]
    if metric.mode == "min":
        df = grouped.min()
    else:
        df = grouped.max()

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-1,
        normalize_y=True,
        n_restarts_optimizer=50,
        random_state=0,
    )

    if isinstance(param1[1], LogUniformHyperParam):
        df[param1[0]] = np.log10(df[param1[0]])

    if isinstance(param2[1], LogUniformHyperParam):
        df[param2[0]] = np.log10(df[param2[0]])

    x = np.array([df[param1[0]], df[param2[0]]])

    gp.fit(x.T, df[metric.name])

    c1 = np.linspace(df[param1[0]].min(), df[param1[0]].max(), 100)
    c2 = np.linspace(df[param2[0]].min(), df[param2[0]].max(), 100)
    xx, yy = np.meshgrid(c1, c2)
    y_mean, y_std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)

    if metric.mode == "min":
        min_color = y_mean.min()
        max_color = np.percentile(y_mean, 50)
    else:
        min_color = np.percentile(y_mean, 50)
        max_color = y_mean.max()

    # This works really well for viewing the loss surface as a function of two hyperparameters
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
    )
    mean_surface = go.Surface(
        x=c1,
        y=c2,
        z=np.reshape(y_mean, (100, 100)),
        colorscale=px.colors.diverging.Tealrose,
        cmin=min_color,
        cmax=max_color,
        name="GP Mean",
    )

    std_surface = go.Surface(
        x=c1,
        y=c2,
        z=np.reshape(y_std, (100, 100)),
        showscale=False,
        colorscale=px.colors.diverging.Tealrose,
        name="GP Standard Deviation",
    )

    scatter = go.Scatter3d(
        x=df[param1[0]],
        y=df[param2[0]],
        z=df[metric.name],
        mode="markers",
        name="Evaluation",
    )

    fig.add_trace(mean_surface, row=1, col=1)
    fig.add_trace(std_surface, row=1, col=2)
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.add_trace(scatter, row=1, col=1)

    xlabel: str = param1[0]
    if isinstance(param1[1], LogUniformHyperParam):
        xlabel = "log10(" + xlabel + ")"
    ylabel: str = param2[0]
    if isinstance(param2[1], LogUniformHyperParam):
        ylabel = "log10(" + ylabel + ")"

    scene_dict: Dict[str, Dict[str, Any]] = dict(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        zaxis=dict(title=metric.name),
    )

    fig.update_layout(
        title=f"{metric.name} Surface ({xlabel} - {ylabel})",
        scene1=scene_dict,
        scene2=scene_dict,
        scene3=scene_dict,
    )

    fig.layout.template = _THEME

    return fig


def _plot_single_categorical(
    search_df: pd.DataFrame,
    metric: Metric,
    param1: Tuple[str, HyperParam],
    param2: Tuple[str, HyperParam],
) -> go.Figure:
    grouped = search_df.groupby("trial_id")[[metric.name, param1[0], param2[0]]]
    if metric.mode == "min":
        df = grouped.min()
    else:
        df = grouped.max()

    metric_column = df[metric.name]
    if metric.mode == "min":
        min_color = metric_column.min()
        max_color = np.percentile(metric_column, 50)
    else:
        min_color = np.percentile(metric_column, 50)
        max_color = metric_column.max()

    fig = px.scatter(
        df,
        x=param1[0],
        y=param2[0],
        color=metric.name,
        color_continuous_scale=px.colors.diverging.Tealrose,
        range_color=(min_color, max_color),
        hover_data=df.columns,
        log_x=isinstance(param1[1], LogUniformHyperParam),
        log_y=isinstance(param2[1], LogUniformHyperParam),
        title=f"{metric.name} Categorical Cross Scatter ({param1[0]} - {param2[0]})",
    )
    fig.layout.template = _THEME

    return fig


def _plot_double_categorical(
    search_df: pd.DataFrame,
    metric: Metric,
    param1: Tuple[str, HyperParam],
    param2: Tuple[str, HyperParam],
) -> go.Figure:
    grouped = search_df.groupby("trial_id")[[metric.name, param1[0], param2[0]]]
    if metric.mode == "min":
        df = grouped.min()
    else:
        df = grouped.max()

    metric_column = df[metric.name]
    if metric.mode == "min":
        min_color = metric_column.min()
        max_color = np.percentile(metric_column, 50)
    else:
        min_color = np.percentile(metric_column, 50)
        max_color = metric_column.max()

    # Handle duplicated values
    grouped = df.groupby([param1[0], param2[0]])
    if metric.mode == "min":
        df = grouped.min()
    else:
        df = grouped.max()

    df = df.reset_index()

    fig = px.scatter(
        df,
        x=param1[0],
        y=param2[0],
        size=metric.name,
        color=metric.name,
        color_continuous_scale=px.colors.diverging.Tealrose,
        range_color=(min_color, max_color),
        hover_data=df.columns,
        title=f"{metric.name} Categorical Cross Scatter ({param1[0]} - {param2[0]})",
    )
    fig.layout.template = _THEME
    return fig


def _count_categorical(
    param1: Tuple[str, HyperParam], param2: Tuple[str, HyperParam]
) -> int:
    num: int = 0
    if isinstance(param1[1], ChoiceHyperParam):
        num += 1

    if isinstance(param2[1], ChoiceHyperParam):
        num += 1

    return num


def _plot_parallel(
    search_df: pd.DataFrame,
    primary_metric: Metric,
    aux_metrics: List[Metric],
    params: Dict[str, HyperParam],
):
    dfs: List[pd.DataFrame] = []
    for metric in aux_metrics + [primary_metric]:
        grouped = search_df.groupby("trial_id")[list(params.keys()) + [metric.name]]
        if metric.mode == "min":
            df = grouped.min()
        else:
            df = grouped.max()

        dfs.append(df)

    df = dfs[0]
    for next_df in dfs[1:]:
        df = pd.merge(df, next_df, on=list(params.keys()))

    for name, hparam in params.items():
        if isinstance(hparam, LogUniformHyperParam):
            df[name] = np.log10(df[name])

    if primary_metric.mode == "min":
        min_color = df[primary_metric.name].min()
        max_color = np.percentile(df[primary_metric.name], 50)
    else:
        min_color = np.percentile(df[primary_metric.name], 50)
        max_color = df[primary_metric.name].max()

    labels: Dict[str, str] = {
        metric.name: metric.name for metric in aux_metrics + [primary_metric]
    }
    for name, hparam in params.items():
        if isinstance(hparam, LogUniformHyperParam):
            labels[name] = f"log10({name})"
        else:
            labels[name] = name

    fig = px.parallel_coordinates(
        df,
        labels=labels,
        color=primary_metric.name,
        color_continuous_scale=px.colors.diverging.Tealrose,
        range_color=(min_color, max_color),
    )

    fig.layout.template = _THEME
    return fig


def _single_hparam_plot(
    search_df: pd.DataFrame, metric: Metric, param: Tuple[str, HyperParam]
):
    grouped = search_df.groupby("trial_id")[[metric.name, param[0]]]
    if metric.mode == "min":
        df = grouped.min()
    else:
        df = grouped.max()

    fig = px.scatter(
        df,
        x=param[0],
        y=metric.name,
        log_x=isinstance(param[1], LogUniformHyperParam),
        trendline="lowess",
        hover_data=df.columns,
        title=f"{metric.name} - {param[0]}",
    )

    fig.layout.template = _THEME

    return fig


class HyperParamReport(SearchSummarizer):
    def __init__(
        self,
        hyperparameters: Dict[str, HyperParam],
        primary_metric: Metric,
        aux_metrics: List[Metric],
        out_path: Path,
    ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.primary_metric = primary_metric
        self.aux_metrics = aux_metrics
        self.out_path = out_path

    def __call__(self, search_df: pd.DataFrame) -> None:
        figs: List[go.Figure] = []

        figs.append(
            _plot_parallel(
                search_df, self.primary_metric, self.aux_metrics, self.hyperparameters
            )
        )

        for hparam1, hparam2 in itertools.combinations(self.hyperparameters.items(), 2):
            num_categorical: int = _count_categorical(hparam1, hparam2)
            if num_categorical == 0:
                figs.append(
                    _plot_surface(search_df, self.primary_metric, hparam1, hparam2)
                )
            elif num_categorical == 1:
                figs.append(
                    _plot_single_categorical(
                        search_df, self.primary_metric, hparam1, hparam2
                    )
                )

            elif num_categorical == 2:
                figs.append(
                    _plot_double_categorical(
                        search_df, self.primary_metric, hparam1, hparam2
                    )
                )

        for hparam in self.hyperparameters.items():
            if isinstance(hparam[1], ChoiceHyperParam):
                continue
            figs.append(_single_hparam_plot(search_df, self.primary_metric, hparam))

        write_figs(figs, self.out_path)
