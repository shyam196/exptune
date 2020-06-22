import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from exptune.exptune import Metric, SearchSummarizer
from exptune.hyperparams import HyperParam, LogUniformHyperParam
from exptune.summaries.plotly_utils import write_figs

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
        alpha=1e-2,
        normalize_y=True,
        n_restarts_optimizer=10,
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
        title=f"{metric.name} Surface ({xlabel}--{ylabel})",
        scene1=scene_dict,
        scene2=scene_dict,
        scene3=scene_dict,
    )

    fig.layout.template = _THEME

    return fig


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

    dimensions: Dict[str, str] = {
        metric.name: metric.name for metric in aux_metrics + [primary_metric]
    }
    for name, hparam in params.items():
        if isinstance(hparam, LogUniformHyperParam):
            dimensions[name] = f"log10({name})"
        else:
            dimensions[name] = name

    fig = px.parallel_coordinates(
        df,
        dimensions=dimensions,
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

    if isinstance(param[1], LogUniformHyperParam):
        df[param[0]] = np.log10(df[param[0]])

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-2,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=0,
    )

    gp.fit(np.array(df[param[0]])[:, np.newaxis], df[metric.name])

    x = np.linspace(df[param[0]].min(), df[param[0]].max(), 500)
    y, std = gp.predict(x.reshape(-1, 1), return_std=True)

    points = go.Scatter(
        x=df[param[0]],
        y=df[metric.name],
        mode="markers",
        name="Evaluations",
        line=dict(color="rgb(55, 55, 255)"),
    )

    upper = go.Scatter(
        x=x,
        y=y + std,
        name="GP Upper",
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor="rgba(255, 68, 68, 0.15)",
        fill="tonexty",
    )
    gp_mean = go.Scatter(
        x=x,
        y=y,
        name="GP Mean",
        mode="lines",
        line=dict(color="rgb(255, 10, 10)"),
        fillcolor="rgba(255, 68, 68, 0.15)",
        fill="tonexty",
    )
    lower = go.Scatter(
        x=x,
        y=y - std,
        name="GP Lower",
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
    )

    fig = go.Figure(data=[lower, gp_mean, upper, points])

    fig.update_layout(
        title=f"{metric.name}--{param[0]}",
        xaxis_title=f"log10({param[0]})"
        if isinstance(param[1], LogUniformHyperParam)
        else param[0],
        yaxis_title=metric.name,
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
            figs.append(_plot_surface(search_df, self.primary_metric, hparam1, hparam2))

        for hparam in self.hyperparameters.items():
            figs.append(_single_hparam_plot(search_df, self.primary_metric, hparam))

        write_figs(figs, self.out_path)
