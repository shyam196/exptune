import abc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import GPUtil
import pandas as pd
from ray import tune as tune
from ray.tune import ExperimentAnalysis

EXP_CONF_KEY = "exp_conf_obj"
HPARAMS_KEY = "hparams"
PINNED_OID_KEY = "pinned_obj_ids"
DEBUG_MODE_KEY = "debug_mode"


@dataclass
class ExperimentSettings:
    exp_name: str
    exp_directory: Path
    timestamp_experiment_name: bool = True
    checkpoint_freq: int = 0
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 1
    reuse_actors: bool = False
    raise_on_failed_trial: bool = False
    max_retries: int = 3
    final_max_iterations: int = 100
    final_repeats: int = 5
    final_run_timeout: Optional[float] = None

    def __post_init__(self):
        self.exp_directory = self.exp_directory.expanduser()

    @property
    def name(self):
        if self.timestamp_experiment_name:
            return f"{self.exp_name}_{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}"
        else:
            return self.exp_name


@dataclass
class TrialResources:
    cpus: float
    gpus: float

    def as_dict(self) -> Dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus}

    def requests_gpu(self) -> bool:
        return self.gpus > 0.0


@dataclass
class Metric:
    name: str
    mode: str  # either "max" or "min"

    def __post_init__(self):
        if self.mode not in ["min", "max"]:
            raise ValueError(f"{self.mode} is not a valid setting for metric mode")


class ComposeStopper(tune.Stopper):
    def __init__(self, stoppers: List[tune.Stopper]):
        super().__init__()
        self.stoppers = stoppers

    def __call__(self, trial_id, result):
        stop = [s(trial_id, result) for s in self.stoppers]
        return any(stop)

    def stop_all(self):
        return any([s.stop_all() for s in self.stoppers])


class SearchSummarizer(abc.ABC):
    def __call__(self, search_df: pd.DataFrame) -> None:
        raise NotImplementedError


class FinalRunsSummarizer(abc.ABC):
    def __call__(self, training_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        raise NotImplementedError


def check_gpu_availability() -> bool:
    return len(GPUtil.getGPUs()) > 0


TRIMMED_COLUMNS = ["timesteps_total", "episodes_total", "timesteps_since_restore"]


def convert_experiment_analysis_to_df(analysis: ExperimentAnalysis) -> pd.DataFrame:
    search_df: pd.DataFrame = analysis.dataframe()
    trial_dfs: List[pd.DataFrame] = list(analysis.fetch_trial_dataframes().values())

    hparams_df: pd.DataFrame = pd.concat(
        (
            search_df["experiment_id"],
            search_df[f"config/{HPARAMS_KEY}"].apply(pd.Series),
        ),
        axis=1,
    )

    processed_dfs: List[pd.DataFrame] = []
    for df in trial_dfs:
        df = df.drop(columns=TRIMMED_COLUMNS)
        processed_dfs.append(pd.merge(df, hparams_df, on="experiment_id"))

    df = pd.concat(processed_dfs)
    # df = df.rename(columns={"experiment_id": "trial_id"})
    return df
