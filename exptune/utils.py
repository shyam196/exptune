import abc
from dataclasses import dataclass
from datetime import datetime
from operator import gt, lt
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


SEARCH_DIR = "ray_search"
SEARCH_SUMMARY_DIR = "search_summaries"
FINAL_RUNS_DIR = "final_runs"
FINAL_RUNS_SUMMARY_DIR = "final_run_summaries"
SEARCH_DF_FILE = "search_df.pickle"
BEST_HYPERPARAMS_FILE = "best_hparams.pickle"
TRAIN_DF_FILE = "final_train_df.pickle"
TEST_DF_FILE = "final_test_df.pickle"


@dataclass
class ExperimentSettings:
    exp_name: str
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


class PatientStopper(tune.Stopper):
    def __init__(self, metric, patience, mode, max_iters=200, *args, **kwargs):
        if mode not in ["min", "max"]:
            raise ValueError("Invalid specification of mode, should be min or max")
        self.patience = patience
        self.metric = metric
        self.cmp = lt if mode == "min" else gt
        self.history = dict()
        self.max_iters = max_iters

    def __call__(self, trial_id, result):
        current_iteration = result["training_iteration"]
        current_metric = result[self.metric]
        if trial_id not in self.history:
            self.history[trial_id] = (current_iteration, current_metric)
            result[f"best_{self.metric}"] = current_metric
            return False
        else:
            best_iter, best_metric = self.history[trial_id]
            result[f"best_{self.metric}"] = best_metric
            if current_iteration > self.max_iters:
                return True
            if self.cmp(current_metric, best_metric):
                self.history[trial_id] = (current_iteration, current_metric)
                return False
            else:
                return (current_iteration - best_iter) > self.patience

    def stop_all(self):
        return False


class SearchSummarizer(abc.ABC):
    def __call__(self, path: Path, search_df: pd.DataFrame) -> None:
        raise NotImplementedError


class FinalRunsSummarizer(abc.ABC):
    def __call__(
        self, path: Path, training_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
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
