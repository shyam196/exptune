from typing import List

import GPUtil
import pandas as pd
from ray.tune import ExperimentAnalysis

EXP_CONF_KEY = "exp_conf_obj"
HPARAMS_KEY = "hparams"
PINNED_OID_KEY = "pinned_obj_ids"
DEBUG_MODE_KEY = "debug_mode"


def check_gpu_availability() -> bool:
    return len(GPUtil.getAvailable()) > 0


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
