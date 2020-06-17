from typing import List

import pandas as pd
from ray.tune import ExperimentAnalysis

from .exptune import _HPARAMS_KEY

TRIMMED_COLUMNS = ["timesteps_total", "episodes_total", "timesteps_since_restore"]


def convert_experiment_analysis_to_df(analysis: ExperimentAnalysis) -> pd.DataFrame:
    search_df: pd.DataFrame = analysis.dataframe()
    trial_dfs: List[pd.DataFrame] = list(analysis.fetch_trial_dataframes().values())

    hparams_df: pd.DataFrame = pd.concat(
        (
            search_df["experiment_id"],
            search_df[f"config/{_HPARAMS_KEY}"].apply(pd.Series),
        ),
        axis=1,
    )

    processed_dfs: List[pd.DataFrame] = []
    for df in trial_dfs:
        df = df.drop(columns=TRIMMED_COLUMNS)
        processed_dfs.append(pd.merge(df, hparams_df, on="experiment_id"))

    df = pd.concat(processed_dfs)
    df = df.rename(columns={"experiment_id": "trial_id"})
    return df
