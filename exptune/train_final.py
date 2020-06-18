from operator import gt, lt
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import ray

from .exptune import (
    ComposeStopper,
    ExperimentConfig,
    ExperimentSettings,
    TrialResources,
)
from .utils import check_gpu_availability


@ray.remote
def _train_model(
    config: ExperimentConfig,
    trial_id: int,
    hparams: Dict[str, Any],
    results_dir: Path,
    pinned_objs: List[ray.ObjectID],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print(results_dir)

    if not check_gpu_availability() and config.resource_requirements().requests_gpu():
        raise ValueError("GPU required for training this model")

    # Seeds should be independent since each run takes place inside its own ray worker
    config.configure_seeds(trial_id)

    data: Any = config.data(pinned_objs, hparams, debug_mode=False)
    model: Any = config.model(hparams, debug_mode=False)
    optimizer: Any = config.optimizer(model, hparams, debug_mode=False)
    extra: Any = config.extra_setup(model, optimizer, hparams, debug_mode=False)
    stopper: ComposeStopper = ComposeStopper(config.stoppers())

    results: List[Dict[str, Any]] = []
    train_capture: List[Any] = []
    val_capture: List[Any] = []
    best_metric: Optional[float] = None

    metric, mode = config.trial_metric()
    cmp: Callable = lt if mode == "min" else gt

    for i in range(1, config.settings().final_max_iterations + 1):
        t_metrics, t_extra = config.train(
            model, optimizer, data, extra, debug_mode=False
        )
        train_capture.append(t_extra)

        v_metrics, v_extra = config.val(model, data, extra, debug_mode=False)
        val_capture.append(v_extra)

        if best_metric is None or cmp(v_metrics[metric], best_metric):
            best_metric = v_metrics[metric]
            config.persist_trial(results_dir, model, optimizer, hparams, extra)

        combined_metrics: Dict[str, Any] = {**t_metrics, **v_metrics}
        results.append(
            {"trial_id": trial_id, "training_iteration": i, **combined_metrics}
        )
        stop: bool = stopper(trial_id, combined_metrics)
        if stop:
            break

    # Restore model to the one where the best validation metric was recorded, and test
    model, optimizer, hparams, extra = config.restore_trial(results_dir)
    test_metrics, test_extra = config.test(model, data, extra, debug_mode=False)
    print(f"\nTrial {trial_id}:")
    pprint(test_metrics)

    test_metrics["trial_id"] = trial_id
    test_df = pd.DataFrame(test_metrics, index=["trial_id"])

    return pd.DataFrame(results), test_df


def train_final_models(
    config: ExperimentConfig, hparams: Dict[str, Any], out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Training final models")
    settings: ExperimentSettings = config.settings()
    resource_reqs: TrialResources = config.resource_requirements()
    pinned_objs: List[ray.ObjectID] = config.dataset_pin(debug_mode=False)

    if not out_dir.exists():
        out_dir.mkdir()

    runs: List[ray.TaskID] = []
    for i in range(1, settings.final_repeats + 1):
        trial_dir: Path = out_dir / f"run_{i}"
        trial_dir.mkdir()
        runs.append(
            _train_model.options(
                num_cpus=resource_reqs.cpus,
                num_gpus=resource_reqs.gpus,
                max_retries=settings.max_retries,
            ).remote(config, i, hparams, trial_dir, pinned_objs)
        )

    print("Waiting for runs...")
    results: List[Tuple[pd.DataFrame, Dict[str, Any]]] = ray.get(
        runs, timeout=settings.final_run_timeout
    )
    print("Runs finished!")

    train_df: pd.DataFrame = pd.concat([r[0] for r in results])
    test_df: pd.DataFrame = pd.concat([r[1] for r in results])

    for summarizer in config.final_runs_summaries():
        summarizer(train_df, test_df)

    print("Saving results")
    train_df.to_pickle(str(out_dir / "train_dataframe.pickle"))
    test_df.to_pickle(str(out_dir / "test_dataframe.pickle"))

    return train_df, test_df
