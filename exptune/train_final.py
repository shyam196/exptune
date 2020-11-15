import traceback
from operator import gt, lt
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from tensorboardX import SummaryWriter

from .exptune import (
    ComposeStopper,
    ExperimentConfig,
    ExperimentSettings,
    Metric,
    TrialResources,
)
from .utils import (
    FINAL_RUNS_DIR,
    FINAL_RUNS_SUMMARY_DIR,
    TEST_DF_FILE,
    TRAIN_DF_FILE,
    check_gpu_availability,
)


def _log_to_tensorboard(
    summary_writer: Optional[SummaryWriter],
    metrics: Dict[str, Any],
    training_iteration: int,
):
    if summary_writer is None:
        return

    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            summary_writer.add_scalar(k, v, training_iteration)
        elif isinstance(v, str):
            summary_writer.add_text(k, v, training_iteration)
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                summary_writer.add_scalar(k, v, training_iteration)
            else:
                summary_writer.add_histogram(k, v, training_iteration)
        else:
            print(f"WARNING: Cannot write {k} (type: {type(k)}) to Tensorboard")


@ray.remote
def _train_model(
    config: ExperimentConfig,
    trial_id: int,
    hparams: Dict[str, Any],
    results_dir: Path,
    pinned_objs: List[ray.ObjectID],
    use_tensorboard: bool,
    print_each_epoch: bool,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:

    try:
        print(results_dir)
        results_dir.mkdir(exist_ok=True)
        summary_writer: Optional[SummaryWriter] = None
        if use_tensorboard:
            summary_writer = SummaryWriter(str(results_dir / "tensorboard"))

        # Seeds should be independent since each run takes place inside its own ray worker
        config.configure_seeds(trial_id)

        data: Any = config.data(pinned_objs, hparams)
        model: Any = config.model(hparams)
        optimizer: Any = config.optimizer(model, hparams)
        extra: Any = config.extra_setup(model, optimizer, hparams)
        stopper: ComposeStopper = ComposeStopper(config.stoppers())

        results: List[Dict[str, Any]] = []
        train_capture: List[Any] = []
        val_capture: List[Any] = []
        best_metric: Optional[float] = None

        metric: Metric = config.trial_metric()
        metric_name, mode = metric.name, metric.mode
        cmp: Callable = lt if mode == "min" else gt

        for i in range(config.settings().final_max_iterations):
            t_metrics, t_extra = config.train(model, optimizer, data, extra, i)
            train_capture.append(t_extra)

            v_metrics, v_extra = config.val(model, data, extra, i)
            val_capture.append(v_extra)

            if best_metric is None or cmp(v_metrics[metric_name], best_metric):
                best_metric = v_metrics[metric_name]
                config.persist_trial(results_dir, model, optimizer, hparams, extra)

            combined_metrics: Dict[str, Any] = {
                "trial_id": trial_id,
                "training_iteration": i,
                **t_metrics,
                **v_metrics,
            }
            _log_to_tensorboard(summary_writer, combined_metrics, i)
            results.append(combined_metrics)
            if print_each_epoch:
                pprint(combined_metrics)

            stop: bool = stopper(trial_id, combined_metrics)
            if stop:
                break

        # Restore model to the one where the best validation metric was recorded, and test
        model, optimizer, hparams, extra = config.restore_trial(results_dir)
        test_metrics, test_extra = config.test(model, data, extra)
        print(f"\nTrial {trial_id}:")
        pprint(test_metrics)

        test_metrics["trial_id"] = trial_id
        test_df = pd.DataFrame(test_metrics, index=["trial_id"])

        if summary_writer is not None:
            summary_writer.close()

        return pd.DataFrame(results), test_df

    except Exception:
        print(f"Trial {trial_id} unexpectedly died!")
        traceback.print_exc()
        return None


def train_final_models(
    config: ExperimentConfig,
    hparams: Dict[str, Any],
    exp_directory: Path,
    use_tensorboard=True,
    override_repeats=None,
    print_per_epoch=True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Training final models")
    settings: ExperimentSettings = config.settings()
    resource_reqs: TrialResources = config.resource_requirements()

    if not check_gpu_availability() and resource_reqs.requests_gpu():
        raise ValueError("GPU required for training this model")

    pinned_objs: List[ray.ObjectID] = config.dataset_pin()

    out_dir = exp_directory / FINAL_RUNS_DIR
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    repeats = settings.final_repeats + 1
    if override_repeats is not None:
        repeats = override_repeats + 1

    runs: List[ray.TaskID] = []
    for i in range(1, repeats):
        trial_dir: Path = out_dir / f"run_{i}"
        runs.append(
            _train_model.options(
                num_cpus=resource_reqs.cpus,
                num_gpus=resource_reqs.gpus,
                max_retries=settings.max_retries,
            ).remote(
                config,
                i,
                hparams,
                trial_dir,
                pinned_objs,
                use_tensorboard,
                print_per_epoch,
            )
        )

    print("Waiting for runs...")
    results: List[Optional[Tuple[pd.DataFrame, pd.DataFrame]]] = ray.get(
        runs, timeout=settings.final_run_timeout
    )
    print("Runs finished!")
    results_without_fails: List[Tuple[pd.DataFrame, pd.DataFrame]] = [
        r for r in results if r is not None
    ]

    train_df: pd.DataFrame = pd.concat([r[0] for r in results_without_fails])
    test_df: pd.DataFrame = pd.concat([r[1] for r in results_without_fails])

    print("Saving results")
    train_df.to_pickle(str(exp_directory.expanduser() / TRAIN_DF_FILE))
    test_df.to_pickle(str(exp_directory.expanduser() / TEST_DF_FILE))

    summary_dir = exp_directory.expanduser() / FINAL_RUNS_SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    print("Summarizing results to", summary_dir)
    for summarizer in config.final_runs_summaries():
        try:
            summarizer(summary_dir, train_df, test_df)
        except Exception:
            print("Failed summariser", summarizer)

    return train_df, test_df
