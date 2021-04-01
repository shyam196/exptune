import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import pandas as pd
from rich.console import Console

from exptune.exptune import ExperimentConfig, HyperParam
from exptune.utils import check_gpu_availability


def _get_default_hparams(config: ExperimentConfig) -> Dict[str, Any]:
    hparams: Dict[str, HyperParam] = config.hyperparams()
    for k, v in hparams.items():
        hparams[k] = v.default()

    # Check if any params have been fixed
    fixed_params: Dict[str, Any] = config.fixed_hyperparams()
    for k, v in fixed_params.items():
        print(f"{k} has been fixed to: {v}")
        hparams[k] = v

    return hparams


def _add_to_collected_results(
    results_dict: DefaultDict[str, List[Any]], current_results: Dict[str, Any]
) -> None:

    for k, v in current_results.items():
        results_dict[k].append(v)


def check_config(
    config: ExperimentConfig, epochs=10, check_persistence=True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    config.trial_init()
    console: Console = Console(width=120)
    if not check_gpu_availability() and config.resource_requirements().requests_gpu():
        console.log(
            "[bold red]Warning[/bold red]: GPU isn't available but config requests it; proceeding anyway"
        )

    hparams: Dict[str, Any] = _get_default_hparams(config)
    console.log("Hyperparameters:\n", hparams)
    data: Any = config.data([], hparams)
    model: Any = config.model(hparams)
    optimizer: Any = config.optimizer(model, hparams)
    extra: Any = config.extra_setup(model, optimizer, hparams)

    console.log("Data:\n", data)
    console.log("Model:\n", model)
    console.log("Optimizer:\n", optimizer)
    console.log("Extra Setup:\n", extra)

    console.log("Starting training loop")
    complete_metrics: DefaultDict[str, List[Any]] = defaultdict(list)

    try:
        for i in range(epochs):
            console.log(f"\n\nEpoch {i}")
            t_metrics, t_extra = config.train(model, optimizer, data, extra, i)
            console.log(t_metrics, t_extra)
            v_metrics, v_extra = config.val(model, data, extra, i)
            console.log(v_metrics, v_extra)

            _add_to_collected_results(
                complete_metrics, {"epoch": i, **t_metrics, **v_metrics}
            )
    except KeyboardInterrupt:
        console.log("Interrupting training...")

    console.log("\n\nTraining finished; testing...")
    test_metrics, test_extra = config.test(model, data, extra)
    console.log(test_metrics)
    console.log(test_extra)

    if check_persistence:
        console.log("\n\nTesting persistence")
        tmp_dir = Path(tempfile.mkdtemp())
        console.log(f"Saving to {tmp_dir}...")
        config.persist_trial(tmp_dir, model, optimizer, hparams, extra)
        console.log("Restoring...")
        config.restore_trial(tmp_dir)

        console.log("Deleting temporary directory...")
        shutil.rmtree(tmp_dir)

    return pd.DataFrame(complete_metrics), test_metrics
