from typing import Any, Dict

import GPUtil
import pandas as pd
from rich.console import Console

from .exptune import ExperimentConfig, HyperParam


def _check_gpu_availability():
    return len(GPUtil.getAvailable()) > 0


def _get_default_hparams(config: ExperimentConfig):
    hparams: Dict[str, HyperParam] = config.hyperparams()
    for k, v in hparams.items():
        hparams[k] = v.default()

    # Check if any params have been fixed
    fixed_params: Dict[str, Any] = config.fixed_hyperparams()
    for k, v in fixed_params.items():
        print(f"{k} has been fixed to: {v}")
        hparams[k] = v

    return hparams


def check_config(config: ExperimentConfig, debug_mode: bool, epochs=10) -> pd.DataFrame:
    console: Console = Console(width=120)

    if not _check_gpu_availability() and config.resource_requirements().requests_gpu():
        console.log(
            "[bold red]Warning[/bold red]: GPU isn't available but config requests it; proceeding anyway"
        )

    hparams: Dict[str, Any] = _get_default_hparams(config)
    console.log("Hyperparameters:\n", hparams)
    data: Any = config.data([], hparams, debug_mode)
    model: Any = config.model(hparams, debug_mode)
    optimizer: Any = config.optimizer(model, hparams, debug_mode)
    extra: Any = config.extra_setup(model, optimizer, hparams, debug_mode)

    console.log("Data:\n", data)
    console.log("Model:\n", model)
    console.log("Optimizer:\n", optimizer)
    console.log("Extra Setup:\n", extra)

    console.log("Starting training loop")

    for i in range(1, epochs + 1):
        console.log(f"\n\nEpoch {i}")
        console.log(config.train(model, optimizer, data, extra, debug_mode))
        console.log(config.val(model, data, extra, debug_mode))

    console.log("\n\nTraining finished; testing...")
    console.log(config.test(model, data, extra, debug_mode))
