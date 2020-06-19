from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import pandas as pd
from rich.console import Console

from .exptune import ExperimentConfig, HyperParam
from .utils import check_gpu_availability


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
    config: ExperimentConfig, debug_mode: bool, epochs=10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    console: Console = Console(width=120)
    if not check_gpu_availability() and config.resource_requirements().requests_gpu():
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
    complete_metrics: DefaultDict[str, List[Any]] = defaultdict(list)

    for i in range(1, epochs + 1):
        console.log(f"\n\nEpoch {i}")
        t_metrics, t_extra = config.train(model, optimizer, data, extra, debug_mode)
        console.log(t_metrics, t_extra)
        v_metrics, v_extra = config.val(model, data, extra, debug_mode)
        console.log(v_metrics, v_extra)

        _add_to_collected_results(
            complete_metrics, {"epoch": i, **t_metrics, **v_metrics}
        )

    console.log("\n\nTraining finished; testing...")
    test_metrics, test_extra = config.test(model, data, extra, debug_mode)
    console.log(test_metrics)
    console.log(test_extra)

    return pd.DataFrame(complete_metrics), test_metrics
