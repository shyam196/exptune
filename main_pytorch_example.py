from pathlib import Path

import click
import ray

from exptune import check_config, run_search, train_final_models
from exptune.pytorch_example import PytorchMnistMlpConfig
from exptune.utils import dump_invocation_state


@click.command()
@click.argument(
    "out-dir",
    type=click.Path(exists=False, file_okay=False, writable=True, resolve_path=True),
)
def main(out_dir):
    experiment_dir = Path(out_dir).expanduser()
    experiment_dir.mkdir()
    print(f"Running experiment under: {experiment_dir}")
    dump_invocation_state(experiment_dir)

    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    conf.debug_mode = True

    # Check the config runs as expected
    print(check_config(conf, epochs=10), "\n\n\n")

    ray.init()
    hparams = run_search(conf, experiment_dir)
    print("Best hyperparams: ", hparams)

    train_final_models(conf, hparams, experiment_dir)


if __name__ == "__main__":
    main()
