from pathlib import Path

import ray

from exptune.config_checker import check_config
from exptune.exptune import run_search
from exptune.pytorch_example import PytorchMnistMlpConfig
from exptune.train_final import train_final_models

if __name__ == "__main__":
    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    print(check_config(conf, debug_mode=True, epochs=10))

    ray.init()
    analysis = run_search(conf, debug_mode=True)
    # TODO: use search analysis to inform training of final models
    train_final_models(
        conf, {"lr": 0.01, "wd": 1e-4}, Path("~pytorch_example/run_final/").expanduser()
    )
