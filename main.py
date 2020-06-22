from pathlib import Path

import ray

from exptune.config_checker import check_config
from exptune.exptune import get_best_hyperparams, run_search
from exptune.pytorch_example import PytorchMnistMlpConfig
from exptune.train_final import train_final_models

if __name__ == "__main__":
    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    print(check_config(conf, debug_mode=True, epochs=10))

    ray.init()
    analysis = run_search(conf, debug_mode=True)
    hparams = get_best_hyperparams(analysis, conf.trial_metric())
    print("Best hyperparams: ", hparams)
    train_final_models(
        conf, hparams, Path("~/pytorch_example/run_final/").expanduser(),
    )
