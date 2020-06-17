from exptune.config_checker import check_config
from exptune.exptune import run_search
from exptune.pytorch_example import PytorchMnistMlpConfig

if __name__ == "__main__":
    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    print(check_config(conf, debug_mode=True, epochs=10))
    run_search(conf, debug_mode=True)
