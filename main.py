from exptune.config_checker import check_config
from exptune.pytorch_example import PytorchMnistMlpConfig

if __name__ == "__main__":
    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    check_config(conf, debug_mode=True, epochs=10)
