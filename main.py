from exptune.config_checker import check_config
from exptune.exptune import run_search
from exptune.pytorch_example import PytorchMnistMlpConfig
from exptune.utils import convert_experiment_analysis_to_df

if __name__ == "__main__":
    conf: PytorchMnistMlpConfig = PytorchMnistMlpConfig()
    print(check_config(conf, debug_mode=True, epochs=10))
    analysis = run_search(conf, debug_mode=True)

    analysis.dataframe().to_pickle("dfs/df.pickle")

    for i, (k, df) in enumerate(analysis.fetch_trial_dataframes().items()):
        print(k)
        df.to_pickle(f"dfs/df{i}.pickle")

    print(convert_experiment_analysis_to_df(analysis))
