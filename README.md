# `exptune`, A Library For Enabling Fast ML Research Iteration

This library deals with much of the boilerplate when starting a new ML research project.
By adopting this framework, you get fast model tuning, Tensorboard support and plot generation, without needing to spend time re-implement it for each project.
This library is framework agnostic.

## High Level Summary

An end-to-end example is provided in `main_pytorch_example.py` and `exptune/pytorch_example.py`.
For now, documentation is not quite complete, although I have made extensive use of type annotations which should hopefully make intent clearer.

The most important class in this library is `ExperimentConfig`.
For any experiment you should implement a configuration class that extends `ExperimentConfig`, and implement the required methods.
The most important thing to be aware of is that your config class should not contain any _trial-specific_ state.
Global state about the experiment configuration is fine, but otherwise your config should be "pure": it is a bad idea to alter state outside of `__init__`.
If you want to have per-trial state, you can put it in the `extra` object that is returned by the `extra_setup` method per trial and passed around.

For train/val/test you need to return a tuple: the first is some trial metrics, and the second is an optional object that will be logged as a pickle.
An example config can be found in `exptune/pytorch_example.py`.

### Checking a Config

Often you just want to check that a config doesn't crash, and does what you expect: you may be testing it on your local machine before committing and submitting to your cluster.
This is where the `check_config` function comes in.
This method will take a config and run the train/validation loop, and pretty-print out useful debugging information.
It can also check that your model's save and load functions do not crash.
For larger experiments you may find it useful to take advantage of the `debug_mode` field on configs.
You can set this manually, and use it inside the config to build smaller models, or use only a subset of the dataset, etc.

### Running a Search

If you are unsure what the hyperparameters should be for your model, override the `hyperparams` method and return a dict of `HyperParam` objects; these can be overidden using the `fixed_hyperparams` method, which is useful for ablation studies.
In combination with a search strategy (e.g. random, grid, bayesian search) and a trial scheduler from Ray (e.g. one that can kill bad trials early) you can quickly set up experiments to find unknown parameters using the `run_search` function.
Since this is a thin layer over Ray Tune, you get benefits of that library, such as automatic writing to Tensorboard.

### Training a Final Set of Models

Once you are aware of what the hyperparameters should be, you can train a set of final models using `train_final_models`.
This will also write to Tensorboard, and generate any plots necessary.

## Installing and Development

You can install this by running pip.
If you want to make changes use the `-e` flag to install in editable mode.
Ensure you run `initial_setup.sh` to get the repo initialised after cloning if you want to upstream commits.

## Roadmap
- [ ] Better documentation + generation.
- [ ] Examples for different frameworks
- [ ] Useful visualizations for search summarization
