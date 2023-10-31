Install conda environment from `env.yaml`. If the survae module cannot be found, clone the [repo](https://github.com/didriknielsen/survae_flows) and install with `pip`.

Sweeps over seeds and hyperparameters were performed using the config yaml files found in `sweeps` and `train_sweep.py`.

Single runs can be run with `python train.py` and make use of default values provided in the yaml files in `configs`.

