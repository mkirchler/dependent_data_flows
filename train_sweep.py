import yaml

import wandb
from ddflow.run_backend import run_flow


def update_configs(sweep, defaults):
    for key in sweep.keys():
        if key in defaults["data_param"]:
            defaults["data_param"][key] = sweep[key]
            if key == "seed":
                defaults[key] = sweep[key]
        else:
            defaults[key] = sweep[key]

    return defaults


if __name__ == "__main__":
    run = wandb.init()
    sweep_config = wandb.config
    defaults = yaml.safe_load(open(sweep_config["default_configs"]))
    run_config = update_configs(sweep_config, defaults)
    run_flow(run_config)
