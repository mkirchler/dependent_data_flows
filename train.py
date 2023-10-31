import yaml

from ddflow.run_backend import run_flow

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/defaults/adni.yaml", "rb"))
    run_flow(config)
