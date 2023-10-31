import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image

import wandb


def hard_sigmoid(x, bound=3.0, eps=1e-2):
    a = 1 / (2 * bound)
    b = a * bound
    return (a * x + b).clip(0, 1 - eps)


def plot_2d(
    samples,
    tag,
    pixels=300,
    dpi=96,
    bounds=[[-3, 3], [-3, 3]],
    bins=256,
):
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    plt.figure(figsize=(pixels / dpi, pixels / dpi), dpi=dpi)
    plt.hist2d(
        samples[..., 0], samples[..., 1], bins=bins, range=bounds, cmap="coolwarm"
    )
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    wandb.log({tag: wandb.Image(plt)})
    plt.close()


def plot_mv(
    samples,
    tag,
    num_samples=None,
    num_cols=None,
):
    samples = samples[:num_samples, :num_cols]
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    else:
        samples = samples.astype(float)
    if np.isinf(samples).any():
        print("\n\n\n sample returned inf")
        return
    if isinstance(samples, np.ndarray):
        samples = pd.DataFrame(samples)
    sns.pairplot(samples)
    wandb.log({tag: wandb.Image(plt)})
    plt.close()


def log_real_data(data_type, tl, vl, incl_inds, num_log=None, num_cols=5):
    if isinstance(vl, list):
        vl = vl[1]
    if data_type == "img":
        batch = next(iter(tl))[0] if incl_inds else next(iter(tl))
        imgs = sample_to_pil(batch[:num_log])
        wandb.log({"real_train": [wandb.Image(img) for img in imgs]})
    elif data_type == "mv":
        # x = tl.dataset.data.values
        plot_mv(tl.dataset.data.values[:num_log], "real_train", num_cols=num_cols)
        plot_mv(vl.dataset.data.values[:num_log], "real_valid", num_cols=num_cols)

    elif data_type == "2d":
        x = tl.dataset.tensors[0] if incl_inds else tl.dataset
        plot_2d(x[:num_log], "real_train")
        plot_2d(vl.dataset[:num_log], "real_valid")


def sample_to_pil(x):
    x = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    if x.shape[-1] == 1:
        x = x[:, :, :, 0]
    x = x.astype(np.uint8)
    return [Image.fromarray(a) for a in x]


def get_inv_fct(activation):
    if activation == "sigmoid":
        inv_fct = lambda x: torch.logit(torch.tensor(x))
    elif activation.startswith("hardsigmoid"):
        act_bound = float(activation.split("-")[1])
        inv_fct = lambda x: invert_hard_sigmoid(torch.tensor(x), bound=act_bound)
    else:
        raise ValueError(f"don't know activation {activation}")
    return inv_fct


def invert_hard_sigmoid(y, bound=3.0, eps=1e-9):
    """set values above/below 0/1 to -+ bound*1.1"""
    a = 1 / (2 * bound)
    b = a * bound
    x = (y - b) / a
    x[y <= eps] = -bound * 1.1
    x[y >= 1 - eps] = bound * 1.1
    return x


def count_param(model):
    return sum(p.numel() for p in model.parameters())


def get_scheduler_config(config, num_batches, stage="main"):
    scheduler_config = {
        "name": config["scheduler"],
        "opt": config["opt"] if stage == "main" else "sgd",
        "lr": config["lr"] if stage == "main" else config["lam_lr"],
        "wd": config["wd"] if stage == "main" else config["lam_wd"],
        "pct_start": config["pct_start"],
        "div_factor": config["div_factor"],
        "final_div_factor": config["final_div_factor"],
        "steplr_steps": config["steplr_steps"],
        "gamma": config["gamma"],
    }
    warmup_steps = int(num_batches * config["lr_warmup_epochs"])
    scheduler_config["warmup"] = warmup_steps

    if config["retain_lr"] and config["alternating"]:
        if stage == "main":
            total_steps = (
                config["epochs"] * num_batches
                + config["alternating_steps"]
                * num_batches
                * config["main_stage_epochs"]
            )
        else:
            total_steps = (
                config["alternating_steps"] * num_batches * config["lam_stage_epochs"]
            )
    else:
        total_steps = config["epochs"] * num_batches
    scheduler_config["total_steps"] = total_steps
    scheduler_config["step_size"] = np.ceil(
        (total_steps - warmup_steps) / scheduler_config["steplr_steps"]
    )
    scheduler_config["restarts_t0"] = int(
        np.ceil((total_steps - warmup_steps) / config["cosine_num_restarts"])
    )
    return scheduler_config
