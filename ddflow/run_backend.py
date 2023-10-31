import shutil
import uuid
from functools import partial
from glob import glob
from os.path import join

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from rich.progress import track
from torch.utils.data import DataLoader

import wandb
from ddflow.baseline_models import (
    get_affine_dependent_flow,
    get_affine_independent_flow,
    get_spline_dependent_flow,
    setup_baseline_img_flow,
)
from ddflow.callbacks import LogSampleData, TrainingStageController
from ddflow.dependent_distributions import CovMixtureMVN, EquiDependentUnitMVN
from ddflow.mixed_data import (
    FWRotatedData,
    get_adni_data,
    get_cov_dataset,
    get_pareto_rho_dataset,
    get_stock_pair_data,
    get_ukb_biomarker_data,
)
from ddflow.mixed_trainer import (
    DependentPLFlow,
    LamPLFlow,
    LargeBatchDependentPLFlow,
    PLFlow,
)
from ddflow.utils import count_param, get_inv_fct, get_scheduler_config, log_real_data

CHECKPOINT_DIR = "checkpoints"
torch.set_num_threads(1)


def get_data(data_config):
    dset = data_config.pop("dset")
    if dset.startswith("pareto"):
        dd = dset.split("-")[1]
        if "abl" in dset:
            data_config["rho_min"], data_config["rho_max"] = data_config["rho_min_max"]
        tl, vl, ttl, meta = get_pareto_rho_dataset(dset=dd, **data_config)
    elif dset == "adni":
        tl, vl, ttl, meta = get_adni_data(**data_config)
    elif dset.startswith("cov"):
        dd = dset.split("-")[1]
        tl, vl, ttl, meta = get_cov_dataset(dset=dd, **data_config)
    elif dset == "stock_pair":
        tl, vl, ttl, meta = get_stock_pair_data(**data_config)
    elif dset == "ukb_biomarker":
        tl, vl, ttl, meta = get_ukb_biomarker_data(**data_config)
    else:
        raise NotImplementedError(f"don't recognize dset {dset}")
    return tl, vl, ttl, meta


SPLINE_FT = [
    "linear",
    "quadratic",
    "cubic",
    "rational_quadratic",
    "unconstrained_rational_quadratic",
]

# TODO: refactor!
def create_flow(
    data_meta,
    config,
    overwrite_default_param=None,
):
    ft = config["flow_type"]
    print(config["experiment"], ft)
    dp = (
        overwrite_default_param
        if not (overwrite_default_param is None)
        else config["default_param"]
    )
    if config["experiment"] == "equi" and ft in ["affine"] + SPLINE_FT:
        if dp == "from_data":
            default_param = data_meta["rhos"]
        elif isinstance(dp, (float, int)):
            default_param = dp
        default_param_raw = get_inv_fct(config["param_activation"])(default_param)
        base_dist = EquiDependentUnitMVN(
            loc=torch.zeros(data_meta["D"]),
            ind_blocks=data_meta["ind_blocks"],
            default_rho=default_param_raw,
            opt_params=config["opt_params"],
            rho_activation=config["param_activation"],
        )
        if ft in SPLINE_FT:
            flow = get_spline_dependent_flow(
                base_dist=base_dist,
                D=data_meta["D"],
                num_layers=config["num_layers"],
                num_bins=config["num_bins"],
                hidden_units=config["hidden_units"],
                activation=config["activation"],
                shuffle=config["shuffle"],
                spline_type=ft,
                bound=config["bound"] if "bound" in config else 3.0,
            )
        else:
            flow = get_affine_dependent_flow(
                base_dist=base_dist,
                D=data_meta["D"],
                hidden_units=config["hidden_units"],
                actnorm=config["actnorm"],
                num_layers=config["num_layers"],
                scale=config["scale"] if "scale" in config else "exp",
                activation=config["activation"],
                shuffle=config["shuffle"],
            )
    elif config["experiment"] == "equi" and config["flow_type"] == "img":
        default_param_raw = get_inv_fct(config["param_activation"])(dp)
        base_dist = EquiDependentUnitMVN(
            loc=torch.zeros(data_meta["D"]),
            ind_blocks=data_meta["ind_blocks"],
            default_rho=default_param_raw,
            opt_params=config["opt_params"],
            rho_activation=config["param_activation"],
        )
        flow = setup_baseline_img_flow(
            base_dist=base_dist,
            data_shape=data_meta["img_shape"],
            num_scales=config["num_scales"],
            num_steps=config["num_steps"],
            dequant="uniform",
            is_mixed=True,
        )
    elif config["experiment"] == "rm" and ft in ["affine"] + SPLINE_FT:
        if dp == "from_data":
            default_param = data_meta["lam"]
        elif isinstance(dp, float):
            default_param = dp
        elif isinstance(dp, int):
            default_param = float(dp)
        default_param_raw = get_inv_fct(config["param_activation"])(default_param)
        base_dist = CovMixtureMVN(
            loc=torch.zeros(data_meta["D"]),
            relationship_matrix=data_meta["cov"],
            spectral_decomp=data_meta["spectral"],
            default_lam=default_param_raw,
            opt_params=config["opt_params"],
            lam_activation=config["param_activation"],
        )
        if ft in SPLINE_FT:
            flow = get_spline_dependent_flow(
                base_dist=base_dist,
                D=data_meta["D"],
                num_layers=config["num_layers"],
                num_bins=config["num_bins"],
                hidden_units=config["hidden_units"],
                activation=config["activation"],
                shuffle=config["shuffle"],
                spline_type=ft,
                bound=config["bound"] if "bound" in config else 3.0,
            )
        else:
            flow = get_affine_dependent_flow(
                base_dist=base_dist,
                D=data_meta["D"],
                hidden_units=config["hidden_units"],
                actnorm=config["actnorm"],
                num_layers=config["num_layers"],
                scale=config["scale"] if "scale" in config else "exp",
                activation=config["activation"],
                shuffle=config["shuffle"],
            )
    elif config["experiment"] == "rm" and config["flow_type"] == "img":
        default_param_raw = get_inv_fct(config["param_activation"])(dp)
        base_dist = CovMixtureMVN(
            loc=torch.zeros(data_meta["D"]),
            relationship_matrix=data_meta["cov"],
            spectral_decomp=data_meta["spectral"],
            default_lam=default_param_raw,
            opt_params=config["opt_params"],
            lam_activation=config["param_activation"],
        )
        flow = setup_baseline_img_flow(
            base_dist=base_dist,
            data_shape=data_meta["img_shape"],
            num_scales=config["num_scales"],
            num_steps=config["num_steps"],
            dequant="uniform",
            is_mixed=True,
        )

    # ignored -- use default_param = 0 // 1 instead
    elif config["experiment"] == "baseline" and config["flow_type"] == "affine":
        flow = get_affine_independent_flow(
            D=data_meta["D"],
            hidden_units=config["hidden_units"],
            actnorm=config["actnorm"],
            num_layers=config["num_layers"],
            scale=config["scale"] if "scale" in config else "exp",
        )

    else:
        raise NotImplementedError()
    return flow


def setup_callbacks(config, data_meta, logger, uid, step=0, multi_vl=True):
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(pl.callbacks.RichProgressBar(leave=True))
    if "fine_tune" in config and config["fine_tune"] and config["alternating"] == False:
        config["alternating_steps"] = 1
    callbacks.append(
        TrainingStageController(
            first_stage_epochs=config["epochs"],
            lam_stage_epochs=config["lam_stage_epochs"],
            main_stage_epochs=config["main_stage_epochs"],
            num_alternating=config["alternating_steps"],
        )
    )
    callbacks.append(
        LogSampleData(
            log_every_n_epochs=config["log_sample_freq"],
            num_samples=config["num_log"],
            data_type=data_meta["data_type"],
        )
    )
    if "gpustats" in config["add_callbacks"] and logger and config["gpus"] != 0:
        # TODO: update this once the pl implementation of DeviceStatsMonitor changes - currently it's unusable
        # callbacks.append(pl.callbacks.DeviceStatsMonitor())
        callbacks.append(
            pl.callbacks.GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
        )
    if "modelsummary" in config["add_callbacks"]:
        callbacks.append(pl.callbacks.RichModelSummary(max_depth=3))
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=join(CHECKPOINT_DIR, uid),
            save_top_k=1,
            monitor="valid_loss/dataloader_idx_0" if multi_vl else "valid_loss",
            verbose=True,
            filename=f"astep={step}_"
            + "epoch={epoch}_{valid_loss/dataloader_idx_0:.3f}"
            if multi_vl
            else f"astep={step}_" + "epoch={epoch}_{valid_loss:.3f}",
            auto_insert_metric_name=False,
        )
    )
    return callbacks


def run_flow(
    config,
):
    if "num_threads" in config:
        torch.set_num_threads(config["num_threads"])
    uid = str(uuid.uuid4())

    config["data_param"]["incl_inds"] = config["experiment"] != "baseline"
    print(config)

    pl.seed_everything(seed=config["seed"], workers=True)
    if config["flow_type"] in SPLINE_FT:
        config["data_param"]["rescale_01"] = True
    tl, vl, ttl, data_meta = get_data(config["data_param"])
    scheduler_config = get_scheduler_config(config, len(tl))
    print(f"train loader n: {len(tl)}")
    print(scheduler_config)
    multi_vl = isinstance(vl, (list, tuple)) and len(vl) > 1

    if config["use_wandb"]:
        logger = WandbLogger(
            project=config["project"] if "project" in config else "ddflow_debug"
        )
        params = {
            "uid": uid,
            **config,
        }
        logger.log_hyperparams(params)
        if config["log_sample_freq"] > 0:
            log_real_data(
                data_meta["data_type"],
                tl,
                vl,
                config["data_param"]["incl_inds"],
                num_log=config["num_log"],
                num_cols=data_meta["num_cols"] if "num_cols" in data_meta else 5,
            )
    else:
        logger = None
    callbacks = setup_callbacks(
        config, data_meta, logger, uid, step=0, multi_vl=multi_vl
    )

    overwrite_default_param = (
        config["init_default_param"] if "init_default_param" in config else None
    )
    flow = create_flow(
        config=config,
        data_meta=data_meta,
        overwrite_default_param=overwrite_default_param,
    )
    fine_tune = "fine_tune" in config and config["fine_tune"]
    if fine_tune:
        for param in flow.base_dist.parameters():
            param.requires_grad_(False)
    if logger:
        logger.log_hyperparams({"num_params": count_param(flow)})

    last_lr, cmin_losses = run_main_stage(
        flow, config, tl, vl, scheduler_config, callbacks, logger
    )
    if not overwrite_default_param is None:
        if config["experiment"] == "equi":
            with torch.no_grad():
                flow.base_dist.rhos[:] = get_inv_fct(config["param_activation"])(0.0)
        elif config["experiment"] == "rm":
            with torch.no_grad():
                flow.base_dist.lam.set_(get_inv_fct(config["param_activation"])(0.0))

    if config["alternating"]:
        for step in range(config["alternating_steps"]):
            if config["alternating"] == "lam" and config["lam_stage_epochs"] > 0:
                for param in flow.base_dist.parameters():
                    param.requires_grad_(True)

                lam_lr = run_lam_stage(
                    flow,
                    config,
                    tl.dataset,
                    callbacks,
                    logger,
                    lr=None if step == 0 else lam_lr,
                )
            else:
                raise NotImplementedError(config["alternating"])

            callbacks = setup_callbacks(
                config, data_meta, logger, uid, step=step + 1, multi_vl=multi_vl
            )
            last_lr, cmin_losses = run_main_stage(
                flow,
                config,
                tl,
                vl,
                scheduler_config,
                callbacks,
                logger,
                first=False,
                lr=last_lr,
                cmin_losses=cmin_losses,
            )
    if fine_tune and not config["alternating"]:
        if config["opt_params"]:
            for param in flow.base_dist.parameters():
                param.requires_grad_(True)
        with torch.no_grad():
            if config["experiment"] == "equi":
                flow.base_dist.rhos[:] = get_inv_fct(config["param_activation"])(
                    config["default_param"]
                )
            elif config["experiment"] == "rm":
                flow.base_dist.lam.set_(
                    get_inv_fct(config["param_activation"])(config["default_param"])
                )
        step = 0
        callbacks = setup_callbacks(
            config, data_meta, logger, uid, step=step + 1, multi_vl=multi_vl
        )
        last_lr, cmin_losses = run_main_stage(
            flow,
            config,
            tl,
            vl,
            scheduler_config,
            callbacks,
            logger,
            first=False,
            lr=last_lr,
            cmin_losses=cmin_losses,
        )

    dev = "cuda:0" if config["gpus"] > 0 else "cpu"
    eval_model(
        flow,
        vl,
        ttl,
        config,
        uid,
        dev=dev,
    )

    if "check_param_fit" in config and config["check_param_fit"]:
        if config["experiment"] == "equi":
            true_rhos = np.array(data_meta["rhos"])
            estimated_rhos = flow.base_dist.get_rhos().detach().numpy()
            mse = ((true_rhos - estimated_rhos) ** 2).mean()
            mae = np.abs(true_rhos - estimated_rhos).mean()
            wandb.run.summary["param_fit_rho_mse"] = mse
            wandb.run.summary["param_fit_rho_mae"] = mae
        elif config["experiment"] == "rm":
            true_lam = data_meta["lam"]
            estimated_lam = flow.base_dist.get_lam().detach().item()
            diff = abs(true_lam - estimated_lam)
            wandb.run.summary["param_fit_lam_abs"] = diff

    if "skip_final_eval" in config and config["skip_final_eval"]:
        return flow, (tl, vl, ttl, data_meta)
    one = get_inv_fct(config["param_activation"])(torch.tensor(1.0))
    for (name, dl) in [("val", vl), ("test", ttl)]:
        if isinstance(flow.base_dist, CovMixtureMVN):
            lam = flow.base_dist.get_lam().item()
            cov = data_meta[f"{name}_cov"]
            spectral = data_meta[f"{name}_spectral"]
            lam_raw = get_inv_fct(config["param_activation"])(lam)
        else:
            rhos = flow.base_dist.get_rhos()
            mean_rho = rhos.mean()
            lam = 1 - mean_rho
            wandb.run.summary["final_rho_mean"] = mean_rho.item()
            lam_raw = get_inv_fct(config["param_activation"])(lam)
            ind_blocks = dl.dataset.get_ind_blocks()
            ns = [len(block) for block in ind_blocks]
            rel_matrix = scipy.linalg.block_diag(
                *[np.ones((n, n), dtype=np.float32) for n in ns]
            )
            spectral = torch.linalg.eigh(torch.from_numpy(rel_matrix))
            spectral = [spectral.eigenvalues, spectral.eigenvectors]
            cov = None

        res = eval_with_cov(
            flow,
            dl,
            cov=cov,
            spectral=spectral,
            dev=dev,
            lams=[lam_raw, one],
        )
        for lam, nll in res.items():
            print(lam_raw, one, lam, res)
            x = {lam_raw: "res", one: 1}[lam]
            metric_name = f"{name}_finaldep_lam{x}"
            wandb.run.summary[metric_name] = nll.item()
    return flow, (tl, vl, ttl, data_meta)


@torch.no_grad()
def eval_model(flow, vl, ttl, config, uid, dev="cpu"):
    checkpoints = glob(join(CHECKPOINT_DIR, uid, "*.ckpt"))
    losses = [float(s.split("_")[-1].split(".ckpt")[0]) for s in checkpoints]
    ckpt_pth = checkpoints[np.argmin(losses)]
    print(f"selecting checkpoint {ckpt_pth}")

    pl_module_class = dict(
        equi=DependentPLFlow,
        rm=DependentPLFlow,
        baseline=PLFlow,
    )[config["experiment"]]
    module = pl_module_class(flow=flow)
    module.load_from_checkpoint(
        ckpt_pth,
        map_location=dev,
        flow=flow,
        opt_mode="joint" if config["experiment"] == "equi" else "flow",
    )
    flow.to(dev)
    if hasattr(flow.base_dist, "rhos"):
        with torch.no_grad():
            r = flow.base_dist.get_rhos()
        wandb.log(
            {
                "min_rho": r.min(),
                "max_rho": r.max(),
                "mean_rho": r.mean(),
                "median_rho": r.median(),
            }
        )
    elif hasattr(flow.base_dist, "lam"):
        with torch.no_grad():
            l = flow.base_dist.get_lam()
        wandb.log({"lam": l})

    for name, dl in [("valid", vl), ("test", ttl)]:
        if dl is None:
            continue
        if not isinstance(dl, (list, tuple)):
            dl = [dl]
        for i, loader in enumerate(dl):
            nlls = []
            for batch in track(loader, total=len(loader)):
                nll = -flow.log_prob(batch.to(dev))
                nlls.append(nll.cpu())
            nlls = torch.cat(nlls)
            wandb.run.summary[f"final_{name}_nll_dl{i}"] = nlls.mean()

    # purge checkpoints to avoid clutter:
    shutil.rmtree(join(CHECKPOINT_DIR, uid))


@torch.no_grad()
def eval_with_cov(flow, dl, cov=None, spectral=None, dev="cpu", lams=[0.0]):
    assert (
        cov is None or spectral is None and not (spectral is None and cov is None)
    ), "need to provide *either* spectral decomposition *or* relationship matrix"
    inds = torch.tensor(list(dl.sampler))
    assert (inds == torch.arange(len(dl.sampler))).all()
    noises = []
    ldjs = []
    for batch in track(dl):
        noise, ldj = flow.data2noise(batch.to(dev), with_ldj=True)
        noises.append(noise.cpu())
        ldjs.append(ldj.cpu())
    noises = torch.cat(noises)
    ldjs = torch.cat(ldjs)
    results = dict()
    for lam in lams:
        C = CovMixtureMVN(
            loc=torch.zeros(noises.shape[1]),
            relationship_matrix=cov,
            spectral_decomp=spectral,
            default_lam=lam,
            opt_params=False,
        )
        base_ll = C.log_prob_mini_batch(noises, inds)
        results[lam] = -(base_ll + ldjs).mean()

    return results


def run_main_stage(
    flow,
    config,
    tl,
    vl,
    scheduler_config,
    callbacks,
    logger,
    first=True,
    lr=None,
    cmin_losses=None,
):
    if logger:
        logger.log_metrics({"training_stage": "first" if first else "main"})
    if config["large_batch"]:
        pl_module_class = partial(
            LargeBatchDependentPLFlow,
            soft_max_size=config["soft_max_size"],
            hard_max_size=config["hard_max_size"],
            clip_type=config["clip_type"],
            grad_clip=config["grad_clip"],
        )
    else:
        pl_module_class = dict(
            equi=DependentPLFlow,
            rm=DependentPLFlow,
            baseline=PLFlow,
        )[config["experiment"]]

    module = pl_module_class(
        flow,
        lr=config["lr"] if lr is None else lr,
        scheduler_config=scheduler_config,
        cmin_losses=cmin_losses,
        opt_mode="joint" if config["experiment"] == "equi" else "flow",
    )
    trainer = pl.Trainer(
        gpus=config["gpus"],
        strategy=config["strategy"],
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config["grad_clip"],
        gradient_clip_algorithm=config["clip_type"],
        max_epochs=config["epochs"] if first else config["main_stage_epochs"],
        deterministic=True,
    )
    trainer.fit(module, tl, vl)
    lr = module.lr_schedulers().get_last_lr()
    cmin_losses = [
        module.min_valid_loss_dl0,
        module.min_valid_bpd_dl0,
        module.min_valid_loss_dl1,
        module.min_valid_bpd_dl1,
    ]
    return lr[0], cmin_losses


def run_lam_stage(flow, config, dset, callbacks, logger, lr=None):
    if logger:
        logger.log_metrics({"training_stage": "lam"})
    if config["default_param"] == 1:
        return lr
    dl = DataLoader(
        FWRotatedData(
            dset, flow, device="cuda:0" if config["lam_stage_gpus"] > 0 else "cpu"
        ),
        batch_size=len(dset),
        shuffle=False,
    )

    lam_config = get_scheduler_config(config, num_batches=1, stage="lam")

    module = LamPLFlow(
        flow=flow,
        lr=lam_config["lr"] if lr is None else lr,
        scheduler_config=lam_config,
        add_constant=dl.dataset.ldjs,
    )
    trainer = pl.Trainer(
        gpus=config["lam_stage_gpus"],
        strategy=config["strategy"],
        fast_dev_run=config["fast_dev_run"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
        callbacks=[
            c for c in callbacks if not isinstance(c, pl.callbacks.GPUStatsMonitor)
        ],
        gradient_clip_val=config["grad_clip"],
        gradient_clip_algorithm=config["clip_type"],
        max_epochs=config["lam_stage_epochs"],
    )
    trainer.fit(module, dl)
    lr = module.lr_schedulers().get_last_lr()
    return lr[0]
