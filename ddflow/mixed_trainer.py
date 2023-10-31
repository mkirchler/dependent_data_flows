import numpy as np
import pytorch_lightning as pl
import torch
import torch_optimizer
from torch import optim


class PLFlow(pl.LightningModule):
    """pytorch lightning flow with survae backend for images"""

    def __init__(
        self,
        flow,
        lr=1e-4,
        scheduler_config={"name": "none"},
        use_wandb=False,
        cmin_losses=None,
        opt_mode="flow",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters("lr", "scheduler_config", "use_wandb")
        self.flow = flow
        self.opt_mode = opt_mode

        self.cmin_losses = cmin_losses

    def loss_train(self, x):
        return self.nll_train(x)

    def loss_test(self, x):
        return self.nll_test(x)

    def nll_train(self, x):
        return -self.flow.log_prob(x)

    def nll_test(self, x):
        return -self.flow.log_prob(x)

    def training_step(self, x, idx):
        loss = self.loss_train(x).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        if not loss.isfinite().all():
            return None
        else:
            return loss

    def nll2bpd(self, nll, x):
        bpd = nll / (np.log(2) * np.prod(x.shape[1:]))
        return bpd

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        name = ["valid", "valid_single"][dataloader_idx]
        nll = self.nll_test(x)
        self.log(f"{name}_loss", nll.mean(), on_epoch=True, prog_bar=True)
        bpd = self.nll2bpd(nll, x)
        self.log(f"{name}_bpd", bpd.mean(), on_epoch=True, prog_bar=True)
        return nll, bpd

    def configure_optimizers(self):
        conf = self.hparams.scheduler_config
        if self.opt_mode == "base":
            params = [
                {
                    "params": self.flow.base_dist.parameters(),
                    "lr": self.hparams.lr,
                    "weight_decay": 0.0,
                },
            ]
        elif self.opt_mode == "flow":
            params = [
                {
                    "params": sum(
                        [list(t.parameters()) for t in self.flow.transforms], []
                    ),
                    "lr": self.hparams.lr,
                    "weight_decay": conf["wd"],
                },
            ]
        elif self.opt_mode == "joint":
            params = [
                {
                    "params": sum(
                        [list(t.parameters()) for t in self.flow.transforms], []
                    ),
                    "lr": self.hparams.lr,
                    "weight_decay": conf["wd"],
                },
                {
                    "params": self.flow.base_dist.parameters(),
                    "lr": self.hparams.lr,
                    "weight_decay": 0.0,
                },
            ]

        if not "opt" in conf or conf["opt"] == "adam":
            opt = optim.Adam(
                params,
            )
        elif conf["opt"] == "sgd":
            print("using SGD trainer")
            opt = optim.SGD(
                params,
            )
        elif conf["opt"] == "adamax":
            opt = optim.Adamax(
                params,
            )
        elif conf["opt"] == "shampoo":
            print("using Shampoo optimizer")
            opt = torch_optimizer.Shampoo(
                params,
            )
        elif conf["opt"] == "lamb":
            print("using LAMB optimizer")
            opt = torch_optimizer.Lamb(
                params,
            )
        elif conf["opt"] == "aggmo":
            print("using AggMo optimizer")
            opt = torch_optimizer.AggMo(
                params,
            )
        else:
            raise NotImplemented(f"don't recognize optimizer {conf['opt']}")

        if conf["name"] == "none":
            return [opt]
        elif conf["name"] == "steplr":
            scheduler = optim.lr_scheduler.StepLR(
                opt,
                step_size=conf["step_size"],
                gamma=conf["gamma"],
            )
            return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        elif conf["name"] == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.hparams.lr,
                total_steps=conf["total_steps"],
                pct_start=conf["pct_start"],
                div_factor=conf["div_factor"],
                final_div_factor=conf["final_div_factor"],
            )
            return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        elif conf["name"] == "exponential":
            # note: exp-lr takes one step/epoch, while others take one step/step
            scheduler = optim.lr_scheduler.ExponentialLR(
                opt,
                gamma=conf["exp_gamma"],
            )
            return (
                [opt],
                [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}],
            )


class DependentPLFlow(PLFlow):
    def __init__(
        self,
        flow,
        lr=1e-4,
        scheduler_config={"name": "none"},
        use_wandb=False,
        cmin_losses=None,
        opt_mode="flow",
        **kwargs,
    ):
        super().__init__(
            flow=flow,
            lr=lr,
            scheduler_config=scheduler_config,
            use_wandb=use_wandb,
            opt_mode=opt_mode,
            **kwargs,
        )
        if not cmin_losses is None:
            (
                self.min_valid_loss_dl0,
                self.min_valid_bpd_dl0,
                self.min_valid_loss_dl1,
                self.min_valid_bpd_dl1,
            ) = cmin_losses
        else:

            self.min_valid_loss_dl0 = torch.inf
            self.min_valid_bpd_dl0 = torch.inf
            self.min_valid_loss_dl1 = torch.inf
            self.min_valid_bpd_dl1 = torch.inf

    def loss_train(self, x):
        x, inds = x
        log_prob = self.flow.log_prob_dependent(x=x, sample_indices=inds)
        loss = -log_prob
        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        name = ["valid", "valid_single"][dataloader_idx]
        nll = self.nll_test(x)
        self.log(f"{name}_loss", nll.mean(), on_epoch=True, prog_bar=True)
        bpd = self.nll2bpd(nll, x)
        self.log(f"{name}_bpd", bpd.mean(), on_epoch=True, prog_bar=True)
        return nll, bpd

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        if hasattr(self.flow.base_dist, "rhos"):
            with torch.no_grad():
                r = self.flow.base_dist.get_rhos()
            self.log("min_rho", r.min())
            self.log("max_rho", r.max())
            self.log("mean_rho", r.mean())
            self.log("median_rho", r.median())
        elif hasattr(self.flow.base_dist, "lam"):
            with torch.no_grad():
                l = self.flow.base_dist.get_lam()
            self.log("lam", l)

        if isinstance(outputs[0][0], torch.Tensor):
            nll = torch.cat([out[0] for out in outputs]).mean()
            bpd = torch.cat([out[1] for out in outputs]).mean()
            self.min_valid_loss_dl0 = min(self.min_valid_loss_dl0, nll)
            self.min_valid_bpd_dl0 = min(self.min_valid_bpd_dl0, bpd)
            self.log("valid_loss_cmin", self.min_valid_loss_dl0, on_epoch=True)
            self.log("valid_bpd_cmin", self.min_valid_bpd_dl0, on_epoch=True)
        else:
            nll = torch.cat([o[0] for o in outputs[0]]).mean()
            bpd = torch.cat([o[1] for o in outputs[0]]).mean()
            self.min_valid_loss_dl0 = min(self.min_valid_loss_dl0, nll)
            self.min_valid_bpd_dl0 = min(self.min_valid_bpd_dl0, bpd)
            self.log("valid_loss_cmin_dl0", self.min_valid_loss_dl0, on_epoch=True)
            self.log("valid_bpd_cmin_dl0", self.min_valid_bpd_dl0, on_epoch=True)

            nll = torch.cat([o[0] for o in outputs[1]]).mean()
            bpd = torch.cat([o[1] for o in outputs[1]]).mean()
            self.min_valid_loss_dl1 = min(self.min_valid_loss_dl1, nll)
            self.min_valid_bpd_dl1 = min(self.min_valid_bpd_dl1, bpd)
            self.log("valid_loss_cmin_dl1", self.min_valid_loss_dl1, on_epoch=True)
            self.log("valid_bpd_cmin_dl1", self.min_valid_bpd_dl1, on_epoch=True)


class LamPLFlow(DependentPLFlow):
    def __init__(self, add_constant=0, **kwargs):
        super().__init__(**kwargs)
        self.add_constant = add_constant
        self.opt_mode = "base"

    def loss_train(self, x):
        rotated_z = x
        log_prob = (
            self.flow.base_dist.log_prob_lam(rotated_value=rotated_z)
            + self.add_constant
        )
        return -log_prob

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        raise NotImplementedError("no validation step for lambda stages!")


class LargeBatchDependentPLFlow(DependentPLFlow):
    """more (GPU) memory-efficient optimization of equicorrelation model"""

    def __init__(
        self,
        flow,
        lr=1e-4,
        scheduler_config={"name": "none"},
        use_wandb=False,
        soft_max_size=2,
        hard_max_size=10,
        clip_type="norm",
        grad_clip=1.0,
        cmin_losses=None,
        **kwargs,
    ):
        super().__init__(
            flow=flow,
            lr=lr,
            scheduler_config=scheduler_config,
            use_wandb=use_wandb,
            cmin_losses=cmin_losses,
        )
        self.automatic_optimization = False
        self.hard_max_batch_size = hard_max_size
        self.soft_max_batch_size = soft_max_size
        self.grad_clip = grad_clip
        self.clip_type = clip_type if grad_clip > 0 else "none"

    def training_step(self, x, idx):
        x, sample_indices = x
        n = len(x)
        full_loss = 0.0
        opt = self.optimizers()
        opt.zero_grad()

        batches = self.batch_to_independent_batches(
            sample_indices=sample_indices,
            hard_max=self.hard_max_batch_size,
            soft_max=self.soft_max_batch_size,
        )

        for batch_sub_ind in batches:
            batch_ind = sample_indices[batch_sub_ind]
            batch = x[batch_sub_ind]

            z, ldj = self.flow.data2noise(batch, with_ldj=True)
            base_lp = self.flow.base_dist.log_prob_mini_batch(
                z, sample_indices=batch_ind, overwrite_bs=n
            )
            loss = -(base_lp + ldj).sum() / n
            self.manual_backward(loss)
            with torch.no_grad():
                full_loss += loss.item()
        if self.clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_clip)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(self.flow.parameters(), self.grad_clip)
        elif not self.clip_type in ["none", None]:
            raise NotImplementedError(f"clip type {self.clip_type}")
        opt.step()
        self.log("train_loss", full_loss, on_epoch=True, on_step=True, prog_bar=True)

        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def batch_to_independent_batches(self, sample_indices, hard_max, soft_max):
        block_inds = torch.tensor(
            [
                self.flow.base_dist.index_lookup[
                    ind.item() if isinstance(ind, torch.Tensor) else ind
                ]
                for ind in sample_indices
            ]
        )
        ublocks, ucounts = torch.unique(block_inds, return_counts=True)
        blocks = []
        block = torch.where(block_inds == ublocks[0])[0]
        blocks.append(block)
        last_count = ucounts[0]

        for i in range(1, len(ublocks)):
            count = ucounts[i]
            block = torch.where(block_inds == ublocks[i])[0]
            if last_count + count <= soft_max:
                blocks[-1] = torch.cat((blocks[-1], block))
                last_count += count
            else:
                blocks.append(block)
                last_count = count

            while last_count > hard_max:
                print("last block exceeding hard max, pruning")
                tmp_block = blocks.pop(-1)
                blocks.append(tmp_block[:hard_max])
                blocks.append(tmp_block[hard_max:])
                last_count = len(blocks[-1])

        return blocks
