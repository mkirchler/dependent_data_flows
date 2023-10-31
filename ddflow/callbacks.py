import torch
from pytorch_lightning.callbacks import Callback

import wandb

from ddflow.utils import plot_2d, plot_mv, sample_to_pil


class LogSampleData(Callback):
    def __init__(
        self, log_every_n_epochs=1, num_samples=100, data_type="2d", mv_num_cols=5
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.data_type = data_type
        self.mv_num_cols = 5
        assert data_type in [
            "2d",
            "mv",
            "img",
        ], 'data_type needs to be in ["2d", "mv", "img"]'

    def on_epoch_end(self, trainer, pl_module):
        if (self.log_every_n_epochs > 0) and (
            (trainer.current_epoch == 0)
            or ((trainer.current_epoch + 1) % self.log_every_n_epochs == 0)
        ):
            with torch.no_grad():
                sample = pl_module.flow.static_seed_sample(self.num_samples)
            if self.data_type == "2d":
                plot_2d(sample, tag="sample")
            elif self.data_type == "img":
                imgs = sample_to_pil(sample)
                wandb.log({"sample": [wandb.Image(img) for img in imgs]})
            elif self.data_type == "mv":
                plot_mv(sample, tag="sample", num_cols=self.mv_num_cols)
            else:
                raise NotImplementedError()

        return super().on_epoch_end(trainer, pl_module)


class TrainingStageController(Callback):
    def __init__(
        self,
        first_stage_epochs=1,
        lam_stage_epochs=0,
        main_stage_epochs=0,
        num_alternating=0,
    ):
        super().__init__()
        self.main_stages = list(range(first_stage_epochs)) + sum(
            [
                list(
                    range(
                        first_stage_epochs
                        + lam_stage_epochs
                        + i * (lam_stage_epochs + main_stage_epochs),
                        first_stage_epochs
                        + (i + 1) * (lam_stage_epochs + main_stage_epochs),
                    )
                )
                for i in range(num_alternating)
            ],
            [],
        )
        self.lam_stages = sum(
            [
                list(
                    range(
                        first_stage_epochs + i * (lam_stage_epochs + main_stage_epochs),
                        first_stage_epochs
                        + lam_stage_epochs
                        + i * (lam_stage_epochs + main_stage_epochs),
                    )
                )
                for i in range(num_alternating)
            ],
            [],
        )

    def on_epoch_start(self, trainer, pl_module):
        ep = trainer.current_epoch
        if ep in self.main_stages:
            pl_module.alternating_training_stage = 1
        elif ep in self.lam_stages:
            pl_module.alternating_training_stage = 2
        else:
            raise ValueError(
                f"misconfiguration! epoch {ep} not in main stages or lam stages"
            )
        return super().on_epoch_start(trainer, pl_module)
