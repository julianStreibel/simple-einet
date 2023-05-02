
from abc import ABC
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
import pytorch_lightning as pl
from rtpt import RTPT

from simple_einet.data import get_data_shape, Dist


class LitModel(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_shape = get_data_shape(cfg.dataset)
        self.rtpt = RTPT(
            name_initials="SL",
            experiment_name="einet_" + name +
            ("_" + str(cfg.tag) if cfg.tag else ""),
            max_iterations=cfg.epochs + 1,
        )
        self.save_hyperparameters()

    def preprocess(self, data: torch.Tensor):
        if self.cfg.dist == Dist.BINOMIAL or self.cfg.dist == Dist.CCLBINOMIAL:
            data *= 255.0

        return data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs),
                        int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self) -> None:
        self.rtpt.step()
