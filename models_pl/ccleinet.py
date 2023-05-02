from abc import ABC
import argparse
import os
from argparse import Namespace
from typing import Dict, Any, Tuple
import numpy as np
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
import pytorch_lightning as pl
import torchvision
from rtpt import RTPT

from torch.nn import functional as F
from args import parse_args
from simple_einet.data import get_data_shape, Dist, get_distribution
from exp_utils import (
    load_from_checkpoint,
)
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from simple_einet.data import build_dataloader
from simple_einet.einet import EinetConfig, Einet
from simple_einet.ccleinet import CCLEinet
from simple_einet.distributions.binomial import Binomial
from models_pl.utils import make_einet, DATALOADER_ID_TO_SET_NAME
from models_pl.litmodel import LitModel


class SpnCCLEinet(LitModel):
    """
    Class Conditional Leaf SPN model.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="")

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10, einet_class=CCLEinet)
        self.cfg = cfg
        if cfg.classification:
            # Define loss function
            self.criterion = nn.NLLLoss()

    def training_step(self, train_batch, batch_idx):
        loss, acc = self.get_scores(
            train_batch, leaf_dropout=self.cfg.leaf_dropout)
        if acc is not None:
            self.log("Train/accuracy", acc, prog_bar=True)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # self.generate_samples(num_samples=100, class_index=list(
        #     range(10)) * 10, mpe_at_leaves=True)
        loss, accuracy = self.get_scores(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self.get_scores(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy",
                 accuracy, add_dataloader_idx=False)
        self.log(f"Test/{set_name}_loss", loss, add_dataloader_idx=False)

    def get_scores(self, batch, leaf_dropout=0):
        data, labels = batch
        data = self.preprocess(data)
        marginalized_scopes = None
        if leaf_dropout > 0:
            n = torch.tensor(data.shape[-2:]).prod()
            n_max = (n * leaf_dropout).int()
            marginalized_scopes = torch.randperm(n)[:n_max]
        if self.cfg.classification:
            joints = self.spn(data, marginalization_mask=marginalized_scopes)
            data_marginial = torch.logsumexp(joints, 1).reshape(-1, 1)
            data_conditional = joints - data_marginial
            acc = self._get_accuracy(data_conditional, labels)
            return self.criterion(data_conditional, labels), acc
        else:
            return - self.spn(data, labels, marginalization_mask=marginalized_scopes).mean(), None

    def _get_accuracy(self, ll, labels) -> torch.Tensor:
        return (labels == ll.argmax(-1)).sum() / ll.shape[0]

    def on_train_epoch_end(self):
        self.optimizers(
        ).param_groups[-1]["weight_decay"] *= self.cfg.weight_decay_decay
        with torch.no_grad():
            samples = self.generate_samples(
                num_samples=100, class_index=list(range(10)) * 10, mpe_at_leaves=True)
            grid = torchvision.utils.make_grid(
                samples.data[:100], nrow=10, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples mpe at leaves", images=[grid])

            samples = self.generate_samples(
                num_samples=100, class_index=list(range(10)) * 10)
            grid = torchvision.utils.make_grid(
                samples.data[:100], nrow=10, pad_value=0.0, normalize=True
            )
            self.logger.log_image(
                key="samples no mpe at leaves", images=[grid])

        super().on_train_epoch_end()

    def generate_samples(self, num_samples: int, class_index=None, mpe_at_leaves=False):
        samples = self.spn.sample(
            num_samples=num_samples,
            mpe_at_leaves=mpe_at_leaves,
            class_index=class_index
        ).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def configure_optimizers(self):

        # weight decay on p of binomials
        optimizer = torch.optim.Adam([
            {"params": self.spn.einsum_layers.parameters()},
            {"params": self.spn._class_dist.parameters()},
            {"params": self.spn._joint_layer.parameters()},
            {"params": self.spn.leaf.parameters(), "weight_decay": self.cfg.weight_decay}
        ], lr=self.cfg.lr)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            steps_per_epoch=self.cfg.num_steps_per_epoch,
            epochs=self.cfg.epochs
        )
        return [optimizer], [lr_scheduler]
