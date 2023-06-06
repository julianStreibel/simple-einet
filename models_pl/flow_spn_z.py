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
from simple_einet.distributions.binomial import Binomial
from models_pl.utils import make_einet, DATALOADER_ID_TO_SET_NAME, make_flow
from models_pl.litmodel import LitModel


class FlowSpnZGenerative(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, name="gen")

        # adding flow features (z)
        self.spn = make_einet(cfg, num_features_multiplyer=2, num_classes=10)
        self.flow = make_flow(cfg)
        self.image_shape = [*self.image_shape]
        self.image_shape[1] *= 2

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data, labels)
        self.log("Train/loss", nll)
        return nll

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data, labels)
        self.log("Val/loss", nll)
        return nll

    def test_step(self, batch, batch_idx, dataloader_id=0):
        data, labels = batch

        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data, labels)

        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_nll", nll, add_dataloader_idx=False)

    def negative_log_likelihood(self, data, labels):
        """
        Compute negative log likelihood of data.

        Args:
            data: Data to compute negative log likelihood of.

        Returns:
            Negative log likelihood of data.
        """
        og_shape = data.shape
        z, ldj = self.flow(
            data.reshape(data.shape[0], -1)
        )

        z = z.reshape(og_shape)

        xz = torch.stack((data, z), dim=2).view(
            -1, *self.image_shape
        )  # stack in feature dim

        nll = -1 * torch.gather(
            self.spn(xz),
            1,
            labels.reshape(-1, 1)
        ).mean()

        # nll = -1 * self.spn(xz).mean()
        return nll

    def generate_samples(self, num_samples: int):
        samples = self.spn.sample(num_samples=num_samples, mpe_at_leaves=True)

        samples = samples.view(
            -1, *self.image_shape
        )
        x_samples, z_samples = torch.split(samples, self.image_shape[2], dim=2)

        og_shape = z_samples.shape
        flow_samples, ldj = self.flow.inverse(
            z_samples.reshape(z_samples.shape[0], -1))
        flow_samples = flow_samples.reshape(og_shape)

        return x_samples, flow_samples

    def on_train_epoch_end(self):

        with torch.no_grad():
            spn_samples, flow_samples = self.generate_samples(num_samples=64)
            grid = torchvision.utils.make_grid(
                spn_samples.data[:64], nrow=8, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="spn samples", images=[grid])

            grid = torchvision.utils.make_grid(
                flow_samples.data[:64], nrow=8, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="flow samples", images=[grid])

        super().on_train_epoch_end()

    def configure_optimizers(self):
        # weight decay on p of binomials
        optimizer = torch.optim.Adam([
            {"params": self.spn.parameters()},
            {"params": self.flow.parameters(), "lr": 0.001},
        ], lr=self.cfg.lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs),
                        int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]


class FlowSpnZGDiscriminative(LitModel):
    """
    Discriminative SPN model. Models the class conditional data distribution at its C root nodes.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="disc")

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10, num_features_multiplyer=2)
        self.flow = make_flow(cfg)
        self.image_shape = [*self.image_shape]
        self.image_shape[1] *= 2

        # Define loss function
        self.criterion = nn.NLLLoss()

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy",
                 accuracy, add_dataloader_idx=False)

    def _get_cross_entropy_and_accuracy(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross entropy loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (cross entropy loss, accuracy).
        """
        data, labels = batch
        data = self.preprocess(data)

        # flow
        og_shape = data.shape
        z, ldj = self.flow(
            data.reshape(data.shape[0], -1)
        )
        z = z.reshape(og_shape)

        xz = torch.stack((data, z), dim=2).view(
            -1, *self.image_shape
        )  # stack in feature dim

        # logp(x | y)
        ll_x_g_y = self.spn(xz)  # [N, C]

        # logp(y | x) = logp(x, y) - logp(x)
        #             = logp(x | y) + logp(y) - logp(x)
        #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)
        ll_y = np.log(1. / self.spn.config.num_classes)
        ll_x_and_y = ll_x_g_y + ll_y
        ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
        ll_y_g_x = ll_x_g_y + ll_y - ll_x

        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        loss = self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy

    def configure_optimizers(self):
        # weight decay on p of binomials
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs),
                        int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def generate_samples(self, num_samples: int):
        samples = self.spn.sample(num_samples=num_samples, mpe_at_leaves=True)

        samples = samples.view(
            -1, *self.image_shape
        )
        x_samples, z_samples = torch.split(samples, self.image_shape[2], dim=2)

        og_shape = z_samples.shape
        flow_samples, ldj = self.flow.inverse(
            z_samples.reshape(z_samples.shape[0], -1))
        flow_samples = flow_samples.reshape(og_shape)

        return x_samples, flow_samples

    def on_train_epoch_start(self):

        with torch.no_grad():
            spn_samples, flow_samples = self.generate_samples(num_samples=64)
            grid = torchvision.utils.make_grid(
                spn_samples.data[:64], nrow=8, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="spn samples", images=[grid])

            grid = torchvision.utils.make_grid(
                flow_samples.data[:64], nrow=8, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="flow samples", images=[grid])

        super().on_train_epoch_start()
