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
from simple_einet.einet import EinetConfig, Einet, CCLEinet
from simple_einet.distributions.binomial import Binomial

from autoencoder import AutoSPN, AutoCCLSPN


# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}


def make_einet(cfg, num_classes: int = 1, class_conditional_leaves=False):
    """
    Make an EinsumNetworks model based off the given arguments.

    Args:
        cfg: Arguments parsed from argparse.

    Returns:
        EinsumNetworks model.
    """
    image_shape = get_data_shape(cfg.dataset)
    # leaf_kwargs, leaf_type = {"total_count": 255}, Binomial
    leaf_kwargs, leaf_type = get_distribution(**cfg)

    config = EinetConfig(
        num_features=image_shape.num_pixels,
        num_channels=image_shape.channels,
        depth=cfg.D,
        num_sums=cfg.S,
        num_leaves=cfg.I,
        num_repetitions=cfg.R,
        num_classes=num_classes,
        leaf_kwargs=leaf_kwargs,
        leaf_type=leaf_type,
        dropout=cfg.dropout,
        cross_product=cfg.cp,
        log_weights=cfg.log_weights,
    )
    return Einet(config) if not class_conditional_leaves else CCLEinet(config)




class LitModel(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_shape = get_data_shape(cfg.dataset)
        self.rtpt = RTPT(
            name_initials="SL",
            experiment_name="einet_" + name + ("_" + str(cfg.tag) if cfg.tag else ""),
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
            milestones=[int(0.7 * self.cfg.epochs), int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self) -> None:
        self.rtpt.step()


class SpnGenerative(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, name="gen")
        self.spn = make_einet(cfg)

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        self.log("Train/loss", nll)
        return nll

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        self.log("Val/loss", nll)
        return nll

    def negative_log_likelihood(self, data):
        """
        Compute negative log likelihood of data.

        Args:
            data: Data to compute negative log likelihood of.

        Returns:
            Negative log likelihood of data.
        """
        nll = -1 * self.spn(data).mean()
        return nll

    def generate_samples(self, num_samples: int):
        samples = self.spn.sample(num_samples=num_samples, mpe_at_leaves=True).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def on_train_epoch_end(self):

        with torch.no_grad():
            samples = self.generate_samples(num_samples=64)
            grid = torchvision.utils.make_grid(
                samples.data[:64], nrow=8, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples", images=[grid])

        super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_id=0):
        data, labels = batch

        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)

        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_nll", nll, add_dataloader_idx=False)


class SpnDiscriminative(LitModel):
    """
    Discriminative SPN model. Models the class conditional data distribution at its C root nodes.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="disc")

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10)

        # Define loss function
        self.criterion = nn.NLLLoss()



    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/accuracy", accuracy)
        self.log("Val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)

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
        # logp(x | y)
        ll_x_g_y = self.spn(data)  # [N, C]

        # logp(y | x) = logp(x, y) - logp(x)
        #             = logp(x | y) + logp(y) - logp(x)
        #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)
        ll_y = np.log( 1. / self.spn.config.num_classes)
        ll_x_and_y = ll_x_g_y + ll_y
        ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
        ll_y_g_x = ll_x_g_y + ll_y - ll_x

        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        loss = self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy




class SpnCCLEinet(LitModel):
    """
    Class Conditional Leaf SPN model.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="")

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10, class_conditional_leaves=True)
        self.cfg = cfg
        if cfg.classification:
            # Define loss function
            self.criterion = nn.NLLLoss()

    def training_step(self, train_batch, batch_idx):
        loss, acc = self.get_scores(train_batch, leaf_dropout=self.cfg.leaf_dropout)
        if acc is not None:
            self.log("Train/accuracy", acc, prog_bar=True)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self.get_scores(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self.get_scores(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)
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
        self.optimizers().param_groups[-1]["weight_decay"] *= self.cfg.weight_decay_decay
        with torch.no_grad():
            samples = self.generate_samples(num_samples=100, class_index=list(range(10)) * 10, mpe_at_leaves=True)
            grid = torchvision.utils.make_grid(
                samples.data[:100], nrow=10, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples mpe at leaves", images=[grid])

            samples = self.generate_samples(num_samples=100, class_index=list(range(10)) * 10)
            grid = torchvision.utils.make_grid(
                samples.data[:100], nrow=10, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples no mpe at leaves", images=[grid])

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




class Lit_AutoSPN(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, name="gen")
        image_shape = get_data_shape(cfg.dataset)
        leaf_kwargs, leaf_type = get_distribution(**cfg)

        self.auto_spn = AutoSPN(
            in_channels=image_shape.channels,
            depth=cfg.nn_depth,
            start_filts=cfg.start_filts,
            up_mode=cfg.up_mode,
            spn_S=cfg.S,
            spn_I=cfg.I,
            spn_R=cfg.R,
            spn_depth=cfg.D,
            spn_leaf_kwargs=leaf_kwargs,
            spn_leaf_type=leaf_type

        )
        self.auto_spn(torch.ones(1, *image_shape[:]))

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        re = self.reconstruction_error(data)
        loss = re + nll
        self.log("Train/nll", nll)
        self.log("Train/re", re)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)
        re = self.reconstruction_error(data)
        loss = re + nll
        self.log("Val/nll", nll)
        self.log("Val/re", re)
        self.log("Val/loss", loss)
        return loss

    def negative_log_likelihood(self, data):
        """
        Compute negative log likelihood of data.

        Args:
            data: Data to compute negative log likelihood of.

        Returns:
            Negative log likelihood of data.
        """
        nll = -1 * self.auto_spn(data).mean()
        return nll


    def reconstruction_error(self, data):
        # have_last_dim_idx = round(data.shape[-1]/2)
        # data_crop = data[..., round(data.shape[-1]/2):]
        # sample = self.auto_spn.sample(evidence=data_crop, mpe_at_leaves=True, training=True)
        sample = self.auto_spn.sample(evidence=data, training=True)
        err = data - sample
        mae = err.abs().mean()
        return mae

    def generate_samples(self, num_samples: int):
        samples = self.auto_spn.sample(num_samples=num_samples, mpe_at_leaves=True).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def on_train_epoch_end(self):

        with torch.no_grad():
            samples = self.generate_samples(num_samples=25)
            grid = torchvision.utils.make_grid(
                samples.data[:25], nrow=5, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples", images=[grid])

        super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_id=0):
        data, labels = batch

        data = self.preprocess(data)
        nll = self.negative_log_likelihood(data)

        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_nll", nll, add_dataloader_idx=False)




class Lit_AutoCCLSPN(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, name="gen")
        image_shape = get_data_shape(cfg.dataset)
        leaf_kwargs, leaf_type = get_distribution(**cfg)
        self.auto_spn = AutoCCLSPN(
            in_channels=image_shape.channels,
            depth=cfg.nn_depth,
            start_filts=cfg.start_filts,
            up_mode=cfg.up_mode,
            spn_S=cfg.S,
            spn_I=cfg.I,
            spn_R=cfg.R,
            spn_depth=cfg.D,
            spn_leaf_kwargs=leaf_kwargs,
            spn_leaf_type=leaf_type

        )
        self.auto_spn(torch.ones(1, *image_shape[:]))

        self.criterion = torch.nn.MSELoss()

        self.prob_training = False # True # probabilistic training and no sampling
        self.last_batch = None

    def switch(self):
        self.prob_training = not self.prob_training

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        data = self.preprocess(data)
        self.last_batch = data
        if self.prob_training:
            loss = self.negative_log_likelihood(data, labels)
            self.log("Train/nll", loss)
        else:
            loss = self.reconstruction_error(data)
            self.log("Train/re", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        data = self.preprocess(data)
        if self.prob_training:
            loss = self.negative_log_likelihood(data, labels)
            self.log("Val/nll", loss)
        else:
            loss = self.reconstruction_error(data)
            self.log("Val/re", loss)
        return loss

    def negative_log_likelihood(self, data, labels):
        """
        Compute negative log likelihood of data.

        Args:
            data: Data to compute negative log likelihood of.

        Returns:
            Negative log likelihood of data.
        """
        nll = -1 * self.auto_spn(data, labels).mean()
        return nll


    def reconstruction_error(self, data):
        # have_last_dim_idx = round(data.shape[-1]/2)
        # data_crop = data[..., round(data.shape[-1]/2):]
        # sample = self.auto_spn.sample(evidence=data_crop, mpe_at_leaves=True, training=True)
        sample = self.auto_spn.sample(evidence=data, training=True)
        return self.criterion(sample, data)

    def generate_samples(self, num_samples: int, class_index=None, mpe_at_leaves=False):
        samples = self.auto_spn.sample(
            num_samples=num_samples,
            mpe_at_leaves=mpe_at_leaves,
            class_index=class_index
        ).view(
            -1, *self.image_shape
        )
        samples = samples / 255.0
        return samples

    def on_train_epoch_end(self):
        with torch.no_grad():
            samples = self.generate_samples(num_samples=30, class_index=list(range(10)) * 3, mpe_at_leaves=True)
            grid = torchvision.utils.make_grid(
                samples.data[:30], nrow=3, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples mpe at leaves", images=[grid])

            samples = self.generate_samples(num_samples=30, class_index=list(range(10)) * 3)
            grid = torchvision.utils.make_grid(
                samples.data[:30], nrow=3, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="samples no mpe at leaves", images=[grid])

            autoencoded_sample = self.auto_spn.sample(evidence=self.last_batch[:30], training=True)
            grid = torchvision.utils.make_grid(
                autoencoded_sample.data[:30], nrow=3, pad_value=0.0, normalize=True
            )
            self.logger.log_image(key="autoencoded output", images=[grid])

        super().on_train_epoch_end()
        self.log("prob_training", 1 if self.prob_training else 0)
        # if self.current_epoch % 2 == 0: # == self.cfg.epochs // 2:
        # if self.current_epoch % 4 == 0 and not self.prob_training:
        #     self.switch()
        # elif self.current_epoch % 4 != 0 and self.prob_training:
        #     self.switch()
        if self.current_epoch == 0:
            self.switch()
    
            # self.optimizers().param_groups[-1]["lr"] /= 100


    def test_step(self, batch, batch_idx, dataloader_id=0):
        accuracy = self._get_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)


    def _get_accuracy(self, batch, prep=True) -> torch.Tensor:
        data, labels = batch
        if prep:
            data = self.preprocess(data)
        # logp(x, y)
        ll = self.auto_spn(data)  # [N, C]

        return (labels == ll.argmax(-1)).sum() / ll.shape[0]



    def configure_optimizers(self):

        # different lr
        # optimizer = torch.optim.Adam([
        #     {"params": self.spn.einsum_layers.parameters()},
        #     {"params": self.spn._class_dist.parameters()},
        #     {"params": self.spn._joint_layer.parameters()},
        #     {"params": self.spn.leaf.parameters(), "lr": self.cfg.lr}
        #     ], lr=self.cfg.lr / 100)

        # weight decay on p of binomials 
        optimizer = torch.optim.Adam([
            {"params": self.auto_spn.down_convs.parameters()}, # , "lr": self.cfg.lr / 100
            {"params": self.auto_spn.conv_final.parameters()},
            {"params": self.auto_spn.up_convs.parameters()},
            {"params": self.auto_spn.spn.parameters()} 
            ], lr=self.cfg.lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs), int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]