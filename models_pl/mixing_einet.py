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

from simple_einet.einet import EinetConfig, Einet
from simple_einet.mixing_einet import MixingEinet
from simple_einet.distributions.binomial import Binomial
from simple_einet.em_optimizer import EmOptimizer
from models_pl.utils import make_einet, DATALOADER_ID_TO_SET_NAME
from models_pl.litmodel import LitModel

from sklearn.metrics import confusion_matrix


class SpnMixingEinet(LitModel):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="disc")

        self.val_confusion_matrix = np.zeros(
            (cfg.num_classes, cfg.num_classes))

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10, einet_class=MixingEinet)

        # Define loss function
        self.criterion = nn.NLLLoss()

        self.training_permutation = False
        if cfg.switch_permutation:
            self.switch(self.training_permutation)
            self.swith_points = [
                0, self.cfg.num_steps_per_epoch // self.cfg.permutation_switch_step_ratio_denominator
            ]
        self.tau = self.cfg.tau
        self.sinkhorn_tau = self.cfg.sinkhorn_tau
        self.train_step = 0

    def switch(self, training_permutation=None):
        if training_permutation is not None:
            self.training_permutation = training_permutation
        else:
            self.training_permutation = not self.training_permutation

        for params in self.spn.parameters():
            params.requires_grad = not self.training_permutation
        for params in self.spn.leaf.permutation_layer.parameters():
            params.requires_grad = self.training_permutation
        for params in self.spn.leaf.base_leaf.parameters():
            params.requires_grad = True

        self.log("train permutation", self.training_permutation, prog_bar=True)

    def entropy(self, ll):
        """
        entropy(p) = - sum(p * log(p))
        We have ll = log(p) -> entropy = - sum(exp(ll) * ll)
        """
        return - (ll * ll.exp()).sum()

    def training_step(self, train_batch, batch_idx):
        if self.cfg.learn_permutations and self.cfg.switch_permutation and batch_idx in self.swith_points:
            self.switch()
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/loss", loss)
        # if self.cfg.tau_decay != 1:
        if self.cfg.learn_permutations:
            self.schedule_sinkhorn_tau()
        if self.tau != 0.0:
            self.schedule_tau()
        self.train_step += 1
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(
            val_batch, valuation=True)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy",
                 accuracy, add_dataloader_idx=False)

    def schedule_sinkhorn_tau(self):
        max_sinkhorn_tau = self.cfg.sinkhorn_tau
        min_sinkhorn_tau = 0.1
        steps = 300
        self.sinkhorn_tau = (max_sinkhorn_tau-min_sinkhorn_tau) * \
            (0.5*np.cos(self.train_step*10/np.pi/steps)+0.5) + min_sinkhorn_tau
        self.log("Sinkhorn Tau", self.sinkhorn_tau, prog_bar=True)

    def schedule_tau(self):
        max_tau = self.cfg.tau
        min_tau = 0.1
        steps = 7000

        # exp
        self.tau *= np.power(min_tau, 1 / steps)

        # linear
        # self.tau = (min_tau - max_tau) / steps * self.train_step + max_tau

        # cyclic
        # self.tau = (max_tau-min_tau) * \
        #     (0.5 * np.cos(self.train_step*10/np.pi/steps) + 0.5)+min_tau
        self.log("Entropy Tau", self.tau, prog_bar=True)

    def _get_cross_entropy_and_accuracy(
        self, batch, valuation=False
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
        ll_x_g_y = self.spn(data, sinkhorn_tau=self.sinkhorn_tau)  # [N, C]

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
        pred_labels = ll_y_g_x.argmax(-1)
        if valuation:
            if self.val_confusion_matrix is None:
                self.val_confusion_matrix = confusion_matrix(
                    labels.cpu(), pred_labels.cpu(), labels=np.arange(self.cfg.num_classes))
            else:
                self.val_confusion_matrix += confusion_matrix(
                    labels.cpu(), pred_labels.cpu(), labels=np.arange(self.cfg.num_classes))
        else:
            self.val_confusion_matrix = None

        ent_of_posterior = self.entropy(ll_y_g_x)
        self.log("Entropy of posterior", ent_of_posterior, prog_bar=True)

        accuracy = (labels == pred_labels).sum() / ll_y_g_x.shape[0]
        if self.cfg.tau != 0.0:
            t = self.tau
        else:
            t = 0

        if self.cfg.use_em:
            loss = -loss

        loss = loss - t * ent_of_posterior

        return loss, accuracy

    def on_train_epoch_end(self):
        self.logger.log_image(key="confusion", images=[
                              self.val_confusion_matrix])
        super().on_train_epoch_end()

    def configure_optimizers(self):

        # weight decay on p of binomials
        param_list = [
            {"params": self.spn.einsum_layers.parameters(), "is_leaf": False},
            {"params": self.spn.leaf.base_leaf.parameters(
            ), "weight_decay": self.cfg.weight_decay, "is_leaf": True}
        ]
        if self.cfg.learn_permutations:
            param_list += [{"params": self.spn.leaf.permutation_layer.parameters(),
                            "lr": self.cfg.permutation_lr}]
        if self.cfg.R > 1:
            param_list += [{"params": self.spn.mixing.parameters(),
                            "is_leaf": False}]

        if self.cfg.use_em:
            optimizer = EmOptimizer(param_list, lr=self.cfg.lr)
        else:
            optimizer = torch.optim.Adam(param_list, lr=self.cfg.lr)

        # optimizer = torch.optim.SGD(param_list, lr=self.cfg.lr)

        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.cfg.lr,
        #     steps_per_epoch=self.cfg.num_steps_per_epoch,
        #     epochs=self.cfg.epochs
        # )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs),
                        int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=int(self.cfg.num_steps_per_epoch * 0.4),
        #     eta_min=1e-7,
        #     last_epoch=self.cfg.epochs
        # )
        return [optimizer], [lr_scheduler]
