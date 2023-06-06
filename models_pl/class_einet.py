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

from simple_einet.mixing_class_einet import MixingClassEinet
from simple_einet.data import build_dataloader
from simple_einet.einet import EinetConfig, Einet
from simple_einet.distributions.binomial import Binomial
from models_pl.utils import make_einet, DATALOADER_ID_TO_SET_NAME
from models_pl.litmodel import LitModel

from sklearn.metrics import confusion_matrix


class ClassSpn(LitModel):
    """
    Discriminative SPN model. Models the class conditional data distribution at its C root nodes.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="disc")

        # Construct SPN
        self.spn = make_einet(cfg, num_classes=10,
                              einet_class=MixingClassEinet)

        # Define loss function
        self.criterion = nn.NLLLoss()
        self.val_confusion_matrix = np.zeros(
            (cfg.num_classes, cfg.num_classes))

        self.tau = self.cfg.tau
        self.train_step = 0

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/loss", loss)
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
        ll_x_g_y = self.spn(data)  # [N, C]

        # logp(y | x) = logp(x, y) - logp(x)
        #             = logp(x | y) + logp(y) - logp(x)
        #             = logp(x | y) + logp(y) - logsumexp(logp(x,y), dim=y)
        ll_y = np.log(1. / self.spn.config.num_classes)
        ll_x_and_y = ll_x_g_y + ll_y
        ll_x = torch.logsumexp(ll_x_and_y, dim=1, keepdim=True)
        ll_y_g_x = ll_x_g_y + ll_y - ll_x
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

        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        loss = self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        if self.cfg.tau != 0.0:
            t = self.tau
        else:
            t = 0
        loss = loss - t * ent_of_posterior
        return loss, accuracy

    def on_train_epoch_start(self):
        self.logger.log_image(key="confusion", images=[
                              self.val_confusion_matrix])
        super().on_train_epoch_start()

    def entropy(self, ll):
        """
        entropy(p) = - sum(p * log(p))
        We have ll = log(p) -> entropy = - sum(exp(ll) * ll)
        """
        return - (ll * ll.exp()).sum()

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
