from abc import ABC
import argparse
import os
from argparse import Namespace
from typing import Dict, Any, Tuple
import numpy as np
from omegaconf import DictConfig

import wandb

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
        self.criterion = nn.NLLLoss()
        self.number_of_params = sum(p.numel() for p in self.parameters())

        if self.cfg.learn_permutations:
            self.number_of_params -= sum(p.numel()
                                         for p in self.spn.leaf.permutation_layer.parameters())
            self.training_permutation = False
            if cfg.switch_permutation:
                self.switch(self.training_permutation)
                self.swith_points = [
                    0, self.cfg.num_steps_per_epoch // 10
                ]

    def switch(self, training_permutation=None):
        if training_permutation is not None:
            self.training_permutation = training_permutation
        else:
            self.training_permutation = not self.training_permutation

        # for params in self.spn.parameters():
        #     params.requires_grad = not self.training_permutation
        for params in self.spn.leaf.permutation_layer.parameters():
            params.requires_grad = self.training_permutation
        # for params in self.spn.leaf.base_leaf.parameters():
        #     params.requires_grad = True

        self.log("train permutation",
                 int(self.training_permutation), prog_bar=True)

    def training_step(self, train_batch, batch_idx):
        if self.cfg.learn_permutations and self.cfg.switch_permutation and batch_idx in self.swith_points:
            self.switch()
        loss, acc = self.get_scores(
            train_batch, leaf_dropout=self.cfg.leaf_dropout)
        if acc is not None:
            self.log("Train/accuracy", acc, prog_bar=True)
        self.log("Train/loss", loss)
        self._log_lr()
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.generate_samples(num_samples=100, class_index=list(
            range(10)) * 10, mpe_at_leaves=True)
        loss, accuracy = self.get_scores(val_batch, training=False)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self.get_scores(batch, training=False)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy",
                 accuracy, add_dataloader_idx=False)
        self.log(f"Test/{set_name}_loss", loss, add_dataloader_idx=False)

    def entropy(self, ll):
        """
        entropy(p) = - sum(p * log(p))
        We have ll = log(p) -> entropy = - sum(exp(ll) * ll)
        """
        return - (ll * ll.exp()).sum()

    def conditional_entropy(self, log_joints, log_conditional):
        """
        cond_entropy = - sum(p(x, y) * log(p(x, y)/p(y)))
        """
        return - (log_joints.exp() * log_conditional).sum()

    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        if self.cfg.dist == Dist.BINOMIAL or self.cfg.dist == Dist.CCLBINOMIAL:
            mixed_x = mixed_x.round()
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def get_scores(self, batch, leaf_dropout=0, training=True):
        if self.cfg.R > 1:
            self._log_repetition_weights()
        data, labels = batch
        data = self.preprocess(data)

        marginalized_scopes = None
        if leaf_dropout > 0:
            n = torch.tensor(data.shape[-2:]).prod()
            n_max = (n * leaf_dropout).int()
            marginalized_scopes = torch.randperm(n)[:n_max]

        if self.cfg.classification or not training:
            if self.cfg.mixup and training:
                data, labels_a, labels_b, lam = self.mixup_data(
                    data, labels, self.cfg.mixup_alpha)

            joints = self.spn(data, marginalization_mask=marginalized_scopes)
            data_marginial = torch.logsumexp(joints, 1).reshape(-1, 1)
            data_conditional = joints - data_marginial

            acc = self._get_accuracy(data_conditional, labels)
            # conditional_entropy
            cond_ent = self.entropy(data_conditional)
            if self.cfg.classification:
                self.log("Cond. Entropy", cond_ent, prog_bar=True)

            if self.cfg.mixup and training:
                loss = self.mixup_criterion(
                    self.criterion, data_conditional, labels_a, labels_b, lam)
            else:
                loss = self.criterion(data_conditional, labels)
            loss = loss - self.cfg.tau * cond_ent

            if not training:
                self.log(
                    "BIC", self.get_bic(
                        - data_conditional.gather(-1,
                                                  labels.unsqueeze(-1)).mean(),
                        data.shape[0]
                    )
                )
        else:
            lls = self.spn(
                data, labels, marginalization_mask=marginalized_scopes)
            ent = self.entropy(lls)
            self.log("Entropy", ent, prog_bar=True)
            loss = -(lls.mean() + self.cfg.tau * ent)
            acc = -1
        return loss, acc

    def _get_accuracy(self, ll, labels) -> torch.Tensor:
        return (labels == ll.argmax(-1)).sum() / ll.shape[0]

    def _log_lr(self):
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("lr", lr, prog_bar=True)

    def _log_repetition_weights(self):
        weights = self.spn.mixing.weights.flatten().cpu().detach()
        activated_weights = F.softmax(weights, dim=0)
        wandb.log({
            "mixing_weights_activated": wandb.Histogram(activated_weights)})
        wandb.log({
            "mixing_weights": wandb.Histogram(weights)})

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

    def get_bic(self, mean_nll, num_samples):
        return 2 * num_samples * mean_nll + np.log(num_samples) * self.number_of_params

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
        param_list = [
            {"params": self.spn.einsum_layers.parameters()},
            {"params": self.spn._class_dist.parameters()},
            {"params": self.spn._joint_layer.parameters()},
            {"params": self.spn.leaf.base_leaf.parameters(
            ), "weight_decay": self.cfg.weight_decay}
        ]
        if self.cfg.learn_permutations:
            param_list += [{"params": self.spn.leaf.permutation_layer.parameters(),
                            "lr": 0.99}]
        optimizer = torch.optim.Adam(param_list, lr=self.cfg.lr)

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
