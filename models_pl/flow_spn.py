import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from omegaconf import DictConfig
from simple_einet.flow.multi_scale_flow import create_flows
from models_pl.utils import DATALOADER_ID_TO_SET_NAME
from simple_einet.distributions import RatNormal, CCRatNormal, ClassRatNormal
from simple_einet.einet import EinetConfig
from simple_einet.mixing_einet import MixingEinet
from simple_einet.mixing_ccleinet import MixingCCLEinet
from simple_einet.mixing_class_einet import MixingClassEinet
from simple_einet.data import get_data_shape
import numpy as np

import torchvision

from sklearn.metrics import confusion_matrix


class FlowSPN(pl.LightningModule):

    def __init__(self, cfg: DictConfig, import_samples=8):
        """
        Inputs:
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        image_shape = get_data_shape(cfg.dataset)
        flows = create_flows(image_shape)
        self.cfg = cfg
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        # self.prior = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)
        # 512 * image_shape.channels
        num_features = int(image_shape.channels * (2 * 2)
                           ** np.log2(image_shape.width))
        self.prior = MixingClassEinet(
            EinetConfig(
                num_features=num_features,
                num_channels=1,
                depth=int(np.floor(np.log2(num_features))) - 1,
                num_sums=2,
                num_mixes=3,
                # num_hidden_mixtures=10,
                # mixing_depth=3,
                num_leaves=2,
                num_repetitions=1,
                num_classes=cfg.num_classes,
                leaf_kwargs={"min_sigma": 0.01, "max_sigma": 10.},
                leaf_type=ClassRatNormal,  # RatNormal,
                cross_product=True
            )
        )

        # self.prior = MixingCCLEinet(
        #     EinetConfig(
        #         num_features=num_features,
        #         num_channels=1,
        #         depth=int(np.floor(np.log2(num_features))) - 1,
        #         num_sums=2,
        #         num_mixes=3,
        #         num_hidden_mixtures=10,
        #         mixing_depth=3,
        #         num_leaves=10,
        #         num_repetitions=2,
        #         num_classes=cfg.num_classes,
        #         leaf_kwargs={"min_sigma": 0.01, "max_sigma": 10.},
        #         leaf_type=CCRatNormal,  # RatNormal,
        #         cross_product=True
        #     )
        # )
        # self.prior = MixingEinet(
        #     EinetConfig(
        #         num_features=num_features,
        #         num_channels=1,
        #         depth=int(np.floor(np.log2(num_features))) - 1,
        #         num_sums=2,
        #         num_mixes=5,
        #         num_hidden_mixtures=10,
        #         mixing_depth=3,
        #         num_leaves=3,
        #         num_repetitions=2,
        #         num_classes=cfg.num_classes,
        #         leaf_kwargs={"min_sigma": 0.001, "max_sigma": 2.},
        #         leaf_type=RatNormal,
        #         cross_product=True
        #     )
        # )
        self.val_confusion_matrix = np.zeros(
            (cfg.num_classes, cfg.num_classes))

        self.criterion = nn.NLLLoss()

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior(z)  # .sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    def _get_cross_entropy_and_accuracy(
        self, batch, valuation=False
    ):
        """
        Compute cross entropy loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (cross entropy loss, accuracy).
        """
        data, labels = batch
        z, ldj = self.encode(data)
        z = z.reshape(
            z.shape[0],
            1,
            z.shape[1] * z.shape[2],
            z.shape[3]
        )
        ll_x_g_y = self.prior(z)
        ll_y = np.log(1. / self.prior.config.num_classes)
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

        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(num_samples=img_shape[0]).to(self.device)
            z = z.reshape(img_shape)
        else:
            z = z_init.to(self.device)
        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        param_list = [
            {"params": self.flows.parameters()},
            {"params": self.prior.parameters(), "lr": self.cfg.lr}
        ]
        optimizer = optim.Adam(param_list, lr=1e-4)

        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        # scheduler = optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     max_lr=self.cfg.lr,
        #     base_lr=1e-4,
        #     step_size_up=100,
        #     cycle_momentum=False,
        #     mode="triangular2")
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        self.log("Train/accuracy", accuracy, prog_bar=True)
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(
            batch, valuation=True)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_id]
        self.log(f"Test/{set_name}_accuracy",
                 accuracy, add_dataloader_idx=False)

    def on_train_epoch_end(self):
        self.logger.log_image(key="confusion", images=[
                              self.val_confusion_matrix])
        super().on_train_epoch_end()
