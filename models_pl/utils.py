
from simple_einet.einet import EinetConfig, Einet
from simple_einet.data import get_data_shape, Dist, get_distribution
from simple_einet.layers import AbstractLayer
from simple_einet.mixing_einsum_layer import MixingEinsumLayer

from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn import nets as nets
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm
import torch
from torch import nn
from torch.nn import functional as F
import wandb
import numpy as np

# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}


def make_einet(
        cfg,
        num_classes: int = 1,
        einet_class=Einet,
        num_features_multiplyer=1,
        num_channels_multiplyer=1):
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

    if einet_class == Einet:
        config = EinetConfig(
            num_features=image_shape.num_pixels * num_features_multiplyer,
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
            log_weights=cfg.log_weights
        )

    else:
        config = EinetConfig(
            num_features=image_shape.num_pixels,
            num_channels=image_shape.channels * num_channels_multiplyer,
            depth=cfg.D,
            num_sums=cfg.S,
            num_mixes=cfg.M,
            num_leaves=cfg.I,
            num_repetitions=cfg.R,
            num_classes=num_classes,
            leaf_kwargs=leaf_kwargs,
            learn_permutations=cfg.learn_permutations,
            sinkhorn_tau=cfg.sinkhorn_tau,
            leaf_type=leaf_type,
            dropout=cfg.dropout,
            sum_dropout=cfg.sum_dropout,
            cross_product=cfg.cp,
            switch_permutation=cfg.switch_permutation,
            log_weights=cfg.log_weights,
            independent_colors=cfg.independent_colors,
            shuffle_features=cfg.shuffle_features,
            mixing_depth=cfg.mixing_depth,
            num_hidden_mixtures=cfg.num_hidden_mixtures,
            weight_temperature=cfg.weight_temperature
        )
    return einet_class(config)


def make_flow(cfg):
    transforms = []
    image_shape = get_data_shape(cfg.dataset)

    mask = torch.ones(image_shape.num_pixels * image_shape.channels)
    mask[::2] = -1

    def create_resnet(in_features, out_features):
        return nets.ResidualNet(
            in_features,
            out_features,
            hidden_features=cfg.num_hidden_features_flow,
            num_blocks=cfg.num_blocks_per_layer_flow,
            activation=F.relu,
            dropout_probability=cfg.dropout_probability_flow,
            use_batch_norm=cfg.batch_norm_within_layers_flow,
        )

    for _ in range(cfg.num_flow_layers):

        transform = AffineCouplingTransform(
            mask=mask, transform_net_create_fn=create_resnet
        )
        transforms.append(transform)
        mask *= -1
        if cfg.batch_norm_between_layers_flow:
            transforms.append(
                BatchNorm(features=image_shape.num_pixels * image_shape.channels))

        # transforms.append(ReversePermutation(features=image_shape.num_pixels))
        # transforms.append(
        #     MaskedAffineAutoregressiveTransform(features=image_shape.num_pixels,
        #                                         hidden_features=cfg.num_hidden_features_flow
        #                                         )
        # )
    return CompositeTransform(transforms)


def log_weight_distribution_per_layer(model):

    weight_temperature = None
    param_projection = None
    for name, module in model.named_modules():
        if "einsum_layers." in name:
            if "mixing" in name:  # mixing weights
                for i, deep_mixture_params in enumerate(module.parameters()):
                    # log unnormalized#
                    params = param_projection(deep_mixture_params)
                    params = params.cpu().detach()
                    np_hist = np.histogram(params, density=True)
                    wandb.log({f"parameters/{name}_{i}": wandb.Histogram(
                        np_histogram=np_hist)})
                    # log normalized
                    params = F.softmax(params /
                                       weight_temperature, dim=-1).cpu().detach()
                    np_hist_grads = np.histogram(params, density=True)
                    wandb.log({f"parameters_normalized/{name}_{i}": wandb.Histogram(
                        np_histogram=np_hist_grads)})
            else:
                weight_temperature = module.weight_temperature
                param_projection = module.project_params
                if isinstance(module, MixingEinsumLayer):
                    # log unnormalized
                    params = param_projection(module.einsum_weights)
                    params = params.cpu().detach()
                    np_hist = np.histogram(params, density=True)
                    wandb.log({f"parameters/{name}": wandb.Histogram(
                        np_histogram=np_hist)})
                    breakpoint()
                    shape = params.shape
                    params = params.view(
                        *shape[:-2], shape[-2] ** 2)
                    params = F.softmax(params /
                                       weight_temperature, dim=-1).cpu().detach()
                    np_hist_grads = np.histogram(params, density=True)
                    wandb.log({f"parameters_normalized/{name}": wandb.Histogram(
                        np_histogram=np_hist_grads)})
                    # log normalized
