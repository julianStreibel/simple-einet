
from simple_einet.einet import EinetConfig, Einet
from simple_einet.data import get_data_shape, Dist, get_distribution

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
from torch.nn import functional as F


# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}


def make_einet(cfg, num_classes: int = 1, einet_class=Einet, num_features_multiplyer=1):
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
            num_channels=image_shape.channels,
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
