
from simple_einet.einet import EinetConfig, Einet
from simple_einet.data import get_data_shape, Dist, get_distribution


def make_einet(cfg, num_classes: int = 1, einet_class=Einet):
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
        num_mixes=cfg.M,
        num_leaves=cfg.I,
        num_repetitions=cfg.R,
        num_classes=num_classes,
        leaf_kwargs=leaf_kwargs,
        leaf_type=leaf_type,
        dropout=cfg.dropout,
        cross_product=cfg.cp,
        log_weights=cfg.log_weights,
    )
    return einet_class(config)


# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}
