#!/usr/bin/env python
import logging
from omegaconf import DictConfig, OmegaConf
import os

import hydra
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, RichProgressBar
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import (
    ModelSummary,
)

from exp_utils import (
    setup_experiment,
    load_from_checkpoint,
    plot_distribution,
)
from models_pl import SpnMixingCCLEinet
from simple_einet.data import Dist
from simple_einet.data import build_dataloader

# A logger for this file
logger = logging.getLogger(__name__)


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.1):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        return x + torch.randn(x.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


@hydra.main(version_base=None, config_path="./conf", config_name="mixing_ccleinet_config")
def main(cfg: DictConfig):
    preprocess_cfg(cfg)

    logger.info(OmegaConf.to_yaml(cfg))

    results_dir, cfg = setup_experiment(
        name="simple-einet", cfg=cfg, remove_if_exists=True)

    seed_everything(cfg.seed, workers=True)

    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # Create dataloader
    additional_transforms = [
        transforms.ToTensor(),
        transforms.RandomApply([
            AddGaussianNoise(0, std=cfg.noise_std),
        ], p=0.5),
        transforms.ToPILImage()
    ]
    normalize = cfg.dist == Dist.NORMAL
    train_loader, val_loader, test_loader = build_dataloader(
        cfg=cfg, loop=False, normalize=normalize, additional_transforms=additional_transforms
    )

    cfg.num_steps_per_epoch = len(train_loader)

    # Load or create model
    if cfg.load_and_eval:
        model = load_from_checkpoint_cc(
            cfg.results_dir, load_fn=SpnMixingCCLEinet.load_from_checkpoint, args=cfg
        )
    else:
        model = SpnMixingCCLEinet(cfg)

    seed_everything(cfg.seed, workers=True)

    print("Training model...")

    # Create callbacks
    logger_wandb = WandbLogger(name=cfg.tag, project="mixing_ccleinet", group=cfg.group_tag,
                               offline=not cfg.wandb)
    logger_wandb.watch(model, log="all")

    # Store number of model parameters
    summary = ModelSummary(model, max_depth=-1)
    print("Model:")
    print(model)
    print("Summary:")
    print(summary)
    logger_wandb.experiment.config[
        "trainable_parameters"
    ] = summary.trainable_parameters
    logger_wandb.experiment.config["trainable_parameters_leaf"] = summary.param_nums[
        summary.layer_names.index("spn.leaf")
    ]
    logger_wandb.experiment.config["trainable_parameters_sums"] = summary.param_nums[
        summary.layer_names.index("spn.einsum_layers")
    ]

    # Setup devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [int(cfg.gpu)]
    # elif torch.backends.mps.is_available():  # Currently leads to errors
    #     accelerator = "mps"
    #     devices = 1
    else:
        accelerator = "cpu"
        devices = None

    # Setup callbacks
    callbacks = []

    # Add StochasticWeightAveraging callback
    if cfg.swa:
        swa_callback = StochasticWeightAveraging()
        callbacks.append(swa_callback)

    # Enable rich progress bar
    callbacks.append(RichProgressBar())

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        # max_steps=cfg.max_steps,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.debug,
        profiler=cfg.profiler,
        auto_lr_find=True,
        gradient_clip_val=cfg.gradient_clip_val
    )

    if not cfg.load_and_eval:
        # trainer.tune(model, train_dataloaders=train_loader,
        #              val_dataloaders=val_loader)
        # Fit model
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    print("Evaluating model...")

    if "synth" in cfg.dataset and not cfg.classification:
        plot_distribution(
            model=model.spn, dataset_name=cfg.dataset, logger_wandb=logger_wandb
        )

    # Evaluate spn reconstruction error
    test_res = trainer.test(
        model=model, dataloaders=[train_loader,
                                  val_loader, test_loader], verbose=True
    )

    print("Finished evaluation...")
    return test_res[2]["Test/test_accuracy"]


def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")

    # If results dir is not set, get from ENV, else take ~/data
    if "data_dir" not in cfg:
        cfg.data_dir = os.getenv("DATA_DIR", os.path.join(home, "data"))

    # If results dir is not set, get from ENV, else take ~/results
    if "results_dir" not in cfg:
        cfg.results_dir = os.getenv(
            "RESULTS_DIR", os.path.join(home, "results"))

    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.precision == "16" or cfg.precision == "32":
        cfg.precision = int(cfg.precision)

    if "profiler" not in cfg:
        cfg.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg:
        cfg.tag = None

    if "group_tag" not in cfg:
        cfg.group_tag = None

    cfg.dist = Dist[cfg.dist.upper()]


if __name__ == "__main__":
    main()
