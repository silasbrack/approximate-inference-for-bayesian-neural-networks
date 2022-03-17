import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from src import data as d
from src.models import MNISTModel


@hydra.main(config_path="../../conf", config_name="nn")
def train_model(cfg: DictConfig):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
        "mura": d.MuraData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    model = MNISTModel(cfg.training.lr, num_classes=data.n_classes)

    trainer = pl.Trainer(
        gpus=cfg.hardware.gpus,
        max_epochs=cfg.training.epochs,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    trainer.test(dataloaders=data.test_dataloader())

    torch.save(
        model.state_dict(),
        os.path.join(cfg.paths.model, cfg.files.state_dict),
    )
    return model


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)

    train_model()
