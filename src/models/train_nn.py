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

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    model = MNISTModel(cfg.training.lr)

    trainer = pl.Trainer(
        gpus=cfg.hardware.gpus,
        max_epochs=cfg.training.epochs,
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir=cfg.paths.logs, name="mnist_model"),
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            ),
            callbacks.ModelCheckpoint(
                dirpath=cfg.paths.checkpoints,
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    trainer.test(dataloaders=data.test_dataloader(), ckpt_path="best")

    torch.save(
        model.state_dict(),
        os.path.join(cfg.paths.model, cfg.files.state_dict),
    )


if __name__ == "__main__":
    train_model()
