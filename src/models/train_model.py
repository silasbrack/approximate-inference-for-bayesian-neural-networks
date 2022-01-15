import os
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models import MNISTModel


@hydra.main(config_path="../conf", config_name="mnist")
def train_model(cfg: DictConfig):

    model = MNISTModel(
        cfg.paths.data,
        cfg.params.lr,
        cfg.params.batch_size,
    )
    trainer = pl.Trainer(
        gpus=cfg.hardware.gpus,
        max_epochs=cfg.params.epochs,
        precision=32 if cfg.hardware.gpus > 0 else 32,
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir=cfg.paths.logs, name="mnist_model"),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            ),
            ModelCheckpoint(
                dirpath=cfg.paths.checkpoints,
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
        ],
    )

    trainer.fit(model)

    torch.save(
        model.state_dict(),
        os.path.join(cfg.paths.model, cfg.files.state_dict),
    )


if __name__ == "__main__":
    train_model()
