import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import MNISTData
from src.models.bayesian_mnist_l import BayesianMnistModelLightning


@hydra.main(config_path="../conf", config_name="bayesian_mnist")
def train_model(cfg: DictConfig):

    model = BayesianMnistModelLightning(
        cfg.params.lr, cfg.params.num_particles
    )
    data = MNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    trainer = pl.Trainer(
        gpus=cfg.hardware.gpus,
        max_epochs=cfg.params.epochs,
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

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    trainer.test(dataloaders=data.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    train_model()
