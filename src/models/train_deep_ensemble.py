import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm

from src import data as d
from src.models import MNISTModel


@hydra.main(config_path="../conf", config_name="mnist")
def train_model(cfg: DictConfig):
    data = d.MNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    n_ensembles = 20
    state_dicts = []
    for i in range(n_ensembles):
        model = MNISTModel(cfg.params.lr)

        trainer = pl.Trainer(
            gpus=cfg.hardware.gpus,
            max_epochs=cfg.params.epochs,
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
        trainer.test(model, test_dataloaders=data.test_dataloader())
        state_dict = model.state_dict()
        state_dicts.append(state_dict)

    state_dict = {key: torch.sum(torch.stack([sd[key] for sd in state_dicts]), dim=0) for key in state_dicts[0]}

    model = MNISTModel(cfg.params.lr)
    model.load_state_dict(state_dict)

    accuracy = tm.Accuracy()
    for x, y in data.test_dataloader():
        logits = model(x)
        # probs = softmax(logits, dim=-1)
        # conf, preds = torch.max(probs, dim=-1)
        # preds = preds.detach()
        # conf = conf.detach()
        accuracy(logits, y)
    print(accuracy.compute()*100)

    torch.save(
        model.state_dict(), os.path.join(cfg.paths.model, cfg.files.state_dict),
    )


if __name__ == "__main__":
    train_model()
