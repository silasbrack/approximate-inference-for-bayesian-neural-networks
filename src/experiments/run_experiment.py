from pathlib import Path

import torch
from azureml.core import Run

from src.data.AmazonReviewDataModule import AmazonReviewDataModule
from src.models.AzureMLLogger import AzureMLLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse

from src.models.distilbertsentimentclassifier \
    import DistilBertSentimentClassifier

project_dir = Path(__file__).resolve().parents[2]


def main():
    args = parse_args()
    train_model(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon review sentiment classification task"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        metavar="N",
        help="number of GPUs (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="batch size (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        metavar="N",
        help="Number of workers for the dataloader (default: 6)",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        default=False,
        help="Run from Azure.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default='data/processed/',
        help="Path to data (default: data/processed/)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate of ADAM optimizer (default: 1e-3)",
    )
    args = parser.parse_args()
    return args


def train_model(args):
    if args.azure:
        run = Run.get_context()

    model_name = "distilbert-base-uncased"
    model = DistilBertSentimentClassifier(model_name, args.lr)

    print(f'Data path: {args.data_path}')
    data = AmazonReviewDataModule(batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  data_path=str(project_dir.joinpath(
                                      args.data_path)))

    checkpoint = "outputs/models/DistilBertSentimentClassifier" \
                 "/checkpoints/weights"

    trainer_params = {
        "gpus": args.gpus,
        "max_epochs": args.epochs,
        "precision": 32 if args.gpus > 0 else 32,
        "progress_bar_refresh_rate": 20,
        "log_every_n_steps": 10,
        "logger": AzureMLLogger() if args.azure else None,
        "callbacks": [
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min"
            ),
            ModelCheckpoint(
                dirpath=checkpoint,
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
        ],
    }

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(
        model,
        train_dataloader=data.train_dataloader(),
        val_dataloaders=data.val_dataloader()
    )

    print('Exporting model')
    torch.save(obj=model.state_dict(), f='outputs/model.pt')

    if args.azure:
        run.complete()


if __name__ == "__main__":
    main()
