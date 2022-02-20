import time
from functools import partial
from typing import Dict

import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from src import data as d
from src.models import MNISTModel

import hydra
import numpy as np
from omegaconf import DictConfig
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

import pyro
from pyro import distributions as dist
from pyro.infer.autoguide import (
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoDelta,
)
import torch
from torch import nn
from torch.nn.functional import softmax
from torchvision import transforms
import torchmetrics as tm

import tyxe
from tyxe.guides import AutoNormal
from src import data as d
from src.guides import AutoRadial


@hydra.main(config_path="../conf", config_name="mnist")
def train_model(cfg: DictConfig):

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224]),
    ])
    df_train = d.MuraDataset(f"{cfg.paths.data}/raw/MURA-v1.1/train_image_paths.csv", f"{cfg.paths.data}/raw/", train_transform)
    train_loader = DataLoader(df_train, batch_size=cfg.params.batch_size, num_workers=cfg.hardware.num_workers, shuffle=True)
    df_val = d.MuraDataset(f"{cfg.paths.data}/raw/MURA-v1.1/valid_image_paths.csv", f"{cfg.paths.data}/raw/", val_transform)
    val_loader = DataLoader(df_val, batch_size=cfg.params.batch_size, num_workers=cfg.hardware.num_workers, shuffle=True)

    model = MNISTModel(cfg.params.lr, dims=(2, 224, 224))

    trainer = pl.Trainer(
        gpus=cfg.hardware.gpus,
        max_epochs=cfg.params.epochs,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    trainer.test(dataloaders=val_loader, ckpt_path="best")

    # torch.save(
    #     model.state_dict(), os.path.join(cfg.paths.model, cfg.files.state_dict),
    # )


if __name__ == "__main__":
    train_model()
