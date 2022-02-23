from functools import partial

import fire
import pyro
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from omegaconf import DictConfig, OmegaConf
from pyro import distributions as dist
from pyro.infer.autoguide import (
    AutoDelta,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
)
from torch import nn
from torch.nn.functional import softmax

import tyxe
from src import data as d
from src.data.fashion_mnist import FashionMNISTData
from src.guides import AutoRadial
from src.models import MNISTModel
from src.models.train_mnist_tyxe import eval_model
from tyxe.guides import AutoNormal


def mnist(
    path: str,  # Path to log folder, e.g., outputs/2022-02-02/15-15-26/
    checkpoint_file: str,  # epoch=14-step=6449.ckpt
):
    # hparams: DictConfig = DictConfig(
    #     OmegaConf.load(f"{path}/logs/mnist_model/version_0/hparams.yaml")
    # )
    # model: MNISTModel = MNISTModel(**hparams)
    # state_dict_path = f"{path}/models/state_dict.pt"
    # model.load_state_dict(torch.load(state_dict_path))

    cfg: DictConfig = DictConfig(OmegaConf.load(f"{path}/.hydra/config.yaml"))
    model: MNISTModel = MNISTModel.load_from_checkpoint(
        checkpoint_path=f"{path}/{cfg.paths.checkpoints}/{checkpoint_file}"
    )
    data: FashionMNISTData = d.FashionMNISTData(
        "data/", cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    trainer = pl.Trainer()
    trainer.test(model, data.test_dataloader())


def tyxe_model(
    path: str,
):  # Path to log folder, e.g., outputs/2022-02-02/15-15-26/
    cfg: DictConfig = DictConfig(OmegaConf.load(f"{path}/.hydra/config.yaml"))

    data = d.FashionMNISTData("data/", cfg.params.batch_size, cfg.hardware.num_workers)
    data.setup()

    hidden_size = 32
    channels, width, height = (1, 28, 28)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, data.n_classes),
    )
    likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
    inference_dict = {
        "ml": None,
        "map": AutoDelta,
        "laplace": AutoLaplaceApproximation,
        "meanfield": partial(AutoNormal, init_scale=1e-2),
        "lowrank": partial(AutoLowRankMultivariateNormal, rank=10),
        "radial": AutoRadial,
    }
    inference = inference_dict[cfg.params.guide]
    prior_kwargs = {"expose_all": False, "hide_all": True} if inference is None else {}
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), **prior_kwargs)
    bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)
    bnn
    # bnn = torch.load(f"{path}/model.pt")

    optim = pyro.optim.Adam({"lr": cfg.params.lr})

    bnn.fit(
        data.train_dataloader(),
        optim,
        num_epochs=1,
        num_particles=cfg.params.num_particles,
    )

    # print(bnn.state_dict())
    #
    # pyro.get_param_store().load(f"{path}/param_store.pt")
    # bnn.load_state_dict(torch.load(f"{path}/state_dict.pt"))
    # optim.load(f"{path}/optim.pt")

    result = eval_model(bnn, data.test_dataloader(), cfg.params.posterior_samples)
    print(result)


if __name__ == "__main__":
    fire.Fire()
