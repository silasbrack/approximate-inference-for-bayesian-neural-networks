import time
import tyxe
from torch import nn
from pyro import distributions as dist
import pyro
from src import data as d
import hydra
from omegaconf import DictConfig
from torchmetrics import Accuracy


@hydra.main(config_path="../conf", config_name="bayesian_mnist")
def train_model(cfg: DictConfig):

    data = d.MNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    hidden_size = 64
    channels, width, height = (1, 28, 28)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, 10),
    )
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
    inference = tyxe.guides.AutoNormal
    bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)

    optim = pyro.optim.Adam({"lr": cfg.params.lr})
    t0 = time.perf_counter()
    bnn.fit(
        data.train_dataloader(),
        optim,
        cfg.params.epochs,
        num_particles=cfg.params.num_particles,
    )
    elapsed = time.perf_counter() - t0

    accuracy = Accuracy()
    for x, y in data.val_dataloader():
        preds = bnn.predict(x, num_predictions=cfg.params.posterior_samples)
        acc = accuracy(preds, y)
    acc = accuracy.compute()
    print(f"Posterior accuracy: {100*acc:.2f}%")
    print(f"Training wall clock time: {elapsed}")


if __name__ == "__main__":
    train_model()
