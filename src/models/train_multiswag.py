import hydra
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pyro.nn import PyroModule, PyroSample
from torch import distributions, Tensor
from torch.nn import functional as F

from src.models.train_swag import train_swag


class BayesianMnistModel(PyroModule):
    def __init__(self, loc, scale):
        super().__init__()
        self.flatten = nn.Flatten()

        label = "model.1.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc1 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc1.weight = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.1.bias"
        self.fc1.bias = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(1)
        )

        label = "model.4.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc2 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc2.weight = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.4.bias"
        self.fc2.bias = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(1)
        )

        label = "model.7.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc3 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc3.weight = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.7.bias"
        self.fc3.bias = PyroSample(
            get_posterior(loc[label], scale[label]).to_event(1)
        )

    def forward(self, x, y=None):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        logits = pyro.deterministic("logits", F.log_softmax(x, dim=-1))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def get_posterior(ensemble_locs: Tensor, ensemble_scales: Tensor):
    num_ensembles: int = ensemble_locs.shape[0]
    with pyro.plate("data", len(data)):
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        return dist.Normal(locs[assignment], scale)


# def get_posterior(ensemble_locs: Tensor, ensemble_scales: Tensor) -> distributions.MixtureSameFamily:
#     num_ensembles: int = ensemble_locs.shape[0]
#
#     # In MultiSWAG we weigh each ensemble equally
#     mix = dist.Categorical(torch.ones((num_ensembles,)))
#
#     comp = dist.Independent(dist.Normal(ensemble_locs, ensemble_scales), 1)
#
#     # MultiSWAG creates a GMM with {num_ensembles} components
#     gaussian_mixture = distributions.MixtureSameFamily(mix, comp)
#     return gaussian_mixture


@hydra.main(config_path="../conf", config_name="swa")
def run(cfg: DictConfig):
    num_ensembles = 3

    ensemble_state_dicts = [train_swag(cfg) for _ in range(num_ensembles)]
    print(ensemble_state_dicts)
    ensemble_locs = {k: torch.stack([sd[0][k] for sd in ensemble_state_dicts]) for k in ensemble_state_dicts[0]}
    ensemble_scales = {k: torch.stack([sd[1][k] for sd in ensemble_state_dicts]) for k in ensemble_state_dicts[0]}

    model = BayesianMnistModel(ensemble_locs, ensemble_scales)


if __name__ == "__main__":
    run()
