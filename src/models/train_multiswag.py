import hydra
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pyro.nn import PyroModule, PyroSample
from torch.nn import functional as F
from functools import partial
from torch import distributions
from torch.distributions import transforms
from pyro_nn import BayesianNeuralNetwork
from torchmetrics import Accuracy
from pyro.infer import Predictive

from src import data as d
from src.models.train_swag import train_swag


class MultiSwagModel(PyroModule):
    def __init__(self, num_ensembles, loc, scale):
        super().__init__()
        self.num_ensembles = num_ensembles
        component_logits = torch.ones((self.num_ensembles,))

        self.flatten = nn.Flatten()

        label = "model.1.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc1 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc1.weight = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
        )
        label = "model.1.bias"
        self.fc1.bias = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
        )

        label = "model.4.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc2 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc2.weight = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
        )
        label = "model.4.bias"
        self.fc2.bias = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
        )

        label = "model.7.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc3 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc3.weight = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
        )
        label = "model.7.bias"
        self.fc3.bias = PyroSample(
            get_posterior(
                loc[label], scale[label],
            )
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


def get_posterior(ensemble_locs, ensemble_scales):
    num_ensembles: int = ensemble_locs.shape[0]

    # In MultiSWAG we weigh each ensemble equally
    mix = dist.Categorical(torch.ones((num_ensembles,)))

    unflattened_shape = ensemble_locs.shape[1:]
    locs_flattened = torch.flatten(ensemble_locs, start_dim=1)
    flattened_shape = locs_flattened.shape[1:]
    scales_flattened = torch.flatten(ensemble_scales, start_dim=1)
    comp = dist.Independent(dist.Normal(locs_flattened, scales_flattened), 1)

    # MultiSWAG creates a GMM with {num_ensembles} components
    gaussian_mixture = distributions.MixtureSameFamily(mix, comp)
    reshape_transform = transforms.ReshapeTransform(in_shape=flattened_shape, out_shape=unflattened_shape)
    transformed = dist.TransformedDistribution(gaussian_mixture, reshape_transform)
    return transformed


@hydra.main(config_path="../conf", config_name="swa")
def run(cfg: DictConfig):
    num_ensembles = cfg.num_ensembles

    ensemble_state_dicts = []
    for _ in range(num_ensembles):
        ensemble_state_dicts.append(train_swag(cfg))

    # ensemble_locs={k: torch.stack([sd[0][k] for sd in ensemble_state_dicts])
    #                  for k in ensemble_state_dicts[0][0]}
    # ensemble_scales = {k: torch.stack([sd[1][k]
    #                                    for sd in ensemble_state_dicts])
    #                    for k in ensemble_state_dicts[0][0]}
    ensemble_locs = {}
    ensemble_scales = {}
    for key in ensemble_state_dicts[0][0].keys():
        ensemble_locs[key] = []
        ensemble_scales[key] = []
        for state_dict in ensemble_state_dicts:
            ensemble_locs[key].append(state_dict[0][key])
            ensemble_scales[key].append(state_dict[1][key])
        ensemble_locs[key] = torch.stack(ensemble_locs[key])
        ensemble_scales[key] = torch.stack(ensemble_scales[key])

    multi_swag_model = MultiSwagModel(num_ensembles, ensemble_locs, ensemble_scales)
    # posterior = partial(dist.MixtureOfDiagNormals,
    #                     component_logits=torch.ones((num_ensembles,)))
    # multi_swag_model = BayesianNeuralNetwork(prior=posterior, loc=ensemble_locs, scale=ensemble_scales)
    posterior_predictive = Predictive(multi_swag_model, num_samples=128)

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

    accuracy_calculator = Accuracy()
    for image, target in data.test_dataloader():
        prediction = posterior_predictive(image)
        # preds = prediction["obs"].mean(dim=0).values
        logits = prediction["logits"].mode(dim=0).values.squeeze(0)
        accuracy_calculator(logits, target)
    accuracy = accuracy_calculator.compute()
    accuracy_calculator.reset()
    print(f"Test accuracy for MultiSWAG = {100 * accuracy:.2f}")


if __name__ == "__main__":
    run()
