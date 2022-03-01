import hydra
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pyro.infer import Predictive
from pyro.nn import PyroModule, PyroSample
from torch.nn import functional as F
from torchmetrics import Accuracy

from src import data as d
from src.models.pyro_nn import BayesianNeuralNetwork
from src.models.train_swa import train_swa


class SwagModel(PyroModule):
    def __init__(self, loc, scale):
        super().__init__()
        self.flatten = nn.Flatten()

        label = "model.1.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc1 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc1.weight = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(2)
        )
        label = "model.1.bias"
        self.fc1.bias = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(1)
        )

        label = "model.4.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc2 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc2.weight = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(2)
        )
        label = "model.4.bias"
        self.fc2.bias = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(1)
        )

        label = "model.7.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc3 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc3.weight = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(2)
        )
        label = "model.7.bias"
        self.fc3.bias = PyroSample(
            dist.Normal(loc[label], scale[label]).to_event(1)
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


def train_swag(cfg: DictConfig):
    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.params.data](
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    state_dicts = train_swa(cfg)
    weight_loc = {key: state_dicts[key].mean(dim=0) for key in state_dicts.keys()}
    weight_scale = {key: state_dicts[key].std(dim=0) + 0.0001 for key in state_dicts.keys()}

    swag_model = SwagModel(weight_loc, weight_scale)
    # posterior = dist.Normal
    # multi_swag_model = BayesianNeuralNetwork(prior=posterior)
    posterior_predictive = Predictive(swag_model, num_samples=128)

    accuracy_calculator = Accuracy()
    for image, target in data.test_dataloader():
        prediction = posterior_predictive(image)
        # preds = prediction["obs"].mode(dim=0).values  # MAP prediction
        logits = prediction["logits"].mode(dim=0).values.squeeze(0)
        accuracy_calculator(logits, target)
    accuracy = accuracy_calculator.compute()
    accuracy_calculator.reset()
    print(f"Test accuracy for SWAG = {100*accuracy:.2f}")

    return weight_loc, weight_scale


@hydra.main(config_path="../conf", config_name="swa")
def run(cfg: DictConfig):
    train_swag(cfg)


if __name__ == "__main__":
    run()
