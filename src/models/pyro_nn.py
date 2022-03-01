import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
from torch.nn import functional as F


class BayesianNeuralNetwork(PyroModule):
    def __init__(self, posterior, loc, scale):
        super().__init__()
        self.flatten = nn.Flatten()

        label = "model.1.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc1 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc1.weight = PyroSample(
            posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.1.bias"
        self.fc1.bias = PyroSample(
            posterior(loc[label], scale[label]).to_event(1)
        )

        label = "model.4.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc2 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc2.weight = PyroSample(
            posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.4.bias"
        self.fc2.bias = PyroSample(
            posterior(loc[label], scale[label]).to_event(1)
        )

        label = "model.7.weight"
        weight_shape: torch.Size = loc[label].shape
        self.fc3 = PyroModule[nn.Linear](weight_shape[1], weight_shape[0])
        self.fc3.weight = PyroSample(
            posterior(loc[label], scale[label]).to_event(2)
        )
        label = "model.7.bias"
        self.fc3.bias = PyroSample(
            posterior(loc[label], scale[label]).to_event(1)
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