import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy


class BayesianMnistModel(PyroModule):
    def __init__(self, lr: float):
        super().__init__()

        hidden_size = 64

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = lr

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        input_size = channels * width * height

        self.flatten = nn.Flatten()
        self.fc1 = PyroModule[nn.Linear](input_size, hidden_size)
        self.fc1.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size, input_size]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size]).to_event(1)
        )
        self.fc2 = PyroModule[nn.Linear](hidden_size, hidden_size)
        self.fc2.weight = PyroSample(
            dist.Normal(0.0, 1.0)
            .expand([hidden_size, hidden_size])
            .to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size]).to_event(1)
        )
        self.fc3 = PyroModule[nn.Linear](hidden_size, self.num_classes)
        self.fc3.weight = PyroSample(
            dist.Normal(0.0, 1.0)
            .expand([self.num_classes, hidden_size])
            .to_event(2)
        )
        self.fc3.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([self.num_classes]).to_event(1)
        )
        self.accuracy = Accuracy()

    def forward(self, x, y=None):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = pyro.deterministic(
            "logits", F.log_softmax(self.fc3(x), dim=-1)
        )
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
