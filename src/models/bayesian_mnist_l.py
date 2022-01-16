from collections import defaultdict
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy

import pytorch_lightning as pl

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
import torch


class CustomSVIWrapper(torch.optim.Optimizer):
    def __init__(self, params, defaults, svi: SVI):
        self.svi: SVI = svi
        self.state = defaultdict(dict)
        self.param_groups = []


class BayesianMnistModelLightning(PyroModule, pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        hidden_size = 64

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.lr = lr

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

        self.guide = AutoDiagonalNormal(self)

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.optimizers().step(*batch)
        loss = torch.tensor(self.optimizers().svi.step(*batch)) / x.shape[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = (
            self.optimizers().svi.loss(self, self.guide, *batch) / x.shape[0]
        )

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.optimizers().svi.loss(self, self.guide, *batch)

        prediction = Predictive(self, guide=self.guide, num_samples=512)(x)
        preds = prediction["obs"].mode(dim=0).values  # MAP prediction
        acc = self.accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        adam = pyro.optim.Adam({"lr": self.lr})
        svi = SVI(self, self.guide, adam, loss=Trace_ELBO())
        optimizer = CustomSVIWrapper(None, None, svi=svi)
        return optimizer
