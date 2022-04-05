import copy
import os
import pickle
import time
from typing import Dict

import pyro
from pyro.distributions import Categorical, Normal
from pyro.infer import Predictive
from pyro.nn.module import PyroModule, PyroSample, to_pyro_module_
from torch.nn import Sequential
from torch.nn.functional import log_softmax, softmax

from src.inference.inference import Inference
from src.inference.swa import Swa


class SwagModule(PyroModule):
    def __init__(self, model: PyroModule[Sequential], loc: Dict, scale: Dict):
        super().__init__()
        self.model = model
        self.set_priors(loc, scale)

    def set_priors(self, loc, scale):
        i = 0
        keys = list(loc.keys())
        for layer in self.model:
            if hasattr(layer, "weight"):
                weight_label = keys[i]
                layer.weight = PyroSample(
                    Normal(loc[weight_label], scale[weight_label]).to_event(2)
                )
                bias_label = keys[i + 1]
                layer.bias = PyroSample(
                    Normal(loc[bias_label], scale[bias_label]).to_event(1)
                )
                i += 2

    def forward(self, x, y=None):
        x = self.model(x)
        logits = pyro.deterministic("logits", log_softmax(x, dim=-1))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", Categorical(logits=logits), obs=y)
        return logits


class Swag(Inference):
    def __init__(self, model, device, swa_start_thresh, posterior_samples):
        self.name = "SWAG"
        self.model = model
        self.device = device
        self.posterior_samples = posterior_samples
        self.swa_model = Swa(model, device, swa_start_thresh)
        self.pyro_model = copy.deepcopy(model)
        to_pyro_module_(self.pyro_model)
        self.weight_loc = None
        self.weight_scale = None
        self.swag_model = None
        self.predictive = None

    def fit(self, train_loader, val_loader, epochs, lr):
        t0 = time.perf_counter()
        self.swa_model.fit(train_loader, val_loader, epochs, lr)
        elapsed = time.perf_counter() - t0
        state_dicts = self.swa_model.state_dicts
        self.weight_loc = {
            key: state_dicts[key].mean(dim=0) for key in state_dicts.keys()
        }
        self.weight_scale = {
            key: state_dicts[key].std(dim=0) + 1e-4
            for key in state_dicts.keys()
        }
        self.swag_model = SwagModule(
            self.pyro_model, self.weight_loc, self.weight_scale
        )
        self.predictive = Predictive(
            self.swag_model, num_samples=self.posterior_samples
        )
        return {"Wall clock time": elapsed}

    def predict(self, x, aggregate=True):
        x = x.to(self.device)
        logits = self.predictive(x)["logits"]
        probs = softmax(logits, dim=-1)
        if aggregate:
            probs = probs.mean(dim=0)
        return probs.squeeze()

    def save(self, path: str):
        with open(os.path.join(path, "state_dicts.pkl"), "wb") as f:
            pickle.dump(
                {"loc": self.weight_loc, "scale": self.weight_scale}, f
            )

    def load(self, path: str):
        with open(os.path.join(path, "state_dicts.pkl"), "rb") as f:
            state_dicts = pickle.load(f)
        self.weight_loc = state_dicts["loc"]
        self.weight_scale = state_dicts["scale"]
        self.swag_model = SwagModule(
            self.pyro_model, self.weight_loc, self.weight_scale
        )
        self.predictive = Predictive(
            self.swag_model, num_samples=self.posterior_samples
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
