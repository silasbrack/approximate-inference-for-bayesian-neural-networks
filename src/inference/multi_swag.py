import copy
import os
import pickle
import time

import hydra.utils
import torch
from torch import nn
from pyro.infer import Predictive

from src.inference.inference import Inference
from src.inference.swag import Swag, SwagModule


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


# TODO: Implement efficient SWAG calculations from Algorithm 1 in Maddox 2019
class MultiSwag(Inference):
    def __init__(
        self,
        model,
        device,
        num_ensembles: int,
        swa_start_thresh: float,
        posterior_samples: int,
    ):
        self.name = f"MultiSWAG@{num_ensembles}"
        self.num_ensembles = num_ensembles
        self.posterior_samples = posterior_samples
        self.device = device
        self.ensembles = []
        for _ in range(num_ensembles):
            model_ = copy.deepcopy(model)
            model_.apply(weight_reset)
            self.ensembles.append(
                Swag(
                    hydra.utils.instantiate(model_),
                    device,
                    swa_start_thresh,
                    posterior_samples,
                )
            )

    def fit(self, train_loader, val_loader, epochs, lr):
        t0 = time.perf_counter()
        for ensemble in self.ensembles:
            ensemble.fit(train_loader, val_loader, epochs, lr)
        elapsed = time.perf_counter() - t0
        return {"Wall clock time": elapsed}

    def predict_ensembles(self, x, aggregate):
        return torch.stack(
            [ensemble.predict(x, aggregate) for ensemble in self.ensembles]
        )

    def predict(self, x, aggregate=True):
        probs = self.predict_ensembles(x, aggregate)
        ensemble_probs = torch.mean(probs, dim=0)
        return ensemble_probs

    def save(self, path: str):
        state_dicts = [
            {"loc": ensemble.weight_loc, "scale": ensemble.weight_scale}
            for ensemble in self.ensembles
        ]
        with open(os.path.join(path, "state_dicts.pkl"), "wb") as f:
            pickle.dump(state_dicts, f)

    def load(self, path: str):
        with open(os.path.join(path, "state_dicts.pkl"), "rb") as f:
            state_dicts = pickle.load(f)
        for swag, state_dict in zip(self.ensembles, state_dicts):
            weight_loc = state_dict["loc"]
            weight_scale = state_dict["scale"]
            swag.weight_loc = weight_loc
            swag.weight_scale = weight_scale
            swag.swag_model = SwagModule(
                swag.pyro_model, weight_loc, weight_scale
            )
            swag.predictive = Predictive(
                swag.swag_model, num_samples=self.posterior_samples
            )

    @property
    def num_params(self):
        return sum(ensemble.num_params for ensemble in self.ensembles)
