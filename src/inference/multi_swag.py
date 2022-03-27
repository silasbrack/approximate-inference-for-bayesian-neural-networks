import os
import pickle
import time

import torch
from pyro.infer import Predictive

from src.inference.inference import Inference
from src.inference.swag import Swag, SwagModule


class MultiSwag(Inference):
    def __init__(self,
                 model,
                 device,
                 num_ensembles: int,
                 swa_start_thresh: float,
                 posterior_samples: int):
        self.model = model
        self.num_ensembles = num_ensembles
        self.posterior_samples = posterior_samples
        self.device = device
        self.ensembles = [
            Swag(model, device, swa_start_thresh, posterior_samples)
            for _ in range(num_ensembles)
        ]

    def fit(self, train_loader, val_loader, epochs, lr):
        t0 = time.perf_counter()
        for ensemble in self.ensembles:
            ensemble.fit(train_loader, val_loader, epochs, lr)
        elapsed = time.perf_counter() - t0
        return {"Wall clock time": elapsed}

    def predict_ensembles(self, x):
        return torch.stack(
            [ensemble.predict(x) for ensemble in self.ensembles]
        )

    def predict(self, x):
        ensemble_probs = self.predict_ensembles(x)
        probs = torch.mean(ensemble_probs, dim=0)
        return probs

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
            swag.swag_model = SwagModule(swag.pyro_model,
                                         weight_loc,
                                         weight_scale)
            swag.predictive = Predictive(swag.swag_model,
                                         num_samples=self.posterior_samples)

    @property
    def num_params(self):
        return self.model.num_params * self.num_ensembles
