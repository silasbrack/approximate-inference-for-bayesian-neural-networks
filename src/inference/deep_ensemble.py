import copy
import os
import time

import torch
from torch import nn

from src.inference.inference import Inference
from src.inference.nn import NeuralNetwork


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class DeepEnsemble(Inference):
    def __init__(self, model, device, num_ensembles: int):
        self.name = f"Ensemble@{num_ensembles}"
        self.num_ensembles = num_ensembles
        self.device = device
        self.ensembles = []
        for _ in range(num_ensembles):
            model_ = copy.deepcopy(model)
            model_.apply(weight_reset)
            self.ensembles.append(NeuralNetwork(model_, device, prior=False))

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

    def predict(self, x, aggregate=True):
        if not aggregate:
            raise NotImplementedError
        probs = self.predict_ensembles(x)
        ensemble_probs = torch.mean(probs, dim=0)
        return ensemble_probs

    def save(self, path: str):
        state_dicts = [
            ensemble.model.state_dict() for ensemble in self.ensembles
        ]
        torch.save(state_dicts, os.path.join(path, "state_dicts.pt"))

    def load(self, path: str):
        state_dicts = torch.load(
            os.path.join(path, "state_dicts.pt"), map_location=self.device
        )
        for ensemble, state_dict in zip(self.ensembles, state_dicts):
            ensemble.model.load_state_dict(state_dict)

    @property
    def num_params(self):
        return sum(ensemble.num_params for ensemble in self.ensembles)
