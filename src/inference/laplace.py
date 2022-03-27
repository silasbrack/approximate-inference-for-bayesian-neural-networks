import logging
import os
import pickle
import time

import laplace
import torch

from src.inference import NeuralNetwork
from src.inference.inference import Inference


# TODO: Get Laplace predict.py working correctly with all weights.
# TODO: Get Laplace saving last_layer working.
class Laplace(Inference):
    def __init__(self,
                 model,
                 device,
                 subset: str,
                 hessian: str):
        self.name = "Laplace"
        # The Laplace library has a likelihood which takes logits as inputs and
        # can't handle Softmax or LogSoftmax layers.
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.nn = NeuralNetwork(model, device)
        self.device = device
        self.la = laplace.Laplace(self.model,
                                  "classification",
                                  subset_of_weights=subset,
                                  hessian_structure=hessian)

    def fit(self, train_loader, val_loader, epochs, lr):
        t0 = time.perf_counter()
        logging.info("Finding MAP solution.")
        self.nn.fit(train_loader, val_loader, epochs, lr)
        logging.info("Calculating Hessian.")
        self.la.fit(train_loader)
        logging.info("Optimizing prior precision.")
        self.la.optimize_prior_precision()
        elapsed = time.perf_counter() - t0
        return {"Wall clock time": elapsed}

    def predict(self, x: torch.Tensor):
        return self.la(x)

    def save(self, path: str):
        with open(os.path.join(path, "la.pkl"), "wb") as f:
            pickle.dump(self.la, f)

    def load(self, path: str):
        with open(os.path.join(path, "la.pkl"), "rb") as f:
            self.la = pickle.load(f)

    @property
    def num_params(self):
        return self.model.num_params
