import time

import torch
from torch.nn import functional as F

from src.inference.inference import Inference
from src.inference.nn import NeuralNetwork


class DeepEnsemble(Inference):
    def __init__(self, model, device, num_ensembles: int):
        self.ensembles = [NeuralNetwork(model, device) for _ in range(num_ensembles)]
    
    def fit(self, train_loader, val_loader, epochs, lr):
        t0 = time.perf_counter()
        for ensemble in self.ensembles:
            ensemble.fit(train_loader, val_loader, epochs, lr)
        elapsed = time.perf_counter() - t0

        return {
            "Wall clock time": elapsed,
        }
    
    def predict(self, x):
        ensemble_probs = torch.stack([ensemble.predict(x) for ensemble in self.ensembles])
        probs = torch.mean(ensemble_probs, dim=0)
        return probs
