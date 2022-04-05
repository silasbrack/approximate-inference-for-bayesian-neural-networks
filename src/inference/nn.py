import os
import time

import torch
from torch.nn import functional as F

from src.inference.inference import Inference


class NeuralNetwork(Inference):
    def __init__(self, model, device):
        self.name = "MAP"
        self.model = model.to(device)
        self.device = device

    def fit(self, train_loader, val_loader, epochs, lr):
        optim = torch.optim.Adam(self.model.parameters(), lr)
        t0 = time.perf_counter()
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                logits = self.model(x)
                loss = F.nll_loss(logits, y)
                loss.backward()
                optim.step()
        elapsed = time.perf_counter() - t0
        return {"Wall clock time": elapsed}

    # TODO: How to implement aggregate
    def predict(self, x, aggregate=True):
        x = x.to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=-1)

    def save(self, path: str) -> None:
        torch.save(
            self.model.state_dict(),
            os.path.join(path, "state_dict.pt"),
        )

    def load(self, path: str):
        state_dict = torch.load(os.path.join(path, "state_dict.pt"))
        self.model.load_state_dict(state_dict)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
