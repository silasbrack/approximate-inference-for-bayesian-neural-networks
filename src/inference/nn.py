import os
import time

import torch
from torch.nn import functional as F
import tqdm
import wandb

from src.inference.inference import Inference


class NeuralNetwork(Inference):
    def __init__(self, model, device, prior):
        if prior:
            self.name = "MAP"
            self.weight_decay = 4e-3  # TODO: Technically not N(0,1) prior?
        else:
            self.name = "ML"
            self.weight_decay = 0.0
        self.model = model.to(device)
        self.device = device
        self.optim = None

    def fit(self, train_loader, val_loader, epochs, lr):
        if self.optim is None:
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr, weight_decay=self.weight_decay
            )
        self.model.train()
        t0 = time.perf_counter()
        for epoch in tqdm.tqdm(range(epochs)):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optim.zero_grad()
                logits = self.model(x)
                loss = F.nll_loss(logits, y)
                loss.backward()
                self.optim.step()
                wandb.log({"Epoch": epoch, "Training loss": loss.item()})
        elapsed = time.perf_counter() - t0
        return {"Wall clock time": elapsed}

    def predict(self, x, aggregate=True):
        if not aggregate:
            raise NotImplementedError
        x = x.to(self.device)
        self.model.eval()
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
