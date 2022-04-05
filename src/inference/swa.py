import copy
import os
import time
from collections import OrderedDict

import torch
from torch.nn.functional import nll_loss, softmax
from torch.optim.swa_utils import AveragedModel

from src.inference.inference import Inference


class Swa(Inference):
    def __init__(self, model, device, swa_start_thresh):
        self.name = "SWA"
        self.model = model.to(device)
        self.device = device
        self.swa_model = AveragedModel(model).to(device)
        self.swa_start_thresh = swa_start_thresh
        self.state_dicts = None

    def fit(self, train_loader, val_loader, epochs, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nll_loss
        swa_start = int(self.swa_start_thresh * epochs)

        state_dicts = []
        t0 = time.perf_counter()
        for epoch in range(epochs):
            for image, target in train_loader:
                optimizer.zero_grad()
                loss_fn(self.model(image), target).backward()
                optimizer.step()
            if epoch >= swa_start:
                self.swa_model.update_parameters(self.model)
                state_dicts.append(copy.deepcopy(self.model.state_dict()))
        elapsed = time.perf_counter() - t0
        self.state_dicts = {
            key: torch.stack([sd[key] for sd in state_dicts])
            for key in state_dicts[0]
        }
        return {"Wall clock time": elapsed}

    # TODO: How to implement aggregate
    def predict(self, x, aggregate=True):
        x = x.to(self.device)
        logits = self.swa_model(x)
        return softmax(logits, dim=-1)

    def save(self, path: str) -> None:
        torch.save(
            self.state_dicts,
            os.path.join(path, "state_dict.pt"),
        )

    def load(self, path: str):
        self.state_dicts = torch.load(os.path.join(path, "state_dict.pt"))
        n_averaged = torch.tensor(
            list(self.state_dicts.items())[0][1].shape[0]
        )
        state_dict = OrderedDict(
            {
                f"module.{key}": val.mean(dim=0)
                for key, val in self.state_dicts.items()
            }
        )
        state_dict["n_averaged"] = n_averaged
        self.swa_model.load_state_dict(state_dict)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
