import os
import time

import numpy as np
import pyro
import torch
import tyxe
from pyro import distributions as dist
from torch.nn.functional import softmax
from tqdm import tqdm
import wandb

from src.inference.inference import Inference


class VariationalInference(Inference):
    def __init__(
        self,
        model,
        device,
        variational_family,
        posterior_samples,
        num_particles,
        local_reparameterization,
    ):
        self.name = f"{variational_family.name}"
        self.posterior_samples = posterior_samples
        self.num_particles = num_particles
        self.local_reparameterization = local_reparameterization
        self.device = device

        net = model.to(device)
        likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
        inference = variational_family.guide()
        prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(0, device=device, dtype=torch.float),
                torch.tensor(1, device=device, dtype=torch.float),
            ),
        )
        self.bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)
        self.optim = None

    def fit(self, train_loader, val_loader, epochs, lr):
        self.bnn.likelihood.dataset_size = len(train_loader.sampler)
        if self.optim is None:
            self.optim = pyro.optim.Adam({"lr": lr})

        elbos = np.zeros((epochs, 1))
        val_err = np.zeros((epochs, 1))
        val_ll = np.zeros((epochs, 1))
        pbar = tqdm(total=epochs, unit="Epochs")

        def callback(b: tyxe.VariationalBNN, i: int, e: float):
            avg_err, avg_ll = 0.0, 0.0
            for x, y in iter(val_loader):
                err, ll = b.evaluate(
                    x.to(self.device),
                    y.to(self.device),
                    num_predictions=self.posterior_samples,
                )
                err, ll = err.detach().cpu(), ll.detach().cpu()
                avg_err += err / len(val_loader.sampler)
                avg_ll += ll / len(val_loader.sampler)
            elbos[i] = e
            val_err[i] = avg_err
            val_ll[i] = avg_ll
            wandb.log(
                {
                    "Epoch": i,
                    "ELBO": elbos[i],
                    "Validation error": avg_err,
                    "Validation LL": avg_ll,
                }
            )
            pbar.update()

        t0 = time.perf_counter()
        if self.local_reparameterization:
            with tyxe.poutine.local_reparameterization():
                self.bnn.fit(
                    train_loader,
                    self.optim,
                    num_epochs=epochs,
                    num_particles=self.num_particles,
                    callback=callback,
                    device=self.device,
                )
        else:
            self.bnn.fit(
                train_loader,
                self.optim,
                num_epochs=epochs,
                num_particles=self.num_particles,
                callback=callback,
                device=self.device,
            )
        elapsed = time.perf_counter() - t0

        return {
            "Wall clock time": elapsed,
            "Training ELBO": elbos,
            "Validation accuracy": 1 - val_err,
            "Validation log-likelihood": val_ll,
        }

    def predict(self, x, aggregate=True):
        x = x.to(self.device)
        logits = self.bnn.predict(
            x, num_predictions=self.posterior_samples, aggregate=aggregate
        )
        return softmax(logits, dim=-1)

    def save(self, path: str) -> None:
        pyro.get_param_store().save(os.path.join(path, "param_store.pt"))

    def load(self, path: str):
        pyro.get_param_store().load(os.path.join(path, "param_store.pt"))

    @property
    def num_params(self):
        return sum(
            val.shape.numel() for _, val in pyro.get_param_store().items()
        )

    def update_prior(self) -> None:
        self.bnn.update_prior(
            tyxe.priors.DictPrior(
                self.bnn.net_guide.get_detached_distributions(
                    tyxe.util.pyro_sample_sites(self.bnn.net)
                )
            )
        )
