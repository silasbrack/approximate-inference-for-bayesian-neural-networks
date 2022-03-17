from src.inference.inference import Inference
import logging
import pickle
import time
from functools import partial
from typing import Dict

import hydra
import numpy as np
import pyro
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from pyro import distributions as dist
from pyro.infer.autoguide import (
    AutoDelta,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
)
from torch import nn
from torch.nn.functional import softmax
from tqdm import tqdm

import tyxe
from src import data as d
from src.guides import AutoRadial
from tyxe.guides import AutoNormal


class VariationalInference(Inference):
    def __init__(self, model, device, guide, posterior_samples, num_particles):
        self.posterior_samples = posterior_samples
        self.num_particles = num_particles
        self.device = device

        net = model.to(device)
        likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
        inference_dict = {
            "map": AutoDelta,
            "laplace": AutoLaplaceApproximation,
            "meanfield": partial(AutoNormal, init_scale=1e-2),
            "lowrank": partial(AutoLowRankMultivariateNormal, rank=10),
            "radial": AutoRadial,
        }
        inference = inference_dict[guide]
        print(inference)
        prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(0, device=device, dtype=torch.float),
                torch.tensor(1, device=device, dtype=torch.float),
            ),
        )
        self.bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)


    def fit(self, train_loader, val_loader, epochs, lr):
        optim = pyro.optim.Adam({"lr": lr})

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
            pbar.update()

        t0 = time.perf_counter()
        # with tyxe.poutine.local_reparameterization():
        print("training")
        self.bnn.fit(
            train_loader,
            optim,
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

    def predict(self, x):
        x = x.to(self.device)
        logits = self.bnn.predict(x, num_predictions=self.posterior_samples)
        return softmax(logits, dim=-1)

    # def save(folder: str):
    #     pyro.get_param_store().save()

