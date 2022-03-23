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
from torch.nn.functional import softmax
from tqdm import tqdm

import tyxe
from src import data as d
from src.data.caching import cache_dataset
from src.guides import AutoRadial
from src.models import DenseNet
from tyxe.guides import AutoNormal


@hydra.main(config_path="../../conf", config_name="tyxe")
def train_model(cfg: DictConfig):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
        "mura": d.MuraData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    device = torch.device("cuda" if cfg.hardware.gpus else "cpu")

    net = DenseNet(num_classes=data.n_classes).to(device)
    likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
    inference_dict = {
        "map": AutoDelta,
        "laplace": AutoLaplaceApproximation,
        "meanfield": partial(AutoNormal, init_scale=1e-2),
        "lowrank": partial(AutoLowRankMultivariateNormal, rank=10),
        "radial": AutoRadial,
    }
    inference = inference_dict[cfg.training.guide]
    prior = tyxe.priors.IIDPrior(
        dist.Normal(
            torch.tensor(0, device=device, dtype=torch.float),
            torch.tensor(1, device=device, dtype=torch.float),
        ),
    )
    bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)

    optim = pyro.optim.Adam({"lr": cfg.training.lr})

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    if cfg.training.cache_data:
        cache_dataset(train_dataloader.dataset.dataset)

    elbos = np.zeros((cfg.training.epochs, 1))
    val_err = np.zeros((cfg.training.epochs, 1))
    val_ll = np.zeros((cfg.training.epochs, 1))
    pbar = tqdm(total=cfg.training.epochs, unit="Epochs")

    def callback(b: tyxe.VariationalBNN, i: int, e: float):
        avg_err, avg_ll = 0.0, 0.0
        for x, y in iter(val_dataloader):
            err, ll = b.evaluate(
                x.to(device),
                y.to(device),
                num_predictions=cfg.training.posterior_samples,
            )
            err, ll = err.detach().cpu(), ll.detach().cpu()
            avg_err += err / len(val_dataloader.sampler)
            avg_ll += ll / len(val_dataloader.sampler)
        elbos[i] = e
        val_err[i] = avg_err
        val_ll[i] = avg_ll
        pbar.update()

    t0 = time.perf_counter()
    # with tyxe.poutine.local_reparameterization():
    logging.info("Training.")
    bnn.fit(
        train_dataloader,
        optim,
        num_epochs=cfg.training.epochs,
        num_particles=cfg.training.num_particles,
        callback=callback,
        device=device,
    )
    elapsed = time.perf_counter() - t0

    guide_params = sum(
        val.shape.numel() for _, val in pyro.get_param_store().items()
    )

    results = {
        "Inference": cfg.training.guide,
        "Trained on": cfg.training.dataset,
        "Wall clock time": elapsed,
        "Number of parameters": guide_params,
        "Training ELBO": elbos,
        "Validation accuracy": 1 - val_err,
        "Validation log-likelihood": val_ll,
    }
    for eval_dataset in cfg.eval.datasets:
        eval_data = data_dict[eval_dataset](
            cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
        )
        eval_data.setup()
        results[f"eval_{eval_dataset}"] = eval_model(
            bnn,
            eval_dataset,
            eval_data.test_dataloader(),
            cfg.training.posterior_samples,
            data.n_classes,
            device,
        )

    for metric, value in results.items():
        print(f"{metric}: {value}")
    with open(f"{cfg.training.guide}.pkl", "wb") as f:
        pickle.dump(results, f)


def eval_model(
    bnn,
    dataset: str,
    test_dataloader,
    posterior_samples: int,
    n_classes: int,
    device,
) -> Dict:
    test_targets = []
    test_probs = []
    accuracy = tm.Accuracy()
    auroc = tm.AUROC(n_classes)
    confidence = tm.MeanMetric()
    confidence_wrong = tm.MeanMetric()
    confidence_right = tm.MeanMetric()
    nll_sum = 0
    n = 0
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        _, log_likelihood = bnn.evaluate(
            x, y, num_predictions=posterior_samples, reduction="sum"
        )
        logits = bnn.predict(x, num_predictions=posterior_samples)
        log_likelihood = log_likelihood.detach().cpu()
        logits = logits.detach().cpu()
        y = y.cpu()
        probs = softmax(logits, dim=-1)
        conf, preds = torch.max(probs, dim=-1)
        preds = preds.detach()
        conf = conf.detach()
        right = torch.where(preds == y)
        wrong = torch.where(preds != y)
        test_targets.append(y)
        test_probs.append(probs)
        confidence(conf)
        confidence_wrong(conf[wrong])
        confidence_right(conf[right])
        accuracy(logits, y)
        auroc(logits, y)
        nll_sum += -log_likelihood
        n += y.shape[0]
    test_targets: np.array = torch.cat(test_targets).numpy()
    test_probs: np.array = torch.cat(test_probs).numpy()

    return {
        "Evaluated on": dataset,
        "NLL": nll_sum.item() / n,
        "Accuracy": accuracy.compute().item(),
        "AUROC": auroc.compute().item(),
        "Average confidence": confidence.compute().item(),
        "Average confidence when wrong": confidence_wrong.compute().item(),
        "Average confidence when right": confidence_right.compute().item(),
        "Test targets": test_targets,
        "Test probabilities": test_probs,
    }


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)

    train_model()
