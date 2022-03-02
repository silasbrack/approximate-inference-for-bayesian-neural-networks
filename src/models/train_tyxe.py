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


@hydra.main(config_path="../conf", config_name="tyxe")
def train_model(cfg: DictConfig):
    pretrained_weights = f"{cfg.paths.project}/{cfg.files.pretrained_weights}"

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.params.data](
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    hidden_size = 32
    channels, width, height = (1, 28, 28)
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, data.n_classes),
    )
    if cfg.files.pretrained_weights:
        sd = torch.load(pretrained_weights)
        print(sd)
        net.load_state_dict(sd)
    likelihood = tyxe.likelihoods.Categorical(dataset_size=60000)
    inference_dict = {
        "ml": None,
        "map": AutoDelta,
        "laplace": AutoLaplaceApproximation,
        "meanfield": partial(AutoNormal, init_scale=1e-2),
        "lowrank": partial(AutoLowRankMultivariateNormal, rank=10),
        "radial": AutoRadial,
    }
    inference = inference_dict[cfg.params.guide]
    if cfg.files.pretrained_weights and inference:
        inference = partial(
            inference,
            init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net),
        )
    prior_kwargs = (
        {"expose_all": False, "hide_all": True} if inference is None else {}
    )
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), **prior_kwargs)
    bnn = tyxe.VariationalBNN(net, prior, likelihood, inference)

    optim = pyro.optim.Adam({"lr": cfg.params.lr})
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    elbos = np.zeros((cfg.params.epochs, 1))
    val_err = np.zeros((cfg.params.epochs, 1))
    val_ll = np.zeros((cfg.params.epochs, 1))
    pbar = tqdm(total=cfg.params.epochs, unit="Epochs")

    def callback(b: tyxe.VariationalBNN, i: int, e: float):
        avg_err, avg_ll = 0.0, 0.0
        for x, y in iter(val_dataloader):
            err, ll = b.evaluate(
                x, y, num_predictions=cfg.params.posterior_samples
            )
            avg_err += err / len(val_dataloader.sampler)
            avg_ll += ll / len(val_dataloader.sampler)
        elbos[i] = e
        val_err[i] = avg_err
        val_ll[i] = avg_ll
        pbar.update()

    t0 = time.perf_counter()
    # with tyxe.poutine.local_reparameterization():
    bnn.fit(
        train_dataloader,
        optim,
        num_epochs=cfg.params.epochs,
        num_particles=cfg.params.num_particles,
        callback=callback,
    )
    elapsed = time.perf_counter() - t0
    # [print(key, val.shape) for key, val in pyro.get_param_store().items()]

    svhn_data = d.SVHNData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    svhn_data.setup()
    guide_params = sum(
        val.shape.numel() for _, val in pyro.get_param_store().items()
    )

    results = {
        "Inference": cfg.params.guide,
        "Trained on": cfg.params.data,
        "Wall clock time": elapsed,
        "Number of parameters": guide_params,
        "Training ELBO": elbos,
        "Validation accuracy": 1 - val_err,
        "Validation log-likelihood": val_ll,
        "eval_mnist": eval_model(
            bnn, data.test_dataloader(), cfg.params.posterior_samples
        ),
        "eval_svhn": eval_model(
            bnn, svhn_data.test_dataloader(), cfg.params.posterior_samples
        ),
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")
    with open(f"{cfg.params.guide}.pkl", "wb") as f:
        pickle.dump(results, f)

    # torch.save(net.state_dict(), "state_dict.pt")
    torch.save(bnn.state_dict(), "state_dict.pt")
    # optim.save("optim.pt")
    pyro.get_param_store().save("param_store.pt")
    # torch.save(bnn, "model.pt")


def eval_model(bnn, test_dataloader, posterior_samples: int) -> Dict:
    test_targets = []
    test_probs = []
    accuracy = tm.Accuracy()
    auroc = tm.AUROC(num_classes=10)
    confidence = tm.MeanMetric()
    confidence_wrong = tm.MeanMetric()
    confidence_right = tm.MeanMetric()
    nll_sum = 0
    n = 0
    for batch in test_dataloader:
        x, y = batch
        _, log_likelihood = bnn.evaluate(
            x, y, num_predictions=posterior_samples, reduction="sum"
        )
        logits = bnn.predict(x, num_predictions=posterior_samples)
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
    train_model()
