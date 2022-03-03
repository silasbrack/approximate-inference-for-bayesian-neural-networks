import pickle
import time
from typing import List

import hydra
import numpy as np
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from pyro.infer import Predictive
from torch import softmax
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from src import data as d
from src.models.train_swag import train_swag


# TODO: Make MultiSWAG work on GPU
@hydra.main(config_path="../../conf", config_name="multiswag")
def run(cfg: DictConfig):
    num_ensembles = cfg.num_ensembles

    swag_models = []
    t0 = time.perf_counter()
    for _ in range(num_ensembles):
        swag_model = train_swag(cfg)
        swag_models.append(swag_model)
    elapsed = time.perf_counter() - t0

    posterior_predictives = []
    for swag_model in swag_models:
        posterior_predictive = Predictive(swag_model, num_samples=128)
        posterior_predictives.append(posterior_predictive)

    params = num_ensembles * sum(
            p.numel() for p in swag_models[0].parameters())

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    results = {
        "Inference": "multiswag",
        "Trained on": cfg.training.dataset,
        "Wall clock time": elapsed,
        "Number of parameters": params,
    }
    for eval_dataset in cfg.eval.datasets:
        eval_data = data_dict[eval_dataset](
            cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
        )
        eval_data.setup()
        results[f"eval_{eval_dataset}"] = eval_model(
            posterior_predictives, eval_dataset, eval_data.test_dataloader(),
            num_ensembles,
        )

    for metric, value in results.items():
        print(f"{metric}: {value}")
    with open(f"multiswag.pkl", "wb") as f:
        pickle.dump(results, f)

    accuracy_calculator = tm.Accuracy()
    for image, target in data.test_dataloader():
        logits = None
        for posterior_predictive in posterior_predictives:
            prediction = posterior_predictive(image)
            swag_logits = prediction["logits"]
            if logits is None:
                logits = swag_logits
            else:
                logits += swag_logits
        logits /= num_ensembles
        logits = logits.mean(dim=0).squeeze(0)
        print(logits.shape)
        accuracy_calculator(logits, target)
    accuracy = accuracy_calculator.compute()
    accuracy_calculator.reset()
    print(f"Test accuracy for MultiSWAG = {100 * accuracy:.2f}")


def eval_model(models: List, dataset: str, test_dataloader: DataLoader,
               num_ensembles: int):
    test_targets = []
    test_probs = []
    accuracy = tm.Accuracy()
    auroc = tm.AUROC(num_classes=10)
    confidence = tm.MeanMetric()
    confidence_wrong = tm.MeanMetric()
    confidence_right = tm.MeanMetric()
    nll_loss = NLLLoss(reduction="sum")
    nll_sum = 0
    n = 0
    for x, y in test_dataloader:
        logits = None
        for posterior_predictive in models:
            prediction = posterior_predictive(x)
            swag_logits = prediction["logits"]  # Do we already take the mean?
            if logits is None:
                logits = swag_logits
            else:
                logits += swag_logits
        logits /= num_ensembles
        logits = logits.mean(dim=0).squeeze(0)
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
        accuracy(probs, y)
        auroc(probs, y)
        nll_sum += nll_loss(probs, y)
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
    run()
