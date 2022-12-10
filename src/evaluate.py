import warnings
from typing import Dict

import numpy as np
import torch
import torchmetrics as tm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.inference.inference import Inference


def evaluate(
    inference: Inference,
    test_loader: DataLoader,
    dataset: str,
    num_classes: int,
) -> Dict:
    warnings.filterwarnings(
        "ignore",
        message="Metric `AUROC` will save all targets "
        "and predictions in buffer. For large "
        "datasets this may lead to large memory "
        "footprint.",
    )

    test_targets = []
    test_probs = []
    accuracy = tm.Accuracy()
    auroc = tm.AUROC(num_classes)
    confidence = tm.MeanMetric()
    confidence_wrong = tm.MeanMetric()
    confidence_right = tm.MeanMetric()
    nll = tm.MeanMetric()

    for x, y in test_loader:
        probs = inference.predict(x).detach().cpu()
        conf, preds = torch.max(probs, dim=-1)
        right = torch.where(preds == y)
        wrong = torch.where(preds != y)
        test_targets.append(y)
        test_probs.append(probs)
        confidence(conf)
        confidence_wrong(conf[wrong])
        confidence_right(conf[right])
        accuracy(probs, y)
        auroc(probs, y)
        nll(F.nll_loss(probs, y))

    test_targets: np.array = torch.cat(test_targets).numpy()
    test_probs: np.array = torch.cat(test_probs).numpy()

    return {
        "Inference": inference.name,
        "Evaluated on": dataset,
        "NLL": nll.compute().item(),
        "Accuracy": accuracy.compute().item(),
        "AUROC": auroc.compute().item(),
        "Average confidence": confidence.compute().item(),
        "Average confidence when wrong": confidence_wrong.compute().item(),
        "Average confidence when right": confidence_right.compute().item(),
        "Test targets": test_targets,
        "Test probabilities": test_probs,
    }


def evaluate_accuracy(inference: Inference, loader: DataLoader) -> Dict:
    accuracy = tm.Accuracy()
    nll = tm.MeanMetric()
    for x, y in iter(loader):
        probs = inference.predict(x).detach().cpu()
        accuracy(probs, y)
        nll(F.nll_loss(probs, y))
    return {
        "Validation accuracy": accuracy.compute().item(),
        "Validation NLL": nll.compute().item(),
    }


def print_dict(dictionary: Dict):
    for key, val in dictionary.items():
        if type(val) is np.ndarray:
            print(f"{key}:", val.shape)
        elif type(val) is float:
            print(f"{key}: {val:.3f}")
        else:
            print(f"{key}:", val)
