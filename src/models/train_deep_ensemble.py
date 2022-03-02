import pickle
import time
from typing import List, Dict

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from omegaconf import DictConfig
from torch import softmax
from torch.nn import NLLLoss
from torch.utils.data import DataLoader

from src import data as d
from src.models import MNISTModel


@hydra.main(config_path="../conf", config_name="deep_ensemble")
def train_model(cfg: DictConfig):
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

    n_ensembles = cfg.params.num_ensembles
    models = []

    t0 = time.perf_counter()
    for i in range(n_ensembles):
        model = MNISTModel(cfg.params.lr)

        trainer = pl.Trainer(
            gpus=cfg.hardware.gpus,
            max_epochs=cfg.params.epochs,
        )

        trainer.fit(
            model,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
        )
        trainer.test(model, dataloaders=data.test_dataloader())
        models.append(model)
    elapsed = time.perf_counter() - t0

    svhn_data = d.SVHNData(cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers)
    svhn_data.setup()

    results = {
        "Trained on": cfg.params.data,
        "Wall clock time": elapsed,
        "Number of parameters": sum(p.numel() for p in models[0].parameters()),
        "eval_mnist": eval_model(models, data.test_dataloader()),
        "eval_svhn": eval_model(models, svhn_data.test_dataloader()),
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")
    with open(f"ensemble_{cfg.params.num_ensembles}.pkl", "wb") as f:
        pickle.dump(results, f)

    for i, model in enumerate(models):
        torch.save(
            model.state_dict(),
            f"{cfg.files.state_dict}_{i}",
        )


def eval_model(models: List, test_dataloader: DataLoader) -> Dict:
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
        ensemble_logits = torch.stack([model(x) for model in models])
        ensemble_probs = softmax(ensemble_logits, dim=-1).detach()
        probs = torch.mean(ensemble_probs, dim=0)
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
        nll_sum += nll_loss(logits, y)
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
