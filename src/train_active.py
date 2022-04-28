import pickle
import random

import hydra
import pyro
import torch
import tyxe
from torch.utils.data import DataLoader, Subset
from src.active.acquisition.random import sample_without_replacement

from src.evaluate import evaluate
from src.inference import VariationalInference
from visualization.experiments.plot_active_cuve import plot_active_curve


@hydra.main(config_path="../conf", config_name="config")
def run(cfg):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()

    initial_pool = cfg.training.initial_pool
    query_size = cfg.training.query_size

    acquisition_function = hydra.utils.instantiate(cfg.acquisition)

    train_loader = data.train_dataloader()
    train_set = train_loader.dataset.dataset
    all_indices = train_loader.dataset.indices.copy()

    sampled_indices = sample_without_replacement(all_indices, initial_pool)
    n_sampled = []
    accuracies = []
    for _ in range(cfg.training.active_queries):
        currently_training_loader = DataLoader(
            Subset(train_set, sampled_indices),
            batch_size=cfg.data.batch_size,
        )

        inference = hydra.utils.instantiate(cfg.inference)
        inference.fit(
            currently_training_loader,
            data.val_dataloader(),
            epochs=cfg.training.epochs,
            lr=cfg.training.lr,
        )
        if inference is VariationalInference:
            inference.bnn.update_prior(
                tyxe.priors.DictPrior(
                    inference.bnn.net_guide.get_detached_distributions(
                        tyxe.util.pyro_sample_sites(inference.bnn.net)
                    )
                )
            )
        new_indices = acquisition_function.query(
            all_indices,
            query_size,
            inference,
            train_set,
            batch_size=8192,
        )
        n_sampled.append(len(new_indices))
        sampled_indices = torch.cat((sampled_indices, new_indices))
        acc = evaluate(
            inference, data.test_dataloader(), data.name, data.n_classes
        )["Accuracy"]
        accuracies.append(acc)

    with open("results.pkl", "wb") as f:
        pickle.dump({"accuracy": accuracies, "samples": n_sampled}, f)

    plot_active_curve("results.pkl", "active.png")


if __name__ == "__main__":
    run()
