import pickle
import random

import hydra
import pyro
import torch
import tyxe
from torch.utils.data import DataLoader, Subset

from src.evaluate import evaluate
from src.inference import VariationalInference
from visualization.experiments.plot_active_cuve import plot_active_curve


def sample_without_replacement(arr, n, *args, **kwargs):
    random.shuffle(arr)
    return [arr.pop() for _ in range(n)]


def evaluate_entropy(dataloader, inference):
    entropies = []
    for x, y in dataloader:
        probs = inference.predict(x)
        log_probs = probs.log() + 1e-10
        entropy = -torch.mul(probs, log_probs).sum(dim=-1)
        entropies.append(entropy)
    entropies = torch.cat(entropies)
    return entropies


def evaluate_information_gain(dataloader, inference):
    entropies = []
    for x, y in iter(dataloader):
        predictive_probs = inference.predict(x, aggregate=False)
        probs = predictive_probs.sum(dim=0)
        log_probs = probs.log() + 1e-10
        predictive_entropy = -torch.mul(probs, log_probs).sum(dim=-1)
        probs = predictive_probs
        log_probs = probs.log() + 1e-10
        # Mean over posterior predictive samples (dim 0) and classes (dim -1)
        expected_likelihood_entropy = (
            -torch.mul(probs, log_probs).mean(dim=0).sum(dim=-1)
        )
        entropy = predictive_entropy - expected_likelihood_entropy
        entropies.append(entropy)
    entropies = torch.cat(entropies)
    return entropies


def max_acquisition(acquisition_fn):
    def fn(all_indices, k, inference, train_set, *args, **kwargs):
        with torch.no_grad():
            remaining_dataloader = DataLoader(
                Subset(train_set, all_indices), batch_size=8192, shuffle=False
            )
            entropies = acquisition_fn(remaining_dataloader, inference)
            assert len(all_indices) == len(entropies)
            top_entropy_indices = torch.topk(entropies, k=k).indices
            sampled_indices = []
            for idx in top_entropy_indices.sort(descending=True).values:
                sampled_indices.append(all_indices.pop(idx))
            return sampled_indices

    return fn


@hydra.main(config_path="../conf", config_name="config")
def run(cfg):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()

    recursive = (
        False
        if cfg.inference["_target_"]
        in ["src.inference.DeepEnsemble", "src.inference.MultiSwag"]
        else True
    )

    initial_pool = cfg.training.initial_pool
    query_size = cfg.training.query_size
    results = {}
    acquisition_funcs = [
        (sample_without_replacement, "Random"),
        (max_acquisition(evaluate_entropy), "Max entropy"),
    ]
    if cfg.inference["_target_"] not in [
        "src.inference.DeepEnsemble",
        "src.inference.Swa",
        "src.inference.NeuralNetwork",
    ]:
        acquisition_funcs.append(
            (max_acquisition(evaluate_information_gain), "BALD")
        )

    for acquisition_function, name in acquisition_funcs:
        pyro.get_param_store().clear()
        # inference = hydra.utils.instantiate(cfg.inference,
        #                                     _recursive_=recursive)

        train_loader = data.train_dataloader()
        train_set = train_loader.dataset.dataset
        all_indices = train_loader.dataset.indices.copy()

        sampled_indices = torch.tensor(
            sample_without_replacement(all_indices, initial_pool)
        )
        n_sampled = []
        accuracies = []
        for _ in range(cfg.training.active_queries):
            currently_training_loader = DataLoader(
                Subset(train_set, sampled_indices),
                batch_size=cfg.data.batch_size,
            )

            inference = hydra.utils.instantiate(
                cfg.inference, _recursive_=recursive
            )
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
            new_indices = torch.tensor(
                acquisition_function(
                    all_indices, query_size, inference, train_set
                )
            )
            n_sampled.append(len(new_indices))
            sampled_indices = torch.cat((sampled_indices, new_indices))
            # n_sampled.append(len(sampled_indices))
            # assert 55000 == (len(all_indices) + sum(n_sampled))
            # sampled_indices = torch.tensor(
            #     acquisition_function(all_indices,
            #                          query_size,
            #                          inference,
            #                          train_set))

            acc = evaluate(
                inference, data.test_dataloader(), data.name, data.n_classes
            )["Accuracy"]
            accuracies.append(acc)
        results[name] = {"accuracy": accuracies, "samples": n_sampled}

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    # inference.save(cfg.training.model_path)
    plot_active_curve("results.pkl", "active.png")


if __name__ == "__main__":
    run()
