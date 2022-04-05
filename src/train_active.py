import random
from math import ceil

import hydra
import torch
import tyxe
from torch.utils.data import DataLoader, Subset, random_split

from src.evaluate import evaluate


def split_dataloader(loader, num_loaders):
    dataset = loader.dataset
    batch_size = loader.batch_size

    subset_size = ceil(len(dataset) / num_loaders)
    subset_sizes = [subset_size] * (num_loaders - 1)
    subset_sizes.append(len(dataset) - sum(subset_sizes))

    subsets = random_split(dataset, subset_sizes)
    return [DataLoader(subset, batch_size, shuffle=True) for subset in subsets]


def sample_without_replacement(arr, n, *args, **kwargs):
    random.shuffle(arr)
    return [arr.pop() for _ in range(n)]


def evaluate_entropy(dataloader, inference):
    entropies = []
    for x, y in iter(dataloader):
        probs = inference.predict(x)
        log_probs = probs.log()
        entropy = -torch.mul(probs, log_probs).sum(dim=-1)
        entropies.append(entropy)
    entropies = torch.cat(entropies)
    return entropies


def evaluate_information_gain(dataloader, inference):
    entropies = []
    for x, y in iter(dataloader):
        predictive_probs = inference.predict(x, aggregate=False)
        probs = predictive_probs.sum(dim=0)
        log_probs = probs.log()
        predictive_entropy = -torch.mul(probs, log_probs).sum(dim=-1)
        probs = predictive_probs + 1e-45
        log_probs = probs.log()
        # Sum over posterior predictive samples (dim 0) and classes (dim -1)
        expected_likelihood_entropy = -torch.mul(probs, log_probs)\
            .sum(dim=0).sum(dim=-1)
        entropy = predictive_entropy - expected_likelihood_entropy
        entropies.append(entropy)
    entropies = torch.cat(entropies)
    return entropies


def max_acquisition(acquisition_fn):
    def fn(all_indices, k, inference, train_set, *args, **kwargs):
        remaining_dataloader = DataLoader(Subset(train_set, all_indices),
                                          batch_size=8192, shuffle=False)
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

    initial_training_samples = 1000
    active_training_samples = 20

    results = {}
    for acquisition_function, name in [
        (sample_without_replacement, "Random"),
        (max_acquisition(evaluate_entropy), "Max entropy"),
        (max_acquisition(evaluate_information_gain), "BALD"),
    ]:
        inference = hydra.utils.instantiate(cfg.inference)

        train_loader = data.train_dataloader()
        train_set = train_loader.dataset.dataset
        all_indices = train_loader.dataset.indices

        sampled_indices = sample_without_replacement(all_indices,
                                                     initial_training_samples)
        n_sampled = []
        accuracies = []
        for _ in range(cfg.training.epochs):
            currently_training_loader = DataLoader(
                Subset(train_set, sampled_indices), batch_size=8192)

            inference.fit(
                currently_training_loader,
                data.val_dataloader(),
                1,
                cfg.training.lr,
            )
            inference.bnn.update_prior(
                tyxe.priors.DictPrior(
                    inference.bnn.net_guide.get_detached_distributions(
                        tyxe.util.pyro_sample_sites(inference.bnn.net)
                    )
                )
            )
            n_sampled.append(len(sampled_indices))

            sampled_indices = acquisition_function(all_indices,
                                                   active_training_samples,
                                                   inference,
                                                   train_set)

            acc = evaluate(inference,
                           data.test_dataloader(),
                           data.name,
                           data.n_classes)["Accuracy"]
            accuracies.append(acc)
            results[name] = {"accuracy": accuracies, "samples": n_sampled}

    print(results)
    # inference.save(cfg.training.model_path)


if __name__ == "__main__":
    run()
