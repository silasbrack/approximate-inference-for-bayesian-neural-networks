from math import ceil

import hydra
import torch
import tyxe
from torch.utils.data import DataLoader, random_split

from src.evaluate import evaluate, print_dict


def split_dataloader(loader, num_loaders):
    dataset = loader.dataset
    batch_size = loader.batch_size

    subset_size = ceil(len(dataset) / num_loaders)
    subset_sizes = [subset_size] * (num_loaders - 1)
    subset_sizes.append(len(dataset) - sum(subset_sizes))

    subsets = random_split(dataset, subset_sizes)
    return [DataLoader(subset, batch_size, shuffle=True) for subset in subsets]


@hydra.main(config_path="../conf", config_name="config")
def run(cfg):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()

    inference = hydra.utils.instantiate(cfg.inference)

    num_tasks = 10
    train_loaders = split_dataloader(data.train_dataloader(), num_tasks)
    for train_loader in train_loaders:
        inference.fit(
            train_loader,
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

        eval_result = evaluate(
            inference, data.test_dataloader(), data.name, data.n_classes
        )
        print_dict(eval_result)
    inference.save(cfg.training.model_path)


if __name__ == "__main__":
    run()
