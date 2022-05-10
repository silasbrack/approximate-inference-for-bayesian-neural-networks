import logging

import torch
from tqdm import tqdm


def cache_dataset(dataset, verbose=False):
    logging.info("Caching dataset.")

    dataset.set_use_cache(False)
    if verbose:
        for _ in tqdm(dataset):
            pass
    else:
        for _ in iter(dataset):
            pass
    dataset.set_use_cache(True)


def save_cached(dataset, path):
    torch.save(
        {
            "cached_data": dataset.cached_data,
            "cached_indices": dataset.cached_indices,
            "n_cached": dataset.n_cached,
        },
        path,
    )


def load_cached(dataset, path):
    cache = torch.load(path)
    dataset.cached_data = cache["cached_data"]
    dataset.cached_indices = cache["cached_indices"]
    dataset.n_cached = cache["n_cached"]
    dataset.set_use_cache(True)
