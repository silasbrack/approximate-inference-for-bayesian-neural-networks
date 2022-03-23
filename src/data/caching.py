import logging

from tqdm import tqdm

from src.data.mura import MuraDataset


def cache_dataset(dataset, verbose=True):
    logging.info("Caching dataset.")

    if not isinstance(dataset, MuraDataset):
        raise NotImplementedError

    dataset.set_use_cache(False)
    if verbose:
        for _ in tqdm(dataset):
            pass
    else:
        for _ in iter(dataset):
            pass
    dataset.set_use_cache(True)
