import logging

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
