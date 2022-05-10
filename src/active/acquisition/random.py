import random

import torch


class RandomAcquisition:
    name = "Random"

    @staticmethod
    def query(all_indices, k, inference, train_set, *args, **kwargs):
        return sample_without_replacement(all_indices, k)


def sample_without_replacement(arr, n, *args, **kwargs):
    random.shuffle(arr)
    return torch.tensor([arr.pop() for _ in range(n)])
