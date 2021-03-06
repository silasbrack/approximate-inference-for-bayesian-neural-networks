import torch
from torch.utils.data import DataLoader, Subset


def max_acquisition(acquisition_fn):
    def fn(all_indices, k, inference, train_set, batch_size, *args, **kwargs):
        with torch.no_grad():
            remaining_dataloader = DataLoader(
                Subset(train_set, all_indices),
                batch_size=batch_size,
                shuffle=False,
            )
            entropies = acquisition_fn(remaining_dataloader, inference)
            assert len(all_indices) == len(entropies)
            top_entropy_indices = torch.topk(entropies, k=k).indices
            sampled_indices = []
            for idx in top_entropy_indices.sort(descending=True).values:
                sampled_indices.append(all_indices.pop(idx))
            return torch.tensor(sampled_indices)

    return fn
