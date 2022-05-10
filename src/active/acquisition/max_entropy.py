import torch

from src.active.acquisition.util import max_acquisition


class MaxEntropy:
    name = "Max entropy"

    def query(self, all_indices, k, inference, train_set, *args, **kwargs):
        query_fn = max_acquisition(self.evaluate_entropy)
        return query_fn(all_indices, k, inference, train_set, *args, **kwargs)

    @staticmethod
    def evaluate_entropy(dataloader, inference):
        entropies = []
        for x, y in dataloader:
            probs = inference.predict(x)
            log_probs = probs.log() + 1e-10
            entropy = -torch.mul(probs, log_probs).sum(dim=-1)
            entropies.append(entropy)
        entropies = torch.cat(entropies)
        return entropies
