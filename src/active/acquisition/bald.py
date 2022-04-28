import torch


class Bald:
    def query(self, ):
        return max_acquisition(self.evaluate_entropy)()

    @staticmethod
    def evaluate_information_gain(dataloader, inference):
        entropies = []
        for x, y in iter(dataloader):
            predictive_probs = inference.predict(x, aggregate=False)
            probs = predictive_probs.sum(dim=0)
            log_probs = probs.log() + 1e-10
            predictive_entropy = -torch.mul(probs, log_probs).sum(dim=-1)
            probs = predictive_probs
            log_probs = probs.log() + 1e-10
            # Mean over posterior predictive samples (dim 0) and classes (dim
            # -1)
            expected_likelihood_entropy = -torch.mul(probs, log_probs) \
                .mean(dim=0).sum(dim=-1)
            entropy = predictive_entropy - expected_likelihood_entropy
            entropies.append(entropy)
        entropies = torch.cat(entropies)
        return entropies