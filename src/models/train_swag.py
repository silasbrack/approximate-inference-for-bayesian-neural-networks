import torch

from src.models.train_swa import train_model


def run():
    model = train_model()
    trainer = model.trainer
    swa_callback = trainer.callbacks[0]
    # TODO: Get this to work
    weights = swa_callback.weights

    weight_loc = torch.mean(weights, dim=-1)
    weight_scale = torch.std(weights, dim=-1)

    # posterior = dist.Normal(weight_loc, weight_scale)
    return weight_loc, weight_scale


if __name__ == "__main__":
    run()
