import hydra
from omegaconf import DictConfig
from pyro.infer import Predictive
from torchmetrics import Accuracy

from src import data as d
from src.models.train_swag import train_swag


# TODO: Make MultiSWAG work on GPU
@hydra.main(config_path="../conf", config_name="multiswag")
def run(cfg: DictConfig):
    num_ensembles = cfg.num_ensembles

    swag_models = []
    posterior_predictives = []
    for _ in range(num_ensembles):
        swag_model = train_swag(cfg)
        swag_models.append(swag_model)
        posterior_predictive = Predictive(swag_model, num_samples=128)
        posterior_predictives.append(posterior_predictive)

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    accuracy_calculator = Accuracy()
    for image, target in data.test_dataloader():
        logits = None
        for posterior_predictive in posterior_predictives:
            prediction = posterior_predictive(image)
            swag_logits = prediction["logits"]  # Do we already take the mean?
            if logits is None:
                logits = swag_logits
            else:
                logits += swag_logits
        logits /= num_ensembles
        logits = logits.mean(dim=0).squeeze(0)
        print(logits.shape)
        accuracy_calculator(logits, target)
    accuracy = accuracy_calculator.compute()
    accuracy_calculator.reset()
    print(f"Test accuracy for MultiSWAG = {100 * accuracy:.2f}")


if __name__ == "__main__":
    run()
