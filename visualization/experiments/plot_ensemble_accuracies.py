import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from torchmetrics import Accuracy

from src.evaluate import evaluate
from src.predict import load_model
from visualization.helper_functions import FIGURE_FOLDER


def plot_ensemble_accuracies(run_path):
    inference, data = load_model(f"{run_path}")
    num_ensembles = len(inference.ensembles)
    accuracy = Accuracy()

    probabilities = []
    accuracies = []
    for i in range(num_ensembles):
        ensemble = inference.ensembles[i]
        result = evaluate(
            ensemble, data.test_dataloader(), data.name, data.n_classes
        )
        targets = torch.tensor(result["Test targets"])
        probs = torch.tensor(result["Test probabilities"])
        probabilities.append(probs)
        accuracy(probs, targets)
        accuracies.append(accuracy.compute().item())
        accuracy.reset()
    probabilities = torch.stack(probabilities)

    ensemble_accuracies = []
    ensemble_range = list(range(1, num_ensembles + 1))
    for i in ensemble_range:
        ensemble_probs = probabilities[:i].mean(dim=0)
        accuracy(ensemble_probs, targets)
        ensemble_accuracies.append(accuracy.compute().item())
        accuracy.reset()

    accuracies = np.array(accuracies)
    ensemble_accuracies = np.array(ensemble_accuracies)
    ensemble_range = np.array(ensemble_range)
    ensemble_accuracies *= 100
    accuracies *= 100
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hlines(
        y=accuracies,
        xmin=1 - 0.3,
        xmax=1 + 0.3,
        color="black",
        label="Individual NNs",
    )
    ax.plot(
        ensemble_range,
        ensemble_accuracies,
        label="Ensembles",
        color="black",
        marker="o",
        markersize=5,
    )
    ax.scatter(ensemble_range, ensemble_accuracies, color="black", s=20)
    # ax.hlines(y=accuracies, xmin=1, xmax=num_ensembles,
    #           color="grey", linestyle="--", label="Individual NNs")
    # ax.scatter([1]*len(accuracies), accuracies, color="black", s=20)
    # ax.legend()
    ax.set(
        xlabel="Number of ensembles",
        ylabel="Accuracy",
        xlim=[0.8, num_ensembles + 1],
    )
    ax.yaxis.set_major_formatter(PercentFormatter())
    fig.tight_layout()
    fig.savefig(f"{FIGURE_FOLDER}/ensemble_comparison.pdf")
    fig.show()


if __name__ == "__main__":
    # plot_ensemble_accuracies("models/deep_ensemble/mnist_20.pt")
    # plot_ensemble_accuracies("multirun/2022-04-06/18-43-53/0")
    # plot_ensemble_accuracies("outputs/2022-04-19/08-59-11")
    plot_ensemble_accuracies("results/mnist/ensemble_20")
