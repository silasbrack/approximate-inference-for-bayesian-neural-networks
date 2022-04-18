import pickle

import numpy as np
from matplotlib import pyplot as plt

from visualization.helper_functions import FIGURE_FOLDER


def plot_ensemble_accuracies(results_file, file_name):
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    for acquisition, rsts in results.items():
        accuracy = np.array(rsts["accuracy"])
        samples = np.cumsum(rsts["samples"])
        # queries = np.arange(len(samples))

        ax.plot(samples, accuracy, label=acquisition)
    ax.set(
        # xlabel="Queries",
        xlabel="Queried images",
        ylabel="Test accuracy",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURE_FOLDER}/{file_name}")


if __name__ == "__main__":
    plot_ensemble_accuracies("results/active/mnist/results.pkl", "active.pdf")
