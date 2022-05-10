import pickle

import numpy as np
from matplotlib import pyplot as plt

from visualization.helper_functions import FIGURE_FOLDER


def plot_active_curve(results_file, file_name):
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, acquisition in enumerate(["Random", "Max Entropy"]):
        with open(results_file, "rb") as f:
            rsts = pickle.load(f)
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
    fig.savefig(f"{file_name}")


if __name__ == "__main__":
    plot_active_curve("results/active/mnist/nn.pkl", "active_nn.png")
    plot_active_curve(
        "results/active/mnist/multi_swag.pkl", "active_multi_swag.png"
    )
    plot_active_curve(
        "results/active/mnist/deep_ensemble.pkl", "active_deep_ensemble.png"
    )
