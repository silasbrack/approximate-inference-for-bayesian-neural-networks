import pickle

import numpy as np
from matplotlib import pyplot as plt

from visualization.helper_functions import FIGURE_FOLDER


def plot_active_curves():
    nns = []
    for i in range(4):
        with open(f"results/active/mnist/convnet/nn_{i}.pkl", "rb") as f:
            nns.append(pickle.load(f))

    with open(f"results/active/mnist/convnet/radial.pkl", "rb") as f:
        radial = pickle.load(f)

    with open(f"results/active/mnist/convnet/multiswag.pkl", "rb") as f:
        multiswag = pickle.load(f)

    with open(f"results/active/mnist/convnet/deep_ensemble.pkl", "rb") as f:
        ensemble = pickle.load(f)

    fig, ax = plt.subplots(figsize=(18, 4), ncols=4, sharey=True)

    results = nns
    for acquisition in ["Random", "Max entropy"]:
        accuracy = np.mean(
            [np.array(res[acquisition]["accuracy"]) for res in results], axis=0
        )
        samples = np.cumsum(results[0][acquisition]["samples"])
        ax[0].plot(
            samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
        )
        ax[0].set(title="Neural network")

    results = radial
    for acquisition, rsts in results.items():
        accuracy = np.array(rsts["accuracy"])
        samples = np.cumsum(rsts["samples"])
        ax[1].plot(
            samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
        )
        ax[1].set(title="Radial variational approximation")

    results = ensemble
    for acquisition, rsts in results.items():
        accuracy = np.array(rsts["accuracy"])
        samples = np.cumsum(rsts["samples"])
        ax[2].plot(
            samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
        )
        ax[2].set(title="Deep ensemble")

    results = multiswag
    for acquisition, rsts in results.items():
        accuracy = np.array(rsts["accuracy"])
        samples = np.cumsum(rsts["samples"])
        ax[3].plot(
            samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
        )
        ax[3].set(title="MultiSWAG")

    ax[0].set(
        xlabel="Queried images",
        ylabel="Test accuracy",
    )
    ax[1].set(xlabel="Queried images")
    ax[2].set(xlabel="Queried images")
    ax[3].set(xlabel="Queried images")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURE_FOLDER}/active_comparison.pdf")


if __name__ == "__main__":
    plot_active_curves()
