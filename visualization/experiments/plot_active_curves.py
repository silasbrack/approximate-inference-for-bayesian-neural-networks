import pickle

import numpy as np
from matplotlib import pyplot as plt

from visualization.helper_functions import FIGURE_FOLDER


def plot_active_curves():
    result_path = "results/active/mura"

    types = [
        ("nn", ["random", "max_entropy"]),
        ("deep_ensemble", ["random", "max_entropy"]),
        ("multiswag", ["random", "max_entropy", "bald"]),
        ("radial", ["random", "max_entropy", "bald"]),
    ]

    fig, axs = plt.subplots(figsize=(18, 4), ncols=len(types), sharey=True)

    for (inference, acquisitions), ax in zip(types, axs):
        for acquisition in acquisitions:
            with open(f"{result_path}/{inference}_{acquisition}.pkl", "rb") as f:
                results = pickle.load(f)

            accuracy = np.array(results["accuracy"])
            samples = np.cumsum(results["samples"])

            ax.plot(
                samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
            )
            ax.set(title=inference)
            ax.legend()
            ax.set(xlabel="Queried images")
    
    axs[0].set(ylabel="Test accuracy")
    fig.tight_layout()
    fig.savefig(f"{FIGURE_FOLDER}/active_comparison.pdf")


# def plot_active_curves():
#     # result_path = "results/active/mnist/convnet"
#     result_path = "results/active/mura"
#     # nns = []
#     # for i in range(4):
#     #     with open(f"{result_path}/nn_{i}.pkl", "rb") as f:
#     #         nns.append(pickle.load(f))
#     with open(f"{result_path}/nn.pkl", "rb") as f:
#         nns = pickle.load(f)

#     with open(f"{result_path}/radial.pkl", "rb") as f:
#         radial = pickle.load(f)

#     with open(f"{result_path}/multiswag.pkl", "rb") as f:
#         multiswag = pickle.load(f)

#     with open(f"{result_path}/deep_ensemble.pkl", "rb") as f:
#         ensemble = pickle.load(f)

#     fig, ax = plt.subplots(figsize=(18, 4), ncols=4, sharey=True)

#     results = nns
#     accuracy = np.array(results["accuracy"])
#     samples = np.cumsum(results["samples"])
    # ax[0].plot(
    #     samples, accuracy, label=f"Max entropy ({100*accuracy[-1]:.1f}%)"
    # )
    # ax[0].set(title="Neural network")
#     # for acquisition in ["Random", "Max entropy"]:
#     # accuracy = np.mean(
#     #     [np.array(res[acquisition]["accuracy"]) for res in results], axis=0
#     # )
#     # samples = np.cumsum(results[0][acquisition]["samples"])
#     # accuracy = np.array(rsts["accuracy"])
#     # samples = np.cumsum(rsts["samples"])
#     # ax[0].plot(
#     #     samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
#     # )
#     # ax[0].set(title="Neural network")

#     results = radial
#     for acquisition, rsts in results.items():
#         accuracy = np.array(rsts["accuracy"])
#         samples = np.cumsum(rsts["samples"])
#         ax[1].plot(
#             samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
#         )
#         ax[1].set(title="Radial variational approximation")

#     results = ensemble
#     for acquisition, rsts in results.items():
#         accuracy = np.array(rsts["accuracy"])
#         samples = np.cumsum(rsts["samples"])
#         ax[2].plot(
#             samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
#         )
#         ax[2].set(title="Deep ensemble")

#     results = multiswag
#     for acquisition, rsts in results.items():
#         accuracy = np.array(rsts["accuracy"])
#         samples = np.cumsum(rsts["samples"])
#         ax[3].plot(
#             samples, accuracy, label=f"{acquisition} ({100*accuracy[-1]:.1f}%)"
#         )
#         ax[3].set(title="MultiSWAG")

#     ax[0].set(
#         xlabel="Queried images",
#         ylabel="Test accuracy",
#     )
#     ax[1].set(xlabel="Queried images")
#     ax[2].set(xlabel="Queried images")
#     ax[3].set(xlabel="Queried images")
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     ax[3].legend()
#     fig.tight_layout()
#     fig.savefig(f"{FIGURE_FOLDER}/active_comparison.png")


if __name__ == "__main__":
    plot_active_curves()
