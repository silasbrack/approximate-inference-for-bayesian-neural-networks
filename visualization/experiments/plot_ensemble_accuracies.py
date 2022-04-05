from matplotlib import pyplot as plt

from src.evaluate import evaluate
from src.predict import load_model
from visualization.helper_functions import FIGURE_FOLDER


def plot_ensemble_accuracies():
    results_path = "/home/silas/Documents/university/approximate-inference" \
                   "-for-bayesian-neural-networks/multirun/2022-03-27/11-56" \
                   "-09/1/"
    inference, data = load_model(results_path)

    ensemble_accuracy = evaluate(
        inference, data.test_dataloader(), data.name, data.n_classes
    )["Accuracy"]
    num_ensembles = inference.num_ensembles
    ensemble_accuracies = [evaluate(ensemble,
                                    data.test_dataloader(),
                                    data.name,
                                    data.n_classes)["Accuracy"]
                           for ensemble in inference.ensembles]

    fig, ax = plt.subplots()
    ax.hlines(y=[ensemble_accuracy], xmin=num_ensembles - 0.4,
              xmax=num_ensembles + 0.4, label="Ensemble")
    ax.hlines(y=ensemble_accuracies, xmin=num_ensembles - 0.4,
              xmax=num_ensembles + 0.4, color="grey", linestyle="--",
              label="Individual NN")
    ax.legend()  # loc="center right"
    ax.set(
        xlabel="Number of ensembles",
        ylabel="Accuracy",
    )
    fig.tight_layout()
    fig.savefig(f"{FIGURE_FOLDER}/ensemble_comparison.pdf")


if __name__ == "__main__":
    plot_ensemble_accuracies()
