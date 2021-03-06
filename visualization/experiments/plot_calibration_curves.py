from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.helper_functions import (
    FIGURE_FOLDER,
    TYPE_DICT,
    calibration_curves,
)
from visualization.load_results import load_results, save_results_to_table


def plot_calibration_curves(
    results: pd.DataFrame,
    eval_datasets: List[str],
    types: List[str],
    file_name: str,
):
    bins = 10

    fig, ax = plt.subplots(
        ncols=len(eval_datasets),
        nrows=2,
        sharex="all",
        sharey="all",
        figsize=(4 * len(eval_datasets), 6),
    )
    for i, dataset in enumerate(eval_datasets):
        ax_curve = ax[0, i] if len(eval_datasets) > 1 else ax[0]
        ax_hist = ax[1, i] if len(eval_datasets) > 1 else ax[1]
        for type in types:
            criteria = (
                f"`Evaluated on` == '{dataset.upper()}' and Type == '{type}'"
            )
            data = results.query(criteria)

            targets = data["Test targets"].values[0][:, None]
            probs = data["Test probabilities"].values[0]
            # confidences = np.max(probs, axis=1)

            ece, prob_true, prob_pred, n_in_bins = calibration_curves(
                targets, probs, bins=bins
            )
            error = 2 * np.sqrt(
                (prob_true * (1 - prob_true)) / n_in_bins[n_in_bins > 0]
            )
            results.loc[results.eval(criteria), "ECE"] = ece

            (line,) = ax_curve.plot(
                prob_pred,
                prob_true,
                label=f"{type}",
                c=TYPE_DICT[type]["color"],
                linestyle=TYPE_DICT[type]["style"],
            )
            ax_curve.errorbar(
                x=prob_pred,
                y=prob_true,
                yerr=error,
                c=line.get_color(),
                linestyle=TYPE_DICT[type]["style"],
            )

            ax_hist.bar(
                np.linspace(0, 1, bins + 1)[-bins:] - 1 / (2 * bins),
                n_in_bins / np.sum(n_in_bins),
                width=1 / bins,
                color=line.get_color(),
                alpha=0.5,
                label=f"{type}",
            )

        ls = np.linspace(0, 1)
        ax_curve.plot(ls, ls, "--", color="grey", alpha=0.3)
        ax_curve.set(
            xlim=(0, 1),
            ylim=(0, 1),
            title=dataset.upper(),
        )
        ax_hist.set(
            xlim=(0, 1),
            ylim=(0, 1),
        )
        if i == 0:
            ax_curve.set(ylabel="True probability")
            ax_hist.set(
                xlabel="Predicted probability",
                ylabel="Frequency",
            )
        if i == len(eval_datasets) - 1:
            ax_hist.legend(*ax_curve.get_legend_handles_labels())
            ax_hist.set(xlabel="Predicted probability")

    fig.tight_layout()

    fig.savefig(f"{FIGURE_FOLDER}/{file_name}")


if __name__ == "__main__":
    for train_set, eval_datasets in (
        ("mnist", ["mnist"]),  # "svhn"
        ("mura", ["mura"]),
    ):
        results = load_results(train_set, eval_datasets)
        to_print = results.drop(
            ["Type", "Test targets", "Test probabilities"], axis=1
        )
        print(to_print)
        save_results_to_table(to_print, f"{train_set}_full.tex")

        plot_calibration_curves(
            results,
            eval_datasets,
            types=["nn", "laplace", "meanfield", "radial", "lowrank"],
            file_name=f"{train_set.capitalize()}VI.png",
        )

        plot_calibration_curves(
            results,
            eval_datasets,
            types=["ensemble_5", "ensemble_10"],
            file_name=f"{train_set.capitalize()}Ensembles.png",
        )

        plot_calibration_curves(
            results,
            eval_datasets,
            types=["multiswag_5", "multiswag_10"],
            file_name=f"{train_set.capitalize()}MultiSwag.png",
        )
