import pickle

import pandas as pd
from pandas.io.formats.style import Styler

from visualization.helper_functions import (RESULTS_FOLDER, TABLE_FOLDER,
                                            calibration_curves)


def load_results(train_set: str = "mnist", eval_datasets=None, types=None):
    if eval_datasets is None:
        eval_datasets = ["mnist", "svhn"]
    if types is None:
        types = [
            # "ensemble_5",
            # "ensemble_10",
            "multiswag_5",
            "multiswag_10",
            "radial",
            "meanfield",
            "lowrank",
            # "laplace",
            # "nn",
        ]

    results_folder = f"{RESULTS_FOLDER}/{train_set}"

    results = pd.DataFrame()
    for dataset in eval_datasets:

        df = []
        for type in types:
            with open(f"{results_folder}/{type}/results.pkl", "rb") as f:
                data = pickle.load(f)
                df.append(data)
        df = pd.DataFrame.from_dict(df)
        df.insert(0, "Type", types)
        results = pd.concat([results, df])

    results = results.rename(
        columns={
            "Average confidence": "Avg. Conf.",
            "Average confidence when wrong": "Avg. Conf. -",
            "Average confidence when right": "Avg. Conf. +",
        }
    )

    bins = 10
    for i, dataset in enumerate(eval_datasets):
        for type in types:
            criteria = f"`Evaluated on` == '{dataset.upper()}' and Type == '{type}'"
            print(criteria)
            data = results.query(criteria)
            
            targets = data["Test targets"].values[0][:, None]
            probs = data["Test probabilities"].values[0]
            # confidences = np.max(probs, axis=1)

            ece, prob_true, prob_pred, n_in_bins = calibration_curves(
                targets, probs, bins=bins
            )
            # error = 2 * np.sqrt(
            #     (prob_true * (1 - prob_true)) / n_in_bins[n_in_bins > 0])
            results.loc[results.eval(criteria), "ECE"] = ece

    return results


def save_results_to_table(results: pd.DataFrame, file_name: str):
    s: Styler = (
        results.drop(["Test targets", "Test probabilities"], axis=1)
        .replace(
            {
                "mnist": "MNIST",
                "svhn": "SVHN",
                "mura": "MURA",
                "ensemble_5": "Ensemble@5",
                "ensemble_10": "Ensemble@10",
                "multiswag_5": "MultiSWAG@5",
                "multiswag_10": "MultiSWAG@10",
                "radial": "Radial",
                "meanfield": "Mean-field",
                "lowrank": "Low-rank",
                "laplace": "Laplace",
                "ml": "ML",
                "map": "MAP",
            }
        )
        .style
    )
    s.format(precision=3)
    s.hide(axis="index")
    latex_table = s.to_latex()

    with open(f"{TABLE_FOLDER}/{file_name}", "w") as f:
        f.write(latex_table)
