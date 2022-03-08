# from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["font.sans-serif"] = ["Fira Sans Condensed"]

# FIGURE_PATH = Path("reports/figures/")

# CMAP = plt.get_cmap("tab10")
# COLOR_SIN = CMAP(0)
# COLOR_COS = CMAP(1)


def improve_legend(ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    for line in ax.lines:
        data_x, data_y = line.get_data()
        right_most_x = data_x[-1]
        right_most_y = data_y[-1]
        ax.annotate(
            line.get_label(),
            xy=(right_most_x, right_most_y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=line.get_color(),
            *args,
            **kwargs
        )
    ax.legend().set_visible(False)


def hide_right_top_axis(ax):
    """Remove the top and right axis"""
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

def calibration_curves(targets, probs, bins=10):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)[:, None]

    real_probs = []
    pred_probs = []
    n_in_bins = []

    _, lims = np.histogram(confidences, range=(0., 1.), bins=bins)
    for i in range(bins):
        lower, upper = lims[i], lims[i+1]
        mask = (lower <= confidences) & (confidences < upper)

        targets_in_range = targets[mask]
        preds_in_range = preds[mask]
        n_in_range = preds_in_range.shape[0]
        range_acc = np.sum(targets_in_range == preds_in_range) / n_in_range

        real_probs.append(range_acc)
        pred_probs.append((lower+upper)/2)
        n_in_bins.append(n_in_range)
    return real_probs, pred_probs, n_in_bins
