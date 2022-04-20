import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(
    context="talk",
    style="ticks",
    rc={
        "font.sans-serif": ["Arial", "Fira Sans Condensed"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": "small",
    },
)


RESULTS_FOLDER = "./results/"
FIGURE_FOLDER = "./reports/figures"
TABLE_FOLDER = "./reports/tables"

TYPE_DICT = {
    "radial": {"color": "#4E79A7", "style": "solid"},
    "meanfield": {"color": "#F28E2B", "style": "dotted"},
    "laplace": {"color": "#59A14F", "style": "dashed"},
    "lowrank": {"color": "#B6992D", "style": "dashdot"},
    "ml": {"color": "#499894", "style": "solid"},
    "map": {"color": "#E15759", "style": "solid"},
    "ensemble_5": {"color": "#BAB0AC", "style": "solid"},
    "ensemble_10": {"color": "#79706E", "style": "solid"},
    "multiswag_5": {"color": "#D37295", "style": "solid"},
    "multiswag_10": {"color": "#FABFD2", "style": "solid"},
}


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
    """Remove the top and right axis."""
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def calibration_curves(targets, probs, bins=10, fill_nans=False):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)[:, None]

    real_probs = np.zeros((bins,))
    pred_probs = np.zeros((bins,))
    bin_sizes = np.zeros((bins,))

    _, lims = np.histogram(confidences, range=(0.0, 1.0), bins=bins)
    for i in range(bins):
        lower, upper = lims[i], lims[i + 1]
        mask = (lower <= confidences) & (confidences < upper)

        targets_in_range = targets[mask]
        preds_in_range = preds[mask]
        probs_in_range = confidences[mask]
        n_in_range = preds_in_range.shape[0]

        range_acc = (
            np.sum(targets_in_range == preds_in_range) / n_in_range
            if n_in_range > 0
            else 0
        )
        range_prob = (
            np.sum(probs_in_range) / n_in_range if n_in_range > 0 else 0
        )

        real_probs[i] = range_acc
        pred_probs[i] = range_prob
        bin_sizes[i] = n_in_range

    bin_weights = bin_sizes / np.sum(bin_sizes)
    ece = np.sum(np.abs(real_probs - pred_probs) * bin_weights)

    if fill_nans:
        return ece, real_probs, pred_probs, bin_sizes
    return ece, real_probs[bin_sizes > 0], pred_probs[bin_sizes > 0], bin_sizes
