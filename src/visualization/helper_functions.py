from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Fira Sans Condensed"]

FIGURE_PATH = Path("reports/figures/")

CMAP = plt.get_cmap("tab10")
COLOR_SIN = CMAP(0)
COLOR_COS = CMAP(1)


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
