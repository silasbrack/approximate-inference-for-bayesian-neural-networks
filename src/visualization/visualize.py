import matplotlib.pyplot as plt
import numpy as np

import src.visualization.helper_functions as hf


def main():
    n = 256
    x = np.linspace(-2 * np.pi, 2 * np.pi, n, endpoint=True)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y_sin, color=hf.COLOR_SIN, linestyle="-", label="sin")
    ax.plot(x, y_cos, color=hf.COLOR_COS, linestyle="--", label="cos")
    hf.improve_legend(ax)
    hf.hide_right_top_axis(ax)

    ax.set(
        title="Sine and cosine plots",
        xlabel="Angle [rad]",
        ylabel="Value",
    )

    fig.tight_layout()
    fig.savefig(hf.FIGURE_PATH.joinpath("sin_cosine.pdf"))
    fig.show()


if __name__ == "__main__":
    main()
