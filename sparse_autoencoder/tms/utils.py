import matplotlib.pyplot as plt

import sparse_autoencoder.array_typing as at


def plot_features_in_2d(
    W: at.TmsWeights, title: str | None = None, save_path: str | None = None
):
    d_model, n_features = W.shape

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax.set_xlim(-1, 1)
    for feat_idx in range(n_features):
        x, y = W[:, feat_idx].tolist()
        (line,) = ax.plot([0, x], [0, y], color="black", lw=1.5)
        (marker,) = ax.plot(
            [x],
            [y],
            color="black",
            marker="o",
            markersize=8,
        )

    if title is not None:
        plt.title(title, fontsize=10)
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
