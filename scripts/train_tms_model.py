# Adapted from: https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx.
#
# python scripts/train_tms_model.py                                       \
#   --d_model 2                                                           \
#   --n_features 5                                                        \
#   --feature_proba 0.01                                                  \
#   --batch_size 1024                                                     \
#   --steps 10_000                                                        \
#   --lr 1e-3                                                             \
#   --model_save_path "artefacts/toy_model-n_feat=5-d_model=2-spars=0.99" \
#   --log_freq 100                                                        \
#   --fast
#
# python scripts/train_tms_model.py                                           \
#   --d_model 32                                                              \
#   --n_features 1024                                                         \
#   --feature_proba 0.01                                                      \
#   --batch_size 4096                                                         \
#   --steps 30_000                                                            \
#   --lr 1e-3                                                                 \
#   --model_save_path "artefacts/toy_model-n_feat=1024-d_model=32-spars=0.99" \
#   --log_freq 100                                                            \
#   --fast

from argparse import ArgumentParser

import torch
from einops import einsum

from sparse_autoencoder.tms.train_toy_model import generate_feature_batch, train_tms
from sparse_autoencoder.tms.utils import plot_features_in_2d


def main(
    d_model: int,
    n_features: int,
    feature_proba: float,
    batch_size: int,
    steps: int,
    lr: float,
    model_save_path: str | None = None,
    log_freq: int = 100,
    use_fast_model: bool = False,
) -> None:
    model = train_tms(
        d_model=d_model,
        n_features=n_features,
        feature_probs=feature_proba,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        lr_scale=None,
        log_freq=log_freq,
        device=torch.device("cuda"),
        model_save_path=f"{model_save_path}.pt",
        use_fast_model=use_fast_model,
    )
    if d_model == 2:
        samples = generate_feature_batch(
            batch_size=300, n_features=n_features, feature_probs=feature_proba
        )
        samples_hidden = einsum(samples, model.W_tr, "b f, f d -> b d")
        plot_features_in_2d(
            W=model.W_tr.transpose(0, 1),
            samples_hidden=samples_hidden,
            title=(
                f"{n_features} features in {d_model}D space\n"
                f"(spars={round(1 - feature_proba, 2)})"
            ),
            save_path=f"{model_save_path}.png",
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a toy model of superposition.")
    parser.add_argument(
        "--d_model",
        type=int,
        help="Model hidden dimension.",
        default=2,
    )
    parser.add_argument(
        "--n_features",
        type=int,
        help="Number of data features (set up to exp(d_model) features).",
        default=5,
    )
    parser.add_argument(
        "--feature_proba",
        type=float,
        help="Feature probability (1 - sparsity).",
        default=0.01,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size.",
        default=1024,
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of training steps.",
        default=10_000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate.",
        default=1e-3,
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="Logging frequency.",
        default=100,
    )
    parser.add_argument(
        "--model_save_path", type=str, help="Path to save the toy model."
    )
    parser.add_argument("--fast", action="store_true", help="Train using fast model.")
    args = parser.parse_args()

    # n_feat=5-d_model=2-spars=0.99: 10,000 steps, loss=0.000184, lr=6.12e-20.
    # n_feat=100-d_model=6-spars=0.96: 20,000 steps, loss=0.0103, lr=6.12e-20.
    # n_feat=1024-d_model=32-spars=0.99: 30,000 steps, loss=0.00253, lr=2.83e-19.
    main(
        d_model=args.d_model,
        n_features=args.n_features,
        feature_proba=args.feature_proba,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        log_freq=args.log_freq,
        model_save_path=args.model_save_path,
        use_fast_model=args.fast,
    )
