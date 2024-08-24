# python scripts/train_tms_sae.py                                              \
#   --tms_filepath="artefacts/toy_model-n_feat=1024-d_model=32-spars=0.99.pt"   \
#   --d_model=32                                                               \
#   --n_features=1024                                                          \
#   --feature_proba=0.01                                                       \
#   --k=4                                                                      \
#   --dead_steps_threshold=512                                                 \
#   --auxk=16                                                                  \
#   --batch_size=16_384                                                        \
#   --steps=30_000                                                             \
#   --lr=1e-4                                                                  \
#   --eval_freq=100                                                            \
#   --log_freq=100                                                             \
#   --model_save_path="artefacts/sae-n_feat=1024-d_model=32-spars=0.99.pt"     \
#   --trace_save_path="artefacts/sae-n_feat=1024-d_model=32-spars=0.99-trace"

from argparse import ArgumentParser

from sparse_autoencoder.tms.train_sae import train_tms_sae


def main(
    tms_filepath: str,
    d_model: int,
    n_features: int,
    feature_proba: float,
    k: int,
    dead_steps_threshold: int,
    auxk: int,
    batch_size: int,
    steps: int,
    lr: float,
    eval_freq: int,
    log_freq: int,
    model_save_path: str | None = None,
    trace_save_path: str | None = None,
) -> None:
    _spars = 1 - feature_proba
    _ = train_tms_sae(
        toy_model_filepath=tms_filepath,
        d_model=d_model,
        n_features=n_features,
        k=k,
        dead_steps_threshold=dead_steps_threshold,
        mse_scale=1,
        auxk=auxk,
        auxk_coeff=1 / 32,
        batch_size=batch_size,
        steps=steps,
        feature_proba=feature_proba,
        lr=lr,
        lr_scale=None,
        clip_grad=None,
        model_save_path=model_save_path,
        wandb_entity="xyznlp",
        run_id=f"tms_sae-n_feat={n_features}-d_model={d_model}-spars={_spars}",
        eval_freq=eval_freq,
        log_freq=log_freq,
        trace_save_path=trace_save_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a sparse autoencoder to learn activations of a TMS model."
    )
    parser.add_argument("--tms_filepath", type=str, help="TMS model filepath.")
    parser.add_argument("--d_model", type=int, help="Activation dimension.", default=32)
    parser.add_argument("--n_features", type=int, help="Num features.", default=1024)
    parser.add_argument(
        "--feature_proba",
        type=float,
        help="Feature probability (1 - sparsity).",
        default=0.01,
    )
    parser.add_argument("--k", type=int, help="`k` for TopK features.", default=4)
    parser.add_argument(
        "--dead_steps_threshold",
        type=int,
        help="Number of inactive steps to consider feature as dead.",
        default=512,
    )
    parser.add_argument("--auxk", type=int, help="`auxk` for AuxK loss.", default=16)
    parser.add_argument(
        "--batch_size", type=int, help="Training batch size.", default=16_384
    )
    parser.add_argument("--steps", type=int, help="Num training steps.", default=30_000)
    parser.add_argument("--lr", type=float, help="Learning rate.", default=1e-4)
    parser.add_argument(
        "--eval_freq", type=int, help="Evaluation frequency.", default=100
    )
    parser.add_argument("--log_freq", type=int, help="Logging frequency.", default=100)
    parser.add_argument(
        "--trace_save_path",
        type=str,
        help="Folder to save the trace of executed Triton kernels.",
        default=None,
    )
    args = parser.parse_args()

    main(
        tms_filepath=args.tms_filepath,
        d_model=args.d_model,
        n_features=args.n_features,
        feature_proba=args.feature_proba,
        k=args.k,
        dead_steps_threshold=args.dead_steps_threshold,
        auxk=args.auxk,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        model_save_path=args.model_save_path,
        trace_save_path=args.trace_save_path,
    )
