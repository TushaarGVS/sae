import torch

from sparse_autoencoder.modules.sparse_matmul import coo_sparse_dense_matmul
from sparse_autoencoder.tms.sae import TmsAutoencoder, TmsFastAutoencoder
from sparse_autoencoder.tms.toy_model import FastToyModel
from sparse_autoencoder.tms.train_toy_model import generate_feature_batch, mse_loss

# Load the trained toy model (spars=0.99) to generate activations for a feature batch.
# Model stats: steps=10_000, loss=0.000184, lr=6.12e-20.
feature_proba = 0.01
toy_model = FastToyModel(d_model=2, n_features=5)
toy_model.W_tr.data = torch.tensor(
    [
        [-0.5670, 1.0149],
        [-0.1375, -1.1547],
        [1.0559, -0.4863],
        [0.7899, 0.8549],
        [-1.1394, -0.2262],
    ]
)
toy_model.bias.data = torch.tensor([-0.2757, -0.2749, -0.2732, -0.2761, -0.2733])
toy_model.eval()
toy_model.cuda()

tms_sae = TmsAutoencoder(n_features=5, d_model=2, k=1, dead_steps_threshold=5, auxk=3)
tms_sae.cuda()

fast_tms_sae = TmsFastAutoencoder(
    n_features=5, d_model=2, k=1, dead_steps_threshold=5, auxk=3
)
fast_tms_sae.cuda()


def _max_diff(tensor1: torch.Tensor, tensor2: torch.Tensor):
    assert tensor1 is not None and tensor2 is not None
    return torch.max(torch.abs(tensor1 - tensor2)).item()


def test_fast_tms_sae(n_steps: int = 15):
    fast_tms_sae.pre_bias.data = tms_sae.pre_bias.data
    fast_tms_sae.latent_bias.data = tms_sae.latent_bias.data
    fast_tms_sae.W_enc.data = tms_sae.W_enc.data
    fast_tms_sae.W_dec.data = tms_sae.W_dec.data
    fast_tms_sae.stats_last_nonzero.data = tms_sae.stats_last_nonzero.data

    for step in range(n_steps):
        with torch.inference_mode():
            x = generate_feature_batch(
                batch_size=1000, n_features=5, feature_probs=feature_proba
            )
            activations = coo_sparse_dense_matmul(x, toy_model.W_tr)

        recons, auxk_dict = tms_sae(activations)
        fast_recons, fast_auxk_dict = fast_tms_sae(activations)
        recons_diff = _max_diff(recons, fast_recons)
        auxk_idxs_diff = _max_diff(auxk_dict["auxk_idxs"], fast_auxk_dict["auxk_idxs"])
        auxk_vals_diff = _max_diff(auxk_dict["auxk_vals"], fast_auxk_dict["auxk_vals"])
        stats_last_nonzero_diff = _max_diff(
            tms_sae.stats_last_nonzero, fast_tms_sae.stats_last_nonzero
        )
        print(
            f"[{step}/{n_steps - 1}]\n\t"
            f"+ {recons_diff=}\n\t"
            f"+ {auxk_idxs_diff=}\n\t"
            f"+ {auxk_vals_diff=}\n\t"
            f"+ {stats_last_nonzero_diff=}"
        )

        # Important: `mse_loss(output, target)` and not `(target, output)`.
        loss_sae = mse_loss(recons, activations)
        loss_fast_sae = mse_loss(fast_recons, activations)
        loss_diff = torch.abs(loss_sae - loss_fast_sae).item()
        print(f"\t+ {loss_diff=}")

        loss_sae.backward(retain_graph=True)
        loss_fast_sae.backward(retain_graph=True)
        dpre_bias_diff = _max_diff(tms_sae.pre_bias.grad, fast_tms_sae.pre_bias.grad)
        dlatent_bias_diff = _max_diff(
            tms_sae.latent_bias.grad, fast_tms_sae.latent_bias.grad
        )
        dW_enc_diff = _max_diff(tms_sae.W_enc.grad, fast_tms_sae.W_enc.grad)
        dW_dec_diff = _max_diff(tms_sae.W_dec.grad, fast_tms_sae.W_dec.grad)
        print(
            f"\t+ {dpre_bias_diff=}\n\t"
            f"+ {dlatent_bias_diff=}\n\t"
            f"+ {dW_enc_diff=}\n\t"
            f"+ {dW_dec_diff=}\n"
            f"--"
        )

        tms_sae._unit_norm_decoder()
        fast_tms_sae._unit_norm_decoder()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")  # only for testing

    test_fast_tms_sae()
