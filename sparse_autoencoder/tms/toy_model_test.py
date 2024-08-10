import torch

from sparse_autoencoder.tms.toy_model import ToyModel, FastToyModel
from sparse_autoencoder.tms.train import generate_feature_batch, mse_loss

toy_model = ToyModel(d_model=2, n_features=5)
toy_model.cuda()

fast_toy_model = FastToyModel(d_model=2, n_features=5)
fast_toy_model.cuda()


def test_fast_toy_model(n_steps: int = 10):
    fast_toy_model.W_tr = toy_model.W_tr
    fast_toy_model.bias = toy_model.bias

    for step in range(n_steps):
        x = generate_feature_batch(batch_size=10, n_features=5, feature_probs=0.8)

        recons_toy_model = toy_model(x)
        recons_fast_toy_model = fast_toy_model(x)
        fwd_diff = torch.max(torch.abs(recons_toy_model - recons_fast_toy_model)).item()
        print(f"[{step}/{n_steps - 1}]\n\t+ {fwd_diff=}")

        loss_toy_model = mse_loss(x, recons_toy_model)
        loss_fast_toy_model = mse_loss(x, recons_fast_toy_model)
        loss_diff = torch.abs(loss_toy_model - loss_fast_toy_model).item()
        print(f"\t+ {loss_diff=}")

        loss_toy_model.backward()
        loss_fast_toy_model.backward()
        dw_toy_model = toy_model.W_tr.grad
        dbias_toy_model = toy_model.bias.grad
        dw_fast_toy_model = fast_toy_model.W_tr.grad
        dbias_fast_toy_model = fast_toy_model.bias.grad
        dw_diff = torch.max(torch.abs(dw_toy_model - dw_fast_toy_model)).item()
        dbias_diff = torch.max(torch.abs(dbias_toy_model - dbias_fast_toy_model)).item()
        print(f"\t+ {dw_diff=}\n\t+ {dbias_diff=}\n--")


if __name__ == "__main__":
    test_fast_toy_model()
