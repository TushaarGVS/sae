import gzip
import json
import os
import pickle
from functools import cache
from pathlib import Path

import kagglehub
import sentencepiece as spm
import torch
from flask import Flask, request, render_template

import recurrentgemma.array_typing as at

DATASET: str = "minipile"
ACTIVATIONS_DIR: str = f"/share/rush/tg352/sae/{DATASET}"

app = Flask(__name__)


@cache
def load_vocab(
    variant: at.Variant,
) -> spm.SentencePieceProcessor:
    weights_dir = Path(
        kagglehub.model_download(f"google/recurrentgemma/pyTorch/{variant}")
    )
    vocab_path = weights_dir / "tokenizer.model"
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))
    return vocab


def normalize(
    activations: torch.Tensor, length_dim: int = 0, min_max: bool = False
) -> torch.Tensor:
    """Normalize within [-1, 1]."""
    if min_max:
        min_vals = torch.min(activations, dim=length_dim).values
        max_vals = torch.max(activations, dim=length_dim).values
        return 2 * ((activations - min_vals) / (max_vals - min_vals + 1e-5)) - 1
    else:
        mean_vals = torch.mean(activations, dim=length_dim)
        std_vals = torch.std(activations, dim=length_dim)
        normalized_activations = (activations - mean_vals) / (std_vals + 1e-5)
        return 2 * (1 / (1 + torch.exp(-normalized_activations))) - 1


@app.route("/")
def home():
    return render_template("home.html")


@cache
@app.route("/visualize_raw_activations", methods=["POST"])
def visualize_raw_activations():
    filename: str = request.form.get("filename")
    variant: at.Variant = request.form.get("variant")

    vocab = load_vocab(variant=variant)
    activations_dict = pickle.load(
        gzip.open(
            os.path.join(ACTIVATIONS_DIR, f"{variant}/artefacts/{filename}"), "rb"
        )
    )
    all_input_ids = activations_dict.pop("input_ids")

    return_dict = {
        "tokens": [
            [vocab.IdToPiece(input_id) for input_id in input_ids.tolist()]
            for input_ids in all_input_ids
        ]
    }
    for key in activations_dict.keys():
        if key.startswith("blocks"):
            return_dict[key] = {}
            for activation_type in activations_dict[key].keys():
                activation_tensors_list = activations_dict[key][activation_type]
                if activation_tensors_list is not None:
                    return_dict[key][activation_type] = [
                        normalize(activations_tensor).tolist()
                        for activations_tensor in activation_tensors_list
                    ]
    return json.dumps(return_dict)


if __name__ == "__main__":
    app.run(port=8000)
