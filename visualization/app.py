import gzip
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import kagglehub
import sentencepiece as spm
from flask import Flask, request, render_template

import recurrentgemma.array_typing as at

VARIANT: at.Variant = "2b"
ACTIVATIONS_DIR: str = f"/share/rush/tg352/sae/minipile/{VARIANT}/artefacts"

app = Flask(__name__)


@lru_cache(1)
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


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/visualize_raw_activations", methods=["POST"])
def visualize_raw_activations():
    filename: str = request.form.get("filename")
    variant: at.Variant = request.form.get("variant")

    vocab = load_vocab(variant=variant)
    activations_dict = pickle.load(
        gzip.open(os.path.join(ACTIVATIONS_DIR, filename), "rb")
    )
    all_input_ids = activations_dict.pop("input_ids")
    if len(all_input_ids) > 1:
        print("more than one inst in the file; visualizing only the first inst")
    activations_dict["tokens"] = [
        [vocab.IdToPiece(input_id) for input_id in input_ids.tolist()]
        for input_ids in all_input_ids
    ]
    return json.dumps(activations_dict)


if __name__ == "__main__":
    app.run(8000)
