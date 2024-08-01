# uvicorn visualization.server:app --host 0.0.0.0 --port=8000
# python visualization/server.py

import os
import pickle
from functools import lru_cache
from pathlib import Path

import kagglehub
import sentencepiece as spm
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import recurrentgemma.array_typing as at

VARIANT: at.Variant = "2b"
ACTIVATIONS_DIR: str = f"/share/rush/tg352/sae/minipile/{VARIANT}/artefacts"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


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


@app.get("/")
async def read_root(variant: at.Variant = VARIANT):
    vocab = load_vocab(variant=variant)
    return {"message": f"RecurrentGemma-{VARIANT} vocab {len(vocab)=} cached."}


@app.post("/visualize_activations")
def visualize_activations(filename: str, variant: at.Variant = VARIANT):
    vocab = load_vocab(variant=variant)
    activations_dict = pickle.load(open(os.path.join(ACTIVATIONS_DIR, filename), "rb"))
    all_input_ids = activations_dict.pop("input_ids")
    activations_dict["tokens"] = [
        [vocab.IdToPiece(input_id) for input_id in input_ids.tolist()]
        for input_ids in all_input_ids
    ]
    return activations_dict


if __name__ == "__main__":
    # Load `localhost:8000/docs`.
    uvicorn.run(app, host="0.0.0.0", port=8000)
