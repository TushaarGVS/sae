# Layers (first, between 3/4 and 5/6 deep; see https://arxiv.org/pdf/2406.04093):
# 9b, 9b-it:
# - recurrent layers: 0, 30
# - attention layers: 2, 29
# 2b, 2b-it:
# - recurrent layers: 0, 21
# - attention layers: 2, 20
#
# Datasets:
# - JeanKaddour/minipile (text_colname: text)
# - codeparrot/github-code (text_colname: code)
# - monology/pile-uncopyrighted (text_colname: text)
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
# scripts/get_recurrentgemma_activations.py         \
#   --hf_dataset_id "JeanKaddour/minipile"          \
#   --text_colname "text"                           \
#   --per_device_batch_size 4                       \
#   --max_len 8192                                  \
#   --variant "2b"                                  \
#   --layer_nums 0 2 20 21                          \
#   --save_dir "/share/rush/tg352/sae/minipile/2b"

import json
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
from typing import Sequence

import datasets
import kagglehub
import sentencepiece as spm
import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from tqdm import tqdm

import recurrentgemma
import recurrentgemma.array_typing as at

# Distributed inference setup.
assert int(os.environ.get("RANK", -1)) != -1, "Distributed setup failed."
dist.init_process_group("nccl")

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = torch.distributed.get_world_size()


def debug_on_master(msg: str):
    if LOCAL_RANK == 0 and DEBUG:
        print(msg)


def print_on_master(msg: str):
    if LOCAL_RANK == 0:
        print(msg)


def load_vocab(weights_dir: Path) -> spm.SentencePieceProcessor:
    vocab_path = weights_dir / "tokenizer.model"
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))
    debug_on_master(f"Loaded vocab: {len(vocab)=}")
    return vocab


def load_model(
    weights_dir: Path,
    variant: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
) -> recurrentgemma.Griffin:
    ckpt_path = weights_dir / f"{variant}.pt"
    params = torch.load(str(ckpt_path))
    params = {key: value.to(device=device) for key, value in params.items()}
    preset = (
        recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
        if "2b" in variant
        else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1
    )
    model_config = recurrentgemma.GriffinConfig.from_torch_params(params, preset=preset)
    model = recurrentgemma.Griffin(model_config, device=device, dtype=dtype)
    model.load_state_dict(params)
    debug_on_master(f"Loaded model: recurrentgemma-{variant}\n{model_config}\n")
    return model


@torch.no_grad()
@torch.inference_mode()
def get_sampler(
    variant: at.Variant,
    device: torch.device = torch.device("cuda"),
) -> recurrentgemma.Sampler:
    weights_dir = Path(
        kagglehub.model_download(f"google/recurrentgemma/pyTorch/{variant}")
    )
    vocab = load_vocab(weights_dir=weights_dir)
    model = load_model(
        weights_dir=weights_dir,
        variant=variant,
        dtype=torch.bfloat16,
        device=device,
    )
    sampler = recurrentgemma.Sampler(
        model=model,
        vocab=vocab,
        is_it_model=("it" in variant),
        greedy_sampling=True,
    )
    return sampler


def _unpad_batch(
    batch_tensor: at.ExpandedActivations,
    pad_lengths: at.Tokens,
    unpad: bool = True,
) -> List[torch.Tensor] | torch.Tensor:
    if unpad:
        return [tensor[length:] for tensor, length in zip(batch_tensor, pad_lengths)]
    return batch_tensor


@torch.no_grad()
@torch.inference_mode()
def get_activations(
    input_strings: Sequence[str],
    sampler: recurrentgemma.Sampler,
    max_len: int = 8192,
    layer_nums: List[int] | None = None,
    unpad: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, List[torch.Tensor] | at.Tokens | Dict[str, List[torch.Tensor] | None]]:
    all_input_ids = [sampler.tokenize(x)[-max_len:] for x in input_strings]
    input_lengths = torch.tensor(
        [len(input_ids) for input_ids in all_input_ids],
        device=device,
        dtype=torch.int32,
    )
    padded_tokens: at.Tokens = sampler._get_padded_tokens(all_input_ids)
    batch_size, prompt_length = padded_tokens.shape
    pad_lengths = prompt_length - input_lengths

    positions = torch.arange(prompt_length, device=device, dtype=torch.int32)
    positions = torch.repeat_interleave(positions[None], batch_size, dim=0)
    positions = positions - prompt_length + input_lengths[:, None]
    positions = torch.clip(positions, min=-1)  # -1: pad tokens

    debug_on_master(f"{padded_tokens.shape=}\n{positions.shape=}")
    _, cache = sampler.model(
        tokens=padded_tokens,
        segment_pos=positions,
        cache=None,
        return_logits=False,
        return_cache=True,
        xai_intermediate_layer_nums=layer_nums,
    )

    activations_dict = dict(
        input_ids=(all_input_ids if unpad else padded_tokens),
        pad_lengths=(None if unpad else pad_lengths),
    )
    for layer_num in layer_nums:
        block_id = f"blocks.{layer_num}"
        block_cache = cache[block_id]
        activations_dict[block_id] = dict(
            rg_lru_states=(
                _unpad_batch(block_cache.all_rg_lru_states, pad_lengths, unpad=unpad)
                if hasattr(block_cache, "all_rg_lru_states")
                else None
            ),
            mlp_activations=_unpad_batch(
                block_cache.mlp_activations, pad_lengths, unpad=unpad
            ),
        )
    return activations_dict


@torch.no_grad()
@torch.inference_mode()
def main(
    hf_dataset_id: str,
    text_colname: str = "text",
    per_device_batch_size: int = 4,
    max_len: int = 8192,
    variant: at.Variant = "2b",
    layer_nums: List[int] | None = None,
    save_dir: str | None = None,
) -> None:
    assert save_dir is not None, f"{save_dir=} (not specified)"
    fn_args = locals()

    device = torch.device("cuda", LOCAL_RANK)
    sampler = get_sampler(variant=variant, device=device)
    if layer_nums is None:
        layer_nums = list(range(sampler.model.config.num_layers))

    if LOCAL_RANK == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "artefacts"), exist_ok=True)

        with open(os.path.join(save_dir, f"config/args.json"), "w") as fp:
            json.dump(fn_args, fp, default=repr, indent=4)

    hf_dataset = datasets.load_dataset(hf_dataset_id, streaming=True, split="train")
    hf_dataset = split_dataset_by_node(
        hf_dataset, rank=LOCAL_RANK, world_size=WORLD_SIZE
    )
    dataloader = DataLoader(hf_dataset, batch_size=per_device_batch_size)

    all_activations_dict = dict() if LOCAL_RANK == 0 else None
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        debug_on_master(
            f"{'-' * 10}\n"
            f"[{LOCAL_RANK=}/{WORLD_SIZE=}] {batch_idx=}\n"
            f"{batch[text_colname][0][:100]} ...\n"
        )
        activations_dict = get_activations(
            input_strings=batch[text_colname],
            sampler=sampler,
            max_len=max_len,
            layer_nums=layer_nums,
            unpad=True,
            device=device,
        )
        with open(
            os.path.join(save_dir, f"artefacts/pid.{LOCAL_RANK}-batch.{batch_idx}.pkl"),
            "wb",
        ) as fp:
            pickle.dump(activations_dict, fp)

    print_on_master(f"Activations saved to {save_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract activations from RecurrentGemma.")
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        help="The huggingface dataset ID (e.g., allenai/c4).",
    )
    parser.add_argument(
        "--text_colname", type=str, help="Text column name", default="text"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        help="Per-device batch size for streaming.",
        default=1,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        help="Max sequence length to run recurrentgemma on (maily for CUDA memory management).",
        default=8192,
    )
    parser.add_argument(
        "--variant",
        choices=["2b", "2b-it", "9b", "9b-it"],
        default="2b",
        help="RecurrentGemma model variant.",
    )
    parser.add_argument(
        "--layer_nums",
        nargs="+",
        type=int,
        help="Layer numbers to extract activations from.",
    )
    parser.add_argument("--save_dir", type=str, help="Directory to save activations.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    # Set a global `DEBUG` variable.
    DEBUG = args.debug

    main(
        hf_dataset_id=args.hf_dataset_id,
        text_colname=args.text_colname,
        per_device_batch_size=args.per_device_batch_size,
        max_len=args.max_len,
        variant=args.variant,
        layer_nums=args.layer_nums,
        save_dir=args.save_dir,
    )
