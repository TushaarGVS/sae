# Layers (first, between 3/4 and 5/6 deep; see https://arxiv.org/pdf/2406.04093):
# - recurrent layers: 0, 30
# - attention layers: 2, 29
#
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
# scripts/get_recurrentgemma_activations.py \
#   --hf_dataset_id "codeparrot/github-code" \
#   --text_colname "code" \
#   --per_device_batch_size 4 \
#   --model_id "9b" \
#   --layer_nums 0 2 30 31 \
#   --save_dir "/home/tg352/test"

import json
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Tuple
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


def _init_dist(debug: bool = False) -> Tuple[int | None, torch.device | None]:
    if int(os.environ.get("RANK", -1)) != -1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        return local_rank, torch.device("cuda", local_rank)
    return None, None


def _dist_cleanup(debug: bool = False):
    dist.destroy_process_group()


def _load_vocab(weights_dir: Path, debug: bool = False) -> spm.SentencePieceProcessor:
    vocab_path = weights_dir / "tokenizer.model"
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))
    if debug:
        print(f"[DEBUG]\t {len(vocab)=}")
    return vocab


def _load_model(
    weights_dir: Path,
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
) -> recurrentgemma.Griffin:
    ckpt_path = weights_dir / f"{model_id}.pt"
    params = torch.load(str(ckpt_path))
    params = {key: value.to(device=device) for key, value in params.items()}
    preset = (
        recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
        if "2b" in model_id
        else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1
    )
    model_config = recurrentgemma.GriffinConfig.from_torch_params(params, preset=preset)
    model = recurrentgemma.Griffin(model_config, device=device, dtype=dtype)
    model.load_state_dict(params)
    if debug:
        print(f"[DEBUG]\t recurrentgemma-{model_id}\n{model_config}\n")
    return model


def _unpad_batch(
    batch_tensor: at.ExpandedActivations,
    pad_lengths: at.Tokens,
    unpad: bool = True,
    debug: bool = False,
) -> List[torch.Tensor] | torch.Tensor:
    if unpad:
        return [tensor[length:] for tensor, length in zip(batch_tensor, pad_lengths)]
    return batch_tensor


@torch.no_grad()
@torch.inference_mode()
def get_activations(
    input_strings: Sequence[str],
    model_id: str = "2b",
    layer_nums: List[int] | None = None,
    unpad: bool = False,
    device: torch.device = torch.device("cuda"),
    debug: bool = False,
) -> Dict[str, List[torch.Tensor] | at.Tokens | Dict[str, List[torch.Tensor] | None]]:
    weights_dir = Path(
        kagglehub.model_download(f"google/recurrentgemma/pyTorch/{model_id}")
    )
    vocab = _load_vocab(weights_dir=weights_dir, debug=debug)
    model = _load_model(
        weights_dir=weights_dir,
        model_id=model_id,
        dtype=torch.bfloat16,
        device=device,
        debug=debug,
    )
    sampler = recurrentgemma.Sampler(
        model=model,
        vocab=vocab,
        is_it_model="it" in model_id,
        greedy_sampling=True,
    )

    if layer_nums is None:
        print(f"[WARNING]\t {layer_nums=}; none specified, using all layers ...")
        layer_nums = list(range(model.config.num_layers))

    all_input_ids = [sampler.tokenize(x, debug=False) for x in input_strings]
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

    _, cache = model(
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
    model_id: str = "2b",
    layer_nums: List[int] | None = None,
    save_dir: str | None = None,
    debug: bool = False,
) -> None:
    assert save_dir is not None, f"{save_dir=} (not specified)"

    local_rank, device = _init_dist()
    world_size = torch.distributed.get_world_size()
    if local_rank == 0:
        args = locals()

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "artefacts"), exist_ok=True)

        with open(os.path.join(save_dir, f"config/args.json"), "w") as fp:
            json.dump(args, fp, default=repr, indent=4)

    hf_dataset = datasets.load_dataset(hf_dataset_id, streaming=True, split="train")
    hf_dataset = split_dataset_by_node(
        hf_dataset, rank=local_rank, world_size=world_size
    )
    dataloader = DataLoader(hf_dataset, batch_size=per_device_batch_size)

    all_activations_dict = dict() if local_rank == 0 else None
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if debug:
            print(f"[DEBUG]\t [{local_rank=}/{world_size=}] {batch_idx=}")

        activations_dict = get_activations(
            input_strings=batch[text_colname],
            model_id=model_id,
            layer_nums=layer_nums,
            unpad=True,
            device=device,
            debug=(debug and local_rank == 0),
        )
        with open(
            os.path.join(save_dir, f"artefacts/pid.{local_rank}-batch.{batch_idx}.pkl"),
            "wb",
        ) as fp:
            pickle.dump(activations_dict, fp)

        if debug:
            if local_rank == 0:
                print(f"[WARNING]\t running with {debug=}; stopping after first batch")
            break

    print(f"activations saved to {save_dir}")
    _dist_cleanup()


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
        "--model_id",
        choices=["2b", "2b-it", "9b", "9b-it"],
        default="2b-it",
        help="RecurrentGemma model identifier.",
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

    main(
        hf_dataset_id=args.hf_dataset_id,
        text_colname=args.text_colname,
        per_device_batch_size=args.per_device_batch_size,
        model_id=args.model_id,
        layer_nums=args.layer_nums,
        save_dir=args.save_dir,
        debug=args.debug,
    )
