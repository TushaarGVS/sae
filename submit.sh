#!/bin/bash

# sbatch -p rush               \
#   --nodelist=rush-compute-03 \
#   --gres=gpu:a6000:5         \
#   --ntasks=1                 \
#   --cpus-per-task=24         \
#   --mem=512G                 \
#   -t 30-00:00:00             \
#   submit.sh

torchrun --standalone --nproc_per_node=5            \
scripts/get_recurrentgemma_activations.py           \
  --hf_dataset_id "JeanKaddour/minipile"            \
  --text_colname "text"                             \
  --per_device_batch_size 1                         \
  --max_len 8192                                    \
  --variant "9b"                                    \
  --layer_nums 0 2 29 30                            \
  --save_dir "/share/rush/tg352/sae/minipile/9b"
