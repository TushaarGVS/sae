#!/bin/bash

# torchrun --standalone --nproc_per_node=5            \
# scripts/get_recurrentgemma_activations.py           \
#   --hf_dataset_id "JeanKaddour/minipile"            \
#   --text_colname "text"                             \
#   --per_device_batch_size 1                         \
#   --max_len 8192                                    \
#   --variant "9b"                                    \
#   --layer_nums 0 2 29 30                            \
#   --save_dir "/share/rush/tg352/sae/minipile/9b"

sbatch -p rush                                    \
  --nodelist=rush-compute-02                      \
  --gres=gpu:a6000:1                              \
  --ntasks=1                                      \
  --cpus-per-task=12                              \
  --mem=128G                                      \
  --time 30-00:00:00                              \
  --wrap "python sparse_autoencoder/train.py"
