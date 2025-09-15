#!/bin/bash


NUM_GPUS=2
echo "Starting training on GPUs $(seq -s, 0 $((NUM_GPUS - 1)))"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) python /home/allanz/Ctrl-G/distillation/lvd_hmm.py \
    --sequences_file /home/allanz/Ctrl-G/sequences.pt --embeddings_file /home/allanz/Ctrl-G/hidden_states.pt \
    --hidden_states 1536 --vocab_size 151646 --eos_token_id 151645 \
    --kmeans_iterations 101 --pseudocount 0.001 \
    --output_file /home/allanz/Ctrl-G/checkpoints/checkpoint-0
