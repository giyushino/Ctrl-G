#!/bin/bash

# lvd, we need embeddings
# /home/allanz/Ctrl-G/distillation/fast_sample.py
#python /home/allanz/Ctrl-G/distillation/fast_sample.py \
#    --chunk_size 100000 --save_embeddings 1 --total_chunks 1 --temperature 1.1 \
#    --batch_size 200 \
#    --save_path "/home/allanz/Ctrl-G/hmm_data/EOS/EOS.lvd"
#
# training data
#
python /home/allanz/Ctrl-G/distillation/fast_sample.py \
    --chunk_size 100000 --save_embeddings 0 --total_chunks 100 --temperature 1.1 \
    --save_path "/home/allanz/Ctrl-G/hmm_data/EOS/EOS.train" --batch_size 200 \

# dev data
python /home/allanz/Ctrl-G/distillation/fast_sample.py \
    --chunk_size 20000 --save_embeddings 0 --total_chunks 100 --temperature 1.1 \
    --save_path "/home/allanz/Ctrl-G/hmm_data/EOS/EOS.dev"

