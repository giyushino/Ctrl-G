#!/bin/bash

LOG_FILE_PATH="/home/allanz/Ctrl-G/log.txt"
CUDA_CORES=0,1,2
BATCH_SIZE=256
SAVE_PER_STEP=10
DROPOUT=0.01

# EM training schedule:
# 1. train for 10 EM steps, each step using 1 chunk of data
# 2. train for 5 EM steps, each step using 2 chunks of data
# 3. train for 4 EM steps, each step using 5 chunks of data
# 4. train for 4 EM steps, each step using 10 chunks of data
# 5. train for 4 EM steps, each step using 20 chunks of data
# 6. train for 1 EM steps, each step using 40 chunks of data
EM_SCHEDULE="\"10,1;5,2;4,5;4,10;4,20;1,40\""

CUDA_VISIBLE_DEVICES=CUDA_CORES torchrun --standalone --nproc_per_node=gpu /home/allanz/Ctrl-G/distillation/train_hmm.py \
    --model_path /home/allanz/Ctrl-G/checkpoints/ --checkpoint 0 --save_per_step $SAVE_PER_STEP \
    --data_path {DATA_PATH} --dataset {DATASET} --total_chunks {TOTAL_CHUNKS} --batch_size {BATCH_SIZE} \
    --em_schedule $EM_SCHEDULE --dropout $DROPOUT --log_file $LOG_FILE_PATH
print(cmd)
