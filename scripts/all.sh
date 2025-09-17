#!/bin/bash

# Define variables
CUDA_CORES='2'
BATCH_SIZE=10
BASE_MODEL_PATH='qwen/qwen2.5-1.5b-instruct'
INPUT_FILE='NONE'
DATASET='EOS'
DATA_PATH="/home/allanz/Ctrl-G/hmm_data/${DATASET}"
LVD_SIZE=100000
CHUNK_SIZE=100000
DEV_SIZE=20000
TOTAL_CHUNKS=100
SEQUENCE_LEN=32

# Create data directory
mkdir -p "$DATA_PATH"

# Sample LVD (Latent Variable Distillation) examples
echo "Starting LVD data sampling..."
CUDA_VISIBLE_DEVICES=$CUDA_CORES python /home/allanz/Ctrl-G/distillation/sglang_sample_data.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --tokenizer_name_or_path "$BASE_MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --chunk_size "$LVD_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$SEQUENCE_LEN" \
    --save_embeddings \
    --output_file "${DATA_PATH}/${DATASET}.lvd" \
    --port 3000

echo "LVD data sampling complete."



# training data sampling
echo "Starting training data sampling..."
CUDA_VISIBLE_DEVICES=$CUDA_CORES python -m torch.distributed.run --nproc_per_node=1 \
    /home/allanz/Ctrl-G/distillation/sglang_sample_data.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --tokenizer_name_or_path "$BASE_MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --chunk_size "$CHUNK_SIZE" \
    --total_chunks "$TOTAL_CHUNKS" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$SEQUENCE_LEN" \
    --output_file "${DATA_PATH}/${DATASET}.train"

echo "Training data sampling complete."

# Sample development examples
echo "Starting dev data sampling..."
CUDA_VISIBLE_DEVICES=$CUDA_CORES python -m torch.distributed.run --nproc_per_node=1 \
    /home/allanz/Ctrl-G/distillation/sglang_sample_data.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --tokenizer_name_or_path "$BASE_MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --chunk_size "$DEV_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$SEQUENCE_LEN" \
    --output_file "${DATA_PATH}/${DATASET}.dev"

echo "Dev data sampling complete."
