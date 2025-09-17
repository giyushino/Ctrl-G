#!/bin/bash

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

# hard coded for my use case
VOCAB_SIZE=151643
HIDDEN_STATES=1536
EOS_TOKEN_ID=151645
HMM_MODEL_ID="qwen_${DATASET}_${HIDDEN_STATES}"
HMM_MODEL_PATH="/home/allanz/Ctrl-G/models/${HMM_MODEL_ID}"

SEQUENCES_FILE="/home/allanz/Ctrl-G/hmm_data/${DATASET}/${DATASET}.lvd"
EMBEDDINGS_FILE="/home/allanz/Ctrl-G/hmm_data/${DATASET}/${DATASET}.lvd.embeddings"
mkdir -p "$HMM_MODEL_PATH"

echo model_id: $HMM_MODEL_PATH
echo sequences file: $SEQUENCES_FILE
echo embeddings file: $EMBEDDINGS_FILE

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python /home/allanz/Ctrl-G/distillation/lvd_hmm.py \
    --sequences_file $SEQUENCES_FILE --embeddings $EMBEDDINGS_FILE \
    --hidden_states $HIDDEN_STATES --vocab_size $VOCAB_SIZE --eos_token_id $EOS_TOKEN_ID \
    --kmeans_iterations 100 --pseudocount 0.001 \
    --output_file $HMM_MODEL_PATH/checkpoint-0
