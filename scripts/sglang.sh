#!/bin/bash

# test
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path qwen/qwen2.5-1.5b-instruct --host 0.0.0.0 --port 3000 --enable-return-hidden-states
