#!/bin/bash
set -e

# pip install lm_eval[wandb]
# pip install lm_eval[vllm]
# pip install langdetect
# pip -qq install -e ..[wandb]
export WANDB_API_KEY="ac6e358c9e02e44fdccee1c0c68e4a7ea095bafb"
wandb login

# Define variables
MODEL_NAME="mistralai/Mistral-7B-v0.3"  # Replace with your desired model
TASKS="leaderboard"                # Replace with your desired tasks
TP_SIZE=8                             # Number of GPUs for tensor parallelism
DP_SIZE=4                             # Number of model replicas
DTYPE="float32"                          # Data type (e.g., auto, float16)
GPU_UTIL=0.9                          # GPU memory utilization
BATCH_SIZE="auto:5"
MAX_LEN=4096                     

# Construct model arguments
MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},data_parallel_size=${DP_SIZE},max_model_len=${MAX_LEN}"

# Execute lm_eval
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_NAME}" \
        --wandb_args project=KD-tis-dpo,name=base_teacher,job_type=eval \
        --log_samples

# https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/about
