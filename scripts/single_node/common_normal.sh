#!/bin/bash
# Single node script for common_normal training

############ CONFIGURATION #############
# MODEL_NAME options: sd3, flux
MODEL_NAME="${MODEL_NAME:-flux}"

# REWARD options: pickscore, geneval, general_ocr
REWARD="${REWARD:-pickscore}"

# Number of GPUs (adjust based on your machine)
NUM_GPUS="${NUM_GPUS:-8}"

# Main process port
MASTER_PORT="${MASTER_PORT:-29501}"

# KL coefficient
BETA="${BETA:-0.0001}"

########################################

datatime=$(date +"%Y_%m_%d_%H_%M_%S")
version="${MODEL_NAME}_${REWARD}_normal_${datatime}_window5_kl_${BETA}"

echo "=========================================="
echo "Running single node training with:"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  REWARD: ${REWARD}"
echo "  NUM_GPUS: ${NUM_GPUS}"
echo "  BETA: ${BETA}"
echo "  VERSION: ${version}"
echo "=========================================="

# Launch training
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=${NUM_GPUS} \
    --main_process_port ${MASTER_PORT} \
    scripts/my_train_${MODEL_NAME}_fast.py \
    --config=config/my_grpo.py:${REWARD}_${MODEL_NAME}_fast \
    --config.run_name=$version \
    --config.train.beta=${BETA}
