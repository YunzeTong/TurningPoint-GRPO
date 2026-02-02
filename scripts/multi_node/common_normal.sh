#!/bin/bash
# Multi-node script for common_normal training
#
# Usage: Run on each node with the node rank as argument
#   Node 0: bash scripts/multi_node/common_normal.sh 0
#   Node 1: bash scripts/multi_node/common_normal.sh 1
#   Node 2: bash scripts/multi_node/common_normal.sh 2
#   Node 3: bash scripts/multi_node/common_normal.sh 3

############ CONFIGURATION #############
# MODEL_NAME options: sd3, flux
MODEL_NAME="${MODEL_NAME:-flux}"

# REWARD options: pickscore, geneval, general_ocr
REWARD="${REWARD:-pickscore}"

# Number of machines
NUM_MACHINES="${NUM_MACHINES:-4}"

# Total number of processes (GPUs across all nodes)
NUM_PROCESSES="${NUM_PROCESSES:-32}"

# Master node address (change to your master node IP)
MASTER_ADDR="${MASTER_ADDR:-10.82.139.22}"

# Master port
MASTER_PORT="${MASTER_PORT:-19001}"

# KL coefficient
BETA="${BETA:-0.0001}"

########################################

# Common NCCL settings for multi-node
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3

# Get node rank from argument
RANK=${1:-0}

datatime=$(date +"%Y_%m_%d_%H_%M_%S")
version="${MODEL_NAME}_${REWARD}_normal_${datatime}_window5_kl_${BETA}"

echo "=========================================="
echo "Running multi-node training with:"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  REWARD: ${REWARD}"
echo "  NUM_MACHINES: ${NUM_MACHINES}"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "  RANK: ${RANK}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  BETA: ${BETA}"
echo "  VERSION: ${version}"
echo "=========================================="

# Launch training
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines ${NUM_MACHINES} \
    --num_processes ${NUM_PROCESSES} \
    --machine_rank ${RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/my_train_${MODEL_NAME}_fast.py \
    --config=config/my_grpo.py:${REWARD}_${MODEL_NAME}_fast \
    --config.run_name=$version \
    --config.train.beta=${BETA}
