#!/bin/bash
# Multi-node script for common_op3 training
#
# Usage: Run on each node with the node rank as argument
#   Node 0: bash scripts/multi_node/common_op3.sh 0
#   Node 1: bash scripts/multi_node/common_op3.sh 1
#   Node 2: bash scripts/multi_node/common_op3.sh 2
#   Node 3: bash scripts/multi_node/common_op3.sh 3

############ CONFIGURATION #############
# MODEL_NAME options: sd3, flux
MODEL_NAME="${MODEL_NAME:-sd3}"

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

OPERATION="op3"
datatime=$(date +"%Y_%m_%d_%H_%M_%S")
COMMENT="delta_g_as_bonus_noise_level_0.7_eta_1.0_firststep_only_ct_interstep_only_ct_balance_use_sde_minus_rt_replace_main_kl_0.0001"
version="${MODEL_NAME}_${REWARD}_${OPERATION}_${COMMENT}_${datatime}"

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
    scripts/my_train_${MODEL_NAME}_fast_with_${OPERATION}.py \
    --config=config/my_grpo.py:${REWARD}_${MODEL_NAME}_fast_${OPERATION} \
    --config.run_name=$version \
    --config.apply_first_step_bonus=true \
    --config.use_delta_global=true \
    --config.take_delta_global_as_main=false \
    --config.sample.noise_level=0.7 \
    --config.eta=1.0 \
    --config.select_first_step_only_from_consistent_trajectory=true \
    --config.select_inter_step_only_from_consistent_trajectory=true \
    --config.use_balanced_bonus=true \
    --config.use_sde_minus_rt=true \
    --config.propagate_bonus_to_future_steps=false \
    --config.drop_original_reward_when_having_bonus=true \
    --config.indicator_wo_delta_global_constraint=false \
    --config.train.beta=${BETA}
