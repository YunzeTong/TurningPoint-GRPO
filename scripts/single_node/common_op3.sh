#!/bin/bash
# Single node script for common_op3 training

############ CONFIGURATION #############
# MODEL_NAME options: sd3, flux
MODEL_NAME="${MODEL_NAME:-sd3}"

# REWARD options: pickscore, geneval, general_ocr
REWARD="${REWARD:-pickscore}"

# Number of GPUs (adjust based on your machine)
NUM_GPUS="${NUM_GPUS:-8}"

# Main process port
MASTER_PORT="${MASTER_PORT:-29501}"

# KL coefficient
BETA="${BETA:-0.0001}"

########################################

OPERATION="op3"
datatime=$(date +"%Y_%m_%d_%H_%M_%S")
COMMENT="delta_g_as_bonus_noise_level_0.7_eta_1.0_firststep_only_ct_interstep_only_ct_balance_use_sde_minus_rt_replace_main_kl_0.0001"
version="${MODEL_NAME}_${REWARD}_${OPERATION}_${COMMENT}_${datatime}"

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
