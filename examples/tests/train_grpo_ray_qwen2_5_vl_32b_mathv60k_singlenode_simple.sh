#!/bin/bash

set -x

# Set environment variables and paths.
NODE_HOSTNAME=$(hostname)
WORKSPACE_DIR="$(pwd)"
DATASET_PATH="/data/datasets/VerMulti/mathv60k_message.jsonl"
PRETRAIN_MODEL_PATH="Qwen/Qwen2.5-VL-32B-Instruct"
SAVE_PATH="./checkpoints"
MODEL_NAME="qwen2.5-vl-32b-ins-mathvista-grpo"
WANDB_DIR="${WORKSPACE_DIR}"

# NCCL Commands
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1

# Suppress unhandled errors in Ray
export RAY_IGNORE_UNHANDLED_ERRORS=1

# Check for WandB API key and prepare WandB args
if [ -z "${WANDB_API_KEY}" ]; then
  echo "[INFO] WANDB_API_KEY not set. WandB logging will be disabled."
  WANDB_ARGS="trainer.logger=[\\\'console\\\']"
else
  echo "[INFO] WANDB_API_KEY found. WandB logging enabled."
  WANDB_ARGS="trainer.logger=[\\\'console\\\',\\\'wandb\\\'] trainer.project_name=\\\'openrlhf-m-training\\\' trainer.experiment_name=\\\'${MODEL_NAME}\\\'"
fi

# Get the IP address of eth1 interface - This might not be strictly necessary for verl's Ray integration,
# but keeping it for potential use or debugging.
ETH1_IP=$(ip addr show eth1 | grep -oP 'inet \K[\d.]+')
echo "Using eth1 IP address: ${ETH1_IP}"

# Submit the training job using verl's entry point and configuration overrides.
# verl typically uses a base YAML configuration file (e.g., in verl/trainer/config).
# The following command overrides parameters in that base config.
# You may need to specify a base config file using the `+trainer=your_config_name` syntax
# or by ensuring the default config (often ppo_trainer.yaml) is suitable.

echo "[HEAD NODE] Submitting verl training job via Ray job submit..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\\\"working_dir\\\": \\\"${WORKSPACE_DIR}/verl\\\"}" \
   -- python3 -m verl.trainer.main_ppo \
       # Mapping OpenRLHF-M parameters to verl parameters:
       # General Trainer Settings:
       trainer.nnodes=1 \
       trainer.n_gpus_per_node=8 \
       trainer.total_epochs=1 \
       trainer.save_freq=5 \
       # Data Settings:
       data.train_files=${DATASET_PATH} \
       data.train_batch_size=128 \
       data.max_prompt_length=4096 \
       data.max_response_length=1024 \
       # Model and Algorithm Settings:
       actor_rollout_ref.model.path=${PRETRAIN_MODEL_PATH} \
       actor_rollout_ref.actor.optim.lr=5e-7 \
       actor_rollout_ref.actor.ppo_mini_batch_size=1 \
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
       algorithm.adv_estimator=grpo \
       algorithm.use_kl_loss=True \
       algorithm.kl_loss_coef=1e-3 \
       algorithm.lambd=1.0 \
       algorithm.gamma=1.0 \
       # vLLM Settings (if using vLLM backend):
       actor_rollout_ref.rollout.name=vllm \
       actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
       actor_rollout_ref.rollout.enforce_eager=False \
       actor_rollout_ref.rollout.n=4 \
       # Checkpointing and Saving:
       trainer.save_path=${SAVE_PATH}/${MODEL_NAME} \
       trainer.load_checkpoint=False \
       # Other Settings (adapt as needed):
       # Note: Some OpenRLHF-M parameters like zero_stage, adam_offload, flash_attn
       # are often handled by backend configurations (e.g., FSDP) or dependencies (FlashAttention).
       # Refer to verl's documentation and example configs for how to set these.
       # example: actor_rollout_ref.actor.fsdp_config.zero_stage=3
       # example: actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
       ${WANDB_ARGS}

# The remote reward model setup is specific to OpenRLHF-M.
# In verl, reward functions are typically defined and configured within the verl framework.
# You will need to implement or configure your reward function according to verl's documentation
# on Implement Reward Function for Dataset: https://verl.readthedocs.io/en/latest/data/reward_function.html
# This may involve creating a custom reward function class and referencing it in your verl config.
