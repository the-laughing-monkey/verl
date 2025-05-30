#!/bin/bash

# Setup script for installing python packages for verl
# Assumes you are in the activated virtual environment.
# This script adapts instructions from sections 4, 5, and 6 of the deployment guide for verl.

#############################
# Define Verl Repository Root
#############################

# Use an absolute path for robustness, assuming /workspace is the intended base.
VERL_REPO_ROOT="/workspace/verl"

#############################
# Set Working Root and Clone Directory
#############################

WORKING_DIR="$(pwd)" # Use current working directory
VERL_CLONE_DIR="$WORKING_DIR/verl"

#############################
# Set maximum number of open file descriptors
#############################

echo "Setting maximum number of open file descriptors to unlimited"
ulimit -n 65536

#############################
# 1. Install Core Python Packages and PyTorch
#############################

# Define core required python packages
PACKAGES="pip wheel packaging setuptools huggingface_hub qwen-vl-utils sgl-kernel mathruler deepspeed vllm"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing core python packages: $PACKAGES"
pip install $PACKAGES


echo "Installing a PyTorch version compatible with verl, vLLM, and CUDA 12.4"
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Core python package and PyTorch installation complete."

#############################
# 2. Clone verl and Install Core Requirements
#############################

# Set repository directory in WORKING_DIR
# REPO_DIR="$WORKING_DIR/OpenRLHF-M" # Original OpenRLHF-M directory
# VERL_REPO_DIR="$WORKING_DIR/verl" # Removed redundant variable

# Remove existing OpenRLHF-M directory if it exists from previous runs
# if [ -d "$REPO_DIR" ]; then
#     echo "Removing existing OpenRLHF-M repository: $REPO_DIR"
#     rm -rf "$REPO_DIR"
# fi

if [ ! -d "$VERL_REPO_ROOT" ]; then
    echo "Cloning verl repository into $VERL_REPO_ROOT"
    git clone https://github.com/the-laughing-monkey/verl.git "$VERL_REPO_ROOT"
fi

echo "Changing directory to verl repository: $VERL_REPO_ROOT"
cd "$VERL_REPO_ROOT"

# Install vLLM and SGLang explicitly with specified versions
echo "Installing vLLM"
pip install vllm

echo "Installing sglang"
pip install sglang[all]

# Install verl in editable mode with default extra (excluding vllm and sglang dependencies already installed)
echo "Installing verl with default extra"
pip install -e .[default]


# Install flash_attn separately with no-build-isolation and specific version/index url
# This is kept separate as it's a common source of issues and explicitly controlling it can help
echo "Installing flash-attn with --no-build-isolation"
pip install --no-build-isolation flash-attn

echo "Verl and required dependencies installation complete."

#############################
# 3. Install Megatron-LM and TransformerEngine (Optional)
#############################

# Set USE_MEGATRON to 1 to enable Megatron-LM installation, 0 to disable
#USE_MEGATRON=${USE_MEGATRON:-0}

#if [ $USE_MEGATRON -eq 1 ]; then
#    echo "Installing TransformerEngine and Megatron-LM from source (requires cuDNN development headers)"
#    echo "Note: This step may fail if cuDNN development headers are not properly installed on the system."
#    # Install TransformerEngine from git
#    NVTE_FRAMEWORK=pytorch pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
#    # Install Megatron-LM core from git
#    pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
#    echo "Megatron-LM and TransformerEngine installation steps completed (check output for errors)."
#else
#    echo "Megatron-LM and TransformerEngine installation skipped (USE_MEGATRON=0)."
#fi


echo "installation complete."

# Verify installed versions
echo "Installed package versions:"
python -c "import torch; print(f'torch: {torch.__version__}')" || echo "torch: Not Found or Failed to Import"
python -c "import vllm; print(f'vllm: {vllm.__version__}')" || echo "vllm: Not Found or Failed to Import"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" || echo "flash-attn: Not Found or Failed to Import"
python -c "import ray; print(f'ray: {ray.__version__}')" || echo "ray: Not Found or Failed to Import"
python -c "import verl; print('verl: Installed')" || echo "verl: Not Found or Failed to Import"