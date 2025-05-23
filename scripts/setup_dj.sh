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

# Define core required python packages, excluding flash_attn for separate installation
PACKAGES="pip wheel packaging setuptools huggingface_hub"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing core python packages: $PACKAGES"
pip install $PACKAGES


echo "Installing a PyTorch version compatible with verl, vLLM, and CUDA 12.4"
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Core python package and PyTorch installation complete."


#############################
# 2. Clone and Install verl
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

# Install verl with default and vllm extras
echo "Installing verl with default and vllm extras"
pip install -e .[default,vllm]

# Install flash_attn separately with no-build-isolation
echo "Installing flash-attn with --no-build-isolation"
pip install --no-build-isolation flash-attn

echo "installation complete."

# Verify installed versions
echo "Installed package versions:"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')" >/dev/null 2>&1 || echo "vllm: Not Found or Failed to Import"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" >/dev/null 2>&1 || echo "flash-attn: Not Found or Failed to Import"
python -c "import ray; print(f'ray: {ray.__version__}')" >/dev/null 2>&1 || echo "ray: Not Found or Failed to Import"
python -c "import verl; print('verl: Installed')" >/dev/null 2>&1 || echo "verl: Not Found or Failed to Import"
python -c "import megatron.core; print('megatron-core: Installed')" >/dev/null 2>&1 || echo "megatron-core: Not Found or Failed to Import"