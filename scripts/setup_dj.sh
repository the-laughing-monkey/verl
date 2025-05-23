#!/bin/bash

# Setup script for installing python packages for verl
# Assumes you are in the activated virtual environment.
# This script adapts instructions from sections 4, 5, and 6 of the deployment guide for verl.

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

# Define core required python packages, including those from the original script
# and potentially others not covered by verl's setup.py install_requires
PACKAGES="pip wheel packaging setuptools huggingface_hub ring_flash_attn"

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

if [ ! -d "$VERL_CLONE_DIR" ]; then
    echo "Cloning verl repository into $VERL_CLONE_DIR"
    git clone https://github.com/the-laughing-monkey/verl.git "$VERL_CLONE_DIR"
fi

echo "Changing directory to verl repository: $VERL_CLONE_DIR"
cd "$VERL_CLONE_DIR"

# Install verl with default and vllm extras
echo "Installing verl with default and vllm extras"
pip install -e .[default,vllm]

echo "installation complete."

# Verify installed versions
echo "Installed package versions:"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')" >/dev/null 2>&1 || echo "vllm: Not Found or Failed to Import"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')" >/dev/null 2>&1 || echo "flash-attn: Not Found or Failed to Import"
python -c "import ray; print(f'ray: {ray.__version__}')"