#!/usr/bin/env python3
"""
Download a specified dataset from Hugging Face or ModelScope.
"""

import argparse
import os
from huggingface_hub import snapshot_download

print("=================================================================")
print("Generic Dataset Downloader")
print("=================================================================")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download a dataset from Hugging Face or ModelScope')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Name of the dataset repository on Hugging Face or ModelScope (e.g., organization/dataset_name)')
parser.add_argument('--root_dir', type=str, required=True,
                    help='Root directory for storing the dataset')
args = parser.parse_args()

# Set root directory and ensure it exists
DIR_ROOT = args.root_dir
DATASET_NAME = args.dataset_name

print(f"[1/2] Creating directory: {DIR_ROOT}")
os.makedirs(DIR_ROOT, exist_ok=True)
print(f"      Using root directory: {DIR_ROOT}")

# Download dataset files from Hugging Face or ModelScope
print(f"[2/2] Downloading dataset \"{DATASET_NAME}\" to {DIR_ROOT}...")

try:
    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=DIR_ROOT,
        local_dir_use_symlinks=False
    )
    print(f"      Downloaded dataset \"{DATASET_NAME}\" to {DIR_ROOT}")

except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please ensure the dataset name is correct and you have the huggingface_hub library installed (`pip install huggingface_hub`).")
    exit(1)

print("=================================================================")
print(f"Dataset download complete!")
print(f"Dataset \"{DATASET_NAME}\" downloaded to {DIR_ROOT}")
print("Please refer to the dataset documentation for specific file paths and structure.")
print("=================================================================") 