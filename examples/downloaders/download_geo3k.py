#!/usr/bin/env python3
"""
Download the hiyouga/geometry3k dataset from Hugging Face.
"""

import argparse
import os

from huggingface_hub import hf_hub_download

print("=================================================================")
print("Geometry3k Dataset Downloader")
print("=================================================================")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download geometry3k dataset')
parser.add_argument('--root_dir', type=str, default="/data/datasets/geo3k",
                    help='Root directory for storing the dataset')
args = parser.parse_args()

# Set root directory and ensure it exists
DIR_ROOT = args.root_dir
print(f"[1/2] Creating directory: {DIR_ROOT}")
os.makedirs(DIR_ROOT, exist_ok=True)
print(f"      Using root directory: {DIR_ROOT}")

# Download train and test parquet files from Hugging Face
repo_id = "hiyouga/geometry3k"

print(f"[2/2] Downloading train and test parquet files from {repo_id}...")

# Corrected filenames and path within the repository
train_filename_repo = "data/train-00000-of-00001.parquet"
test_filename_repo = "data/test-00000-of-00001.parquet"

# Define local filenames for clarity, though hf_hub_download handles this
train_filename_local = "train.parquet"
test_filename_local = "test.parquet"

try:
    print(f"      Downloading {train_filename_repo}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=train_filename_repo, # Specify the actual path in the repo
        repo_type="dataset",
        local_dir=DIR_ROOT,
        local_dir_use_symlinks=False
    )
    print(f"      Downloaded {train_filename_repo} to {DIR_ROOT}")

    print(f"      Downloading {test_filename_repo}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=test_filename_repo, # Specify the actual path in the repo
        repo_type="dataset",
        local_dir=DIR_ROOT,
        local_dir_use_symlinks=False
    )
    print(f"      Downloaded {test_filename_repo} to {DIR_ROOT}")

except Exception as e:
    print(f"Error downloading files: {e}")
    print("Please ensure you have the huggingface_hub library installed (`pip install huggingface_hub`).")
    exit(1)

print("=================================================================")
print(f"Dataset download complete!")
print(f"Files saved to {DIR_ROOT}")
print(f"- Train file: {os.path.join(DIR_ROOT, train_filename_repo)}")
print(f"- Test file: {os.path.join(DIR_ROOT, test_filename_repo)}")
print(f"To use this dataset with verl, set data.train_files and data.val_files to these paths, or consider renaming the downloaded files for simplicity.")
print("=================================================================") 