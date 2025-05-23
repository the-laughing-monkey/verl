# Running verl on RunPod: The Ultimate Deployment Guide

Welcome to the verl RunPodding guide!  
Here, we show you how to unleash verl on a RunPod instance. Whether you're working on large language model post-training or multimodal reinforcement learning, this guide will get you set up with a system that boasts substantial storage and leverages best practices for distributed training.

---

## Prerequisites

- A RunPod account with sufficient credits
- Basic familiarity with Linux commands and SSH
- An SSH client on your local machine  
- Patience—and maybe a snack!
- When using Ray for distributed training, **ensure that port 6379 is open** on the Ray head node for worker join and **port 8265 is open** for job submission and dashboard access. These ports are used for:
  - **Worker Join:** The head node listens on `http://<HEAD_NODE_IP>:6379` for worker nodes connecting to the cluster.
  - **Job Submission & Dashboard:** The head node listens on `http://<HEAD_NODE_IP>:8265` for job submissions and dashboard communications.
- **Before running any training or job submission scripts, be sure to start Ray on your pod.**  
  Please refer to the [Ray documentation](https://docs.ray.io/).

---

## Step-by-Step Instructions

### 1. Create Your Storage Volume

1. Log in to your RunPod account.
2. Navigate to the **Volumes** section and click **Create Volume**.
3. Name your volume (we recommend `data`) and set the size (2000GB).
4. Select your preferred datacenter and click **Create**.

### 2. Launch Your Pod

1. In the **Pods** section, click **+ Deploy**.

Choose the number of nodes you want to deploy.
 

2. Set your GPU count (more than one if you like parallel power!).

3. Choose the correct template:
RunPod Pytorch 2.4.0  (by default it pickes 2.2.1) OR 2.8 for Blackwell cards.

4. Click **Edit Template** to adjust:
   - Container disk size (100GB is a good start).
   - Attach your volume by mounting it to `/data`.
   - Enable a public IP.

5. Ensure that "ssh" is checked, then click **Deploy**.

### 3. Configure Your SSH Access

1. Generate your SSH key locally:
```bash
   ssh-keygen -t my_runpod_key -C "your_email@example.com"
   cat ~/.ssh/my_runpod_key.pub
```
2. Log into your RunPod account and paste your public key under **SSH Public Keys**.

3. Once your pod is live, note its IP and SSH port. Then connect using:
```bash
   ssh -i ~/.ssh/my_runpod_key root@<POD_IP_ADDRESS> -p <SSH_PORT>
```

### 4. Updates your OS, Set Up Your Python Environment and Install verl

1. Update the system and install Python tools:
```bash
   apt update && apt upgrade -y && apt install -y python3-pip python3-venv python3-dev build-essential git curl vim lsof net-tools rsync libopenmpi-dev build-essential dkms dnsutils dnsutils iputils-ping
```


2. Create a virtual environment in your data directory:


```bash
   mkdir /workspace
   cd /workspace
   python3 -m venv verl-env
   source verl-env/bin/activate
```

### 5. Clone the verl repository

```bash
  git clone https://github.com/the-laughing-monkey/verl.git
```

### 6. Pip install huggingface hub so you can download models and data while the rest installs
```bash
  pip install huggingface_hub
```

# Now you can use the examples/scripts/downloaders/download_model.py or examples/scripts/downloaders/download_mathv60k.py to download models and a sample dataset in another terminal to work in parallel.

### 7. Set the file descriptor limit to 65536 to avoid "Too many open files" error
```bash
ulimit -n 65536
# OR FOR Unlimited do:
ulimit -s unlimited
```
---

### 8. Run setup script to install everything else

```bash
  cd ./verl
  bash ./scripts/setup_dj.sh
```

### 8. (Optional) Set Your WandB API Key

If you wish to use Weights & Biases (wandb) for experiment tracking, set your API key:

```bash
    export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
This step is optional but recommended for more integrated experiment monitoring.


### 9. Prepare Your Cache

Move model caches to your larger `/data` volume to conserve space:
```bash
    mkdir -p /data/cache-models/huggingface/hub /data/cache-models/modelscope/hub /data/cache-ray
    rm -rf /root/.cache/huggingface && ln -s /data/cache-models/huggingface /root/.cache/huggingface
    rm -rf /root/.cache/modelscope && ln -s /data/cache-models/modelscope /root/.cache/modelscope
    # DO NOT DO THE RAY PART ON A NETWORK DISK. Insread make your local /root disk LARGE. Like 2GB.
    rm -rf /root/.cache/ray && ln -s /data/cache-ray /root/.cache/ray
    # Verify symlinks
    ls -la /root/.cache/
```

This is a critical step because:
- Model training checkpoints can be large (multiple GB each)
- The default container disk (50GB) will quickly fill up during training
- Moving these to your data volume (500GB-1000GB) prevents "No space left on device" errors

---

### 10. Download and Prepare the MathV60K Dataset

Before running a training job, you'll need to prepare the dataset:

1. Create the datasets directory:
```bash
    mkdir -p /data/datasets
```

2. Download and prepare the geo3k dataset:

```bash
    cd /workspace/verl
    python3 examples/downloaders/download_geo3k.py --root_dir /data/datasets/geo3k
```

3. Process the geo3k dataset:
```bash
    python3 examples/data_preprocess/geo3k.py --local_dir /data/datasets/geo3k
```


4. Download the Qwen2.5-VL-3B model:
```bash
    cd /workspace/verl
    python3 examples/downloaders/download_model.py --model_name Qwen/Qwen2.5-VL-3B-Instruct
```


### 11. Set your NCCL environment variables to use the eth1 interface:

CRITICAL: RunPod only allows internode communication over eth1. So you need to set your NCCL to use the eth1 IP or NCCL will fail to update weights across nodes.

```bash
    export NCCL_DEBUG=INFO
```


### 12. Start Ray (if you are using Ray - can be run with FSDP or Megatron as well on veRL)

2. Stop any running Ray instances (if any):
```bash
    ray stop
```

3. Start the Ray head node bound to all available IPs:
```bash
    ray start --head --node-ip-address 0.0.0.0 --port=6379 --dashboard-port=8265   --temp-dir ~/.cache/ray
```


### 13. Run Your First verl Training Job with MathV60K

Now you're ready to launch a training job using the MathV60K dataset and the Qwen2.5-VL-3B model:

1. First, copy the script to your pod:

```bash
    cd /workspace/verl
    mkdir -p ./scripts
    cp examples/tests/train_grpo_qwen2_5_vl_3b_singlenode_simple.sh ./scripts/my_train_script.sh
```

2. Edit the script to match your pod's GPU configuration:

```bash
    nano my_train_script.sh
```

Make the following adjustments:

Adust for your number of GPUs and nodes.

```bash
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
```

3. Run the adjusted training script:
```bash
    bash ./scripts/my_train_script.sh
```


### 14. Monitoring NVIDIA GPU Memory

To monitor the NVIDIA GPU memory usage while the script loads and runs, open a new terminal session (or use a multiplexer like tmux/screen) and run:

```bash
watch -n 1 nvidia-smi
```

# or

```bash
watch -n 1 "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
```

# or

```bash
watch -n 1 "echo 'GPU   Total(MiB)   Used(MiB)'; nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{printf \"%-3s %-12s %-10s\n\", \$1, \$2, \$3}'"
```

### 16. Monitoring and Managing Disk Space

Running out of disk space is a common issue during training. To monitor disk usage:

```bash
# Check overall disk usage
df -h

# Find largest directories and files in /root
du -h --max-depth=2 /root | sort -hr | head -20

# Find largest directories in Ray cache
du -h --max-depth=2 /data/cache-ray | sort -hr | head -20

# Find large checkpoint files
find /data -name "*.pt" -size +1G | xargs ls -lh
```

If you're running low on disk space despite using the data volume:

```bash
# Clear Triton cache (safe to delete)
rm -rf /root/.triton/autotune

# Clear older Ray session directories (if not using the symlink setup)
find /root/.cache/ray/session_* -maxdepth 0 -type d | sort | head -n -2 | xargs rm -rf

# Reduce checkpoint frequency in training scripts
# Example: Change --save_steps 10 to --save_steps 50

# Limit the number of checkpoints kept
# Example: Add --max_ckpt_num 2 to training arguments
```

For critical low disk situations, you can safely clear caches:

```bash
# Clear PyTorch hub cache
rm -rf /root/.cache/torch/hub/*

# Remove older checkpoints if needed
find /data/checkpoints -name "global_step*" | sort | head -n -2 | xargs rm -rf
```

### Troubleshooting

#### "Too many open files" error (Raylet or other Ray process)

This error occurs when a Ray process (like the raylet) exceeds the operating system's limit on the number of open file descriptors. Ray uses file descriptors for various resources, including network sockets for internal communication (gRPC). With a large number of tasks or actors, Ray can open many connections, hitting the default limit.

To resolve this, increase the file descriptor limit (`ulimit -n`) for the user running the Ray processes *before* starting Ray. A common value is 65536.

```bash
ulimit -n 65536
# Then run your ray start or training command
ray start --head ... # or ray start --address=...
```

You should apply this command in the terminal session *before* executing the Ray start command on both head and worker nodes.

### 12. Prepare Your Dataset for verl

Before running a training job, you'll need to prepare your dataset in a format compatible with verl. verl uses dataset configurations specified within its training configuration files (often YAML). Reward functions are also integrated within verl's framework rather arrested than requiring a separate remote server.

verl supports multimodal datasets. The Qwen2.5-VL-32B model you are using is a visual language model, so you will likely need a multimodal dataset like MathV60K or geo3k.

**Option 1: Using the MathV60K Dataset**

If you plan to use the MathV60K dataset (as covered in the original OpenRLHF-M setup), follow these steps:

1. Create the datasets directory:
```bash
    mkdir -p /data/datasets
```

2. Download and prepare the MathV60K dataset:

On the head node:
```bash
    cd /workspace/verl
    python3 examples/scripts/downloaders/download_mathv60k.py --root_dir /data/datasets/VerMulti
```

This script will download and prepare the MathV60K dataset in `/data/datasets/VerMulti`.

**Option 2: Using the geo3k Dataset (for replicating verl examples)**

If you want to run examples that use the geo3k dataset (e.g., the Qwen2.5-VL-7B GRPO example), you can use the provided downloader script.

1. Navigate to the verl examples downloaders directory:
```bash
    cd /workspace/verl/examples/downloaders
```

2. Run the geo3k downloader script:
```bash
    python3 download_geo3k.py --root_dir /data/datasets/geo3k
```

This script will download the geo3k dataset files into `/data/datasets/geo3k/`. The train file will be located at `/data/datasets/geo3k/train.parquet` and the test file at `/data/datasets/geo3k/test.parquet`.

Refer to the verl documentation on [Data Preparation](https://verl.readthedocs.io/en/latest/data/data_prep.html) and [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/data/reward_function.html) for detailed instructions on preparing your specific dataset and integrating your reward function. Ensure the `data.train_files` and `data.val_files` parameters in your training script point to the correct dataset paths.

### 15. Run Your verl Training Job

Unlike OpenRLHF-M, verl typically uses YAML configuration files to manage training parameters. While a base configuration is defined in a YAML file, you can override specific settings directly from the command line using a `parameter.subparameter=value` syntax.

Here's an example command to launch a verl training job using `verl.trainer.main_ppo`, adapting settings from the previous OpenRLHF-M script. Note that this assumes you have prepared your dataset and integrated your reward function according to the verl documentation ([Data Preparation](https://verl.readthedocs.io/en/latest/data/data_prep.html), [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/data/reward_function.html)). You will also likely need a base YAML configuration file which this command will override.

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=${PRETRAIN_MODEL_PATH} \
    data.train_files=${DATASET_PATH} \
    trainer.n_gpus_per_node=8 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='your_verl_project' \
    trainer.experiment_name='${MODEL_NAME}' \
    # Add other overrides as needed, referring to verl documentation and example YAMLs
    # Example overrides for batch sizes, learning rate, etc. might look like:
    # data.train_batch_size=128 \
    # actor_rollout_ref.actor.optim.lr=5e-7 \
    # algorithm.use_kl_loss=True \
    # algorithm.kl_loss_coef=1e-3 \
    # trainer.save_freq=5 \
    $@
```

**Explanation of key overrides:**

*   `actor_rollout_ref.model.path`: Specifies the path to your pre-trained model (e.g., Qwen/Qwen2.5-VL-32B-Instruct).
*   `data.train_files`: Points to your prepared training dataset file(s).
*   `trainer.n_gpus_per_node`: Sets the number of GPUs to use per node.
*   `trainer.logger`: Configures logging (e.g., to console and WandB).
*   `trainer.project_name` and `trainer.experiment_name`: Sets the project and experiment names for WandB logging.

Consult the verl documentation and the example scripts in the `examples` directory for the full range of configuration options and how to structure your base YAML file.
