#!/bin/bash
#SBATCH --job-name=train_pi05
#SBATCH --partition=gpu              # <-- Change to your GPU partition name
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 1 task per node (torchrun handles per-GPU processes)
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64           # Adjust based on your cluster (8 workers/GPU × 8 GPUs)
#SBATCH --mem=0                      # Request all available memory on each node
#SBATCH --time=48:00:00              # Max walltime — adjust as needed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive                  # Exclusive node access for best performance

# ──────────────────────────────────────────────
# Environment Setup  (edit paths to match your cluster)
# ──────────────────────────────────────────────
# module load cuda/12.8              # Uncomment / adjust for your module system
# module load nccl
# source /path/to/your/venv/bin/activate   # Uncomment and set your virtualenv path

# Create log directory if it doesn't exist
mkdir -p logs

# ──────────────────────────────────────────────
# Distributed training environment variables
# ──────────────────────────────────────────────
export MASTER_PORT=$(( 29500 + RANDOM % 1000 ))   # Avoid port collisions across jobs
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export WORLD_SIZE=$(( SLURM_NNODES * 8 ))         # 2 nodes × 8 GPUs = 16

# NCCL tuning
export NCCL_DEBUG=INFO                # Set to WARN after debugging
export NCCL_IB_DISABLE=0             # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2          # GPU-Direct RDMA level (set 0 if unsupported)

# ──────────────────────────────────────────────
# Training Configuration  (edit to match your setup)
# ──────────────────────────────────────────────
DATASET_REPO_ID="Jeongeun/tutorial_v2"          # Your HuggingFace dataset repo
DATASET_ROOT="dataset/leader_data"              # Local path to dataset
OUTPUT_DIR="ckpt/pi05_multinode"
JOB_NAME="pi05_2node_16gpu"
PRETRAINED_PATH="lerobot/pi05_base"
STEPS=10000
BATCH_SIZE=2                                    # Per-GPU batch size (effective = 2 × 16 = 32)
CHUNK_SIZE=20
N_ACTION_STEPS=20
LOG_FREQ=50
SAVE_FREQ=2000

# ──────────────────────────────────────────────
# Launch with torchrun via srun
# ──────────────────────────────────────────────
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED_PATH" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.device=cuda \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --eval_freq=-1 \
    --wandb.enable=false
