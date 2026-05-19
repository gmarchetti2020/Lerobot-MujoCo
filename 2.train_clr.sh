#!/bin/bash

# Default parameters
MODEL_TYPE=${1:-"pi0"} # Options: "pi0", "pi05", "xvla", "wall_x", "smolvla", "molmoact2"
EXPERIMENT_CONFIG=${2:-""}

if [ -z "$EXPERIMENT_CONFIG" ]; then
    EXPERIMENT_CONFIG="experiment-${MODEL_TYPE}.cfg"
fi

# Load .env variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set NCCL environment variables for distributed training in GCP G4
# single instance, multi-GPU setup. Adjust the interface name (e.g., "ens3") as needed.
unset NCCL_NET
export NCCL_SOCKET_IFNAME="ens7"
export NCCL_P2P_LEVEL="PHB"
export TOKENIZERS_PARALLELISM="false"

# Remove checkpoint and logs directories if they already exist to avoid conflicts
rm -rf ckpt
rm -rf logs

SUPPORTED_MODELS=("pi0" "pi05" "xvla" "wall_x" "smolvla" "molmoact2")
if [[ ! " ${SUPPORTED_MODELS[@]} " =~ " ${MODEL_TYPE} " ]]; then
    echo "Error: Unknown MODEL_TYPE '${MODEL_TYPE}'. Choose from: ${SUPPORTED_MODELS[*]}"
    exit 1
fi

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "Error: Config file not found: $EXPERIMENT_CONFIG"
    exit 1
fi

echo "Loaded config: $EXPERIMENT_CONFIG"

# Parse config file
get_config_val() {
    local key=$1
    # Extract the value from the config file, ignoring comments and stripping spaces
    grep "^${key}[[:space:]]*=" "$EXPERIMENT_CONFIG" | cut -d'=' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

DATASET_ROOT=$(get_config_val "DATASET_ROOT")
DATASET_REPO=$(get_config_val "DATASET_REPO")
POLICY_REPO=$(get_config_val "POLICY_REPO")
OUTPUT_DIR=$(get_config_val "OUTPUT_DIR")
JOB_NAME="$(get_config_val "JOB_NAME")_$(date +"%Y-%m-%d_%H-%M-%S")"
MAX_TRAIN_STEPS=$(get_config_val "MAX_TRAIN_STEPS")
CHUNK_SIZE=$(get_config_val "CHUNK_SIZE")
ACTION_STEPS=$(get_config_val "ACTION_STEPS")
BATCH_SIZE=$(get_config_val "BATCH_SIZE")

echo "MODEL_TYPE:      $MODEL_TYPE"
echo "DATASET_ROOT:    $DATASET_ROOT"
echo "DATASET_REPO:    $DATASET_REPO"
echo "POLICY_REPO:     $POLICY_REPO"
echo "OUTPUT_DIR:      $OUTPUT_DIR"
echo "JOB_NAME:        $JOB_NAME"
echo "MAX_TRAIN_STEPS: $MAX_TRAIN_STEPS"
echo "CHUNK_SIZE:      $CHUNK_SIZE"
echo "ACTION_STEPS:    $ACTION_STEPS"
echo "BATCH_SIZE:      $BATCH_SIZE"

# Download dataset from Hugging Face Hub
hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DATASET_ROOT"

# Build model-specific training flags using an array to preserve spaces correctly
case "$MODEL_TYPE" in
    "pi0")
        MODEL_FLAGS=(
            "--policy.type=pi0"
            "--policy.pretrained_path=lerobot/pi0_base"
            "--policy.compile_model=false"
            "--policy.gradient_checkpointing=true"
            "--policy.freeze_vision_encoder=false"
            "--policy.train_expert_only=false"
            "--policy.dtype=bfloat16"
            "--policy.device=cuda"
        )
        ;;
    "pi05")
        MODEL_FLAGS=(
            "--policy.type=pi05"
            "--policy.pretrained_path=lerobot/pi05_base"
            "--policy.compile_model=false"
            "--policy.gradient_checkpointing=true"
            "--policy.freeze_vision_encoder=false"
            "--policy.train_expert_only=false"
            "--policy.dtype=bfloat16"
            "--policy.device=cuda"
        )
        ;;
    "xvla")
        MODEL_FLAGS=(
            "--policy.path=lerobot/xvla-base"
            "--policy.freeze_vision_encoder=false"
            "--policy.freeze_language_encoder=true"
            "--policy.train_policy_transformer=true"
            "--policy.train_soft_prompts=true"
            "--policy.action_mode=auto"
            "--policy.dtype=bfloat16"
            "--policy.device=cuda"
            "--rename_map='{\"observation.image\": \"observation.images.image\", \"observation.left_scene_image\": \"observation.images.image2\", \"observation.wrist_image\": \"observation.images.image3\"}'"
        )
        ;;
    "wall_x")
        MODEL_FLAGS=(
            "--policy.type=wall_x"
            "--policy.pretrained_name_or_path=x-square-robot/wall-oss-flow"
            "--policy.prediction_mode=diffusion"
            "--policy.attn_implementation=eager"
            "--policy.device=cuda"
        )
        ;;
    "smolvla")
        MODEL_FLAGS=(
            "--policy.type=smolvla"
            "--policy.pretrained_path=lerobot/smolvla_base"
            "--policy.freeze_vision_encoder=false"
            "--policy.train_expert_only=false"
            "--policy.device=cuda"
        )
        ;;
    "molmoact2")
        MODEL_FLAGS=(
            "--policy.type=molmoact2"
            "--policy.device=cuda"
            "--dataset.video_backend=pyav"
            "--dataset.image_transforms.enable=true"
            "--policy.action_mode=both"
            "--policy.model_dtype=bfloat16"
            "--policy.num_flow_timesteps=8"
            "--policy.gradient_checkpointing=true"
            "--policy.freeze_embedding=true"
            "--policy.normalize_gripper=false"
            "--policy.enable_knowledge_insulation=false"
        )
        ;;
esac

echo "Model-specific flags:"
printf "  %s\n" "${MODEL_FLAGS[@]}"

# Train the policy. The trained model is pushed to the Hugging Face Hub under POLICY_REPO after training.
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=2 \
    --mixed_precision=bf16 \
    "$(which lerobot-train)" \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --policy.push_to_hub=true \
    --policy.repo_id="$POLICY_REPO" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    --wandb.enable=true \
    --steps="$MAX_TRAIN_STEPS" \
    --log_freq=50 \
    --eval_freq=-1 \
    --policy.chunk_size="$CHUNK_SIZE" \
    --policy.n_action_steps="$ACTION_STEPS" \
    --batch_size="$BATCH_SIZE" \
    "${MODEL_FLAGS[@]}"
