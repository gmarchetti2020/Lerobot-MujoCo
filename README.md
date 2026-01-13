# 🤖 LeRobot MuJoCo Tutorial v2

A comprehensive tutorial for training and evaluating custom robotic manipulation policies using LeRobot and MuJoCo simulation.

## 📋 Table of Contents

- [🤖 LeRobot MuJoCo Tutorial v2](#-lerobot-mujoco-tutorial-v2)
  - [📋 Table of Contents](#-table-of-contents)
  - [🚀 Installation](#-installation)
  - [📁 Project Structure](#-project-structure)
    - [⌨️ Keyboard Teleoperation Demo](#️-keyboard-teleoperation-demo)
    - [📊 Dataset Visualization](#-dataset-visualization)
  - [🏋️ Baseline Model Training](#️-baseline-model-training)
    - [📌 Pi-0.5 Training](#-pi-05-training)
    - [🚀 GR00T Training](#-gr00t-training)
  - [📈 Model Evaluation](#-model-evaluation)
    - [📊 Pi-0.5 Evaluation](#-pi-05-evaluation)
  - [Custom Policy Training and Evaluation](#custom-policy-training-and-evaluation)
    - [🔄 Data Transformation](#-data-transformation)
    - [🎓 Custom Policy Training](#-custom-policy-training)
    - [✅ Custom Policy Evaluation](#-custom-policy-evaluation)
  - [📊 Model Performance](#-model-performance)
  - [🔧 Custom Policy Implementation](#-custom-policy-implementation)
    - [📝 Training Your Custom Policy](#-training-your-custom-policy)
    - [📝 Evaluating Your Custom Policy](#-evaluating-your-custom-policy)
  - [📡 Data Collection with Leader Arm](#-data-collection-with-leader-arm)
    - [✋ Prerequisites](#-prerequisites)
    - [🔧 Procedure](#-procedure)
  - [💬 Contact](#-contact)

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

### ⌨️ Keyboard Teleoperation Demo
**File:** `0.teleop.ipynb`

Interactive keyboard teleoperation for manual robot control and data collection.

**Controls:**
- **WASD** - XY plane movement
- **R/F** - Z-axis movement
- **Q/E** - Tilt adjustment
- **Arrow Keys** - Rotation control
- **Spacebar** - Toggle gripper state
- **Z** - Reset environment (discard episode data)

### 📊 Dataset Visualization
**File:** `1.Visualize.ipynb`

Download and visualize datasets from Hugging Face Hub.

**Quick Start:**
```bash
python download_data.py
```

**Python Usage:**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = './dataset/leader_data'
dataset = LeRobotDataset('Jeongeun/tutorial_v2', root=root)
```

Running this code will automatically download the dataset.


## 🏋️ Baseline Model Training

### 📌 Pi-0.5 Training
**File:** `2.train.ipynb` (First Section)

Train Pi-0.5 model on your dataset using the LeRobot training pipeline.

**Prerequisites:**
```bash
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi
```

**Training Command:**
```bash
lerobot-train \
    --dataset.repo_id=Jeongeun/tutorial_v2 \
    --dataset.root=dataset/leader_data \
    --policy.type=pi05 \
    --policy.push_to_hub=true \
    --policy.repo_id={YOUR REPO} \
    --output_dir=./ckpt/tutorial_v2_pi05 \
    --job_name=tutorial_v2_pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=5000 \
    --log_freq=50 \
    --eval_freq=-1 \
    --policy.device=cuda \
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --batch_size=32
```

**Key Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy.type` | `pi05` | Policy architecture type |
| `policy.pretrained_path` | `lerobot/pi05_base` | Pre-trained model checkpoint |
| `policy.compile_model` | `true` | Enable model compilation for faster inference |
| `policy.dtype` | `bfloat16` | Use bfloat16 for memory efficiency |
| `steps` | `5000` | Total training steps |
| `batch_size` | `32` | Batch size for training |
| `chunk_size` | `20` | Action chunk size |
| `n_action_steps` | `20` | Number of action prediction steps |

**Output:** 
- Trained model saved to `./ckpt/tutorial_v2_pi05`
- Optional: Pushed to Hugging Face Hub if `push_to_hub=true`

**⏱️ Training Time:** ~2-4 hours on single GPU

---

### 🚀 GR00T Training
**File:** `2.train.ipynb` (Second Section)

Train GR00T N 1.5 model on your dataset.

**Prerequisites:**
```bash
pip install ninja "packaging>=24.2,<26.0"
pip install peft
pip install dm-tree==0.1.9
pip install -U transformers
pip install flash-attn==2.7.3 --no-build-isolation
```

**Training Command:**
```bash
lerobot-train \
    --dataset.repo_id=Jeongeun/tutorial_v2 \
    --dataset.root=dataset/leader_data \
    --policy.type=groot \
    --policy.push_to_hub=true \
    --policy.repo_id={YOUR REPO} \
    --policy.tune_diffusion_model=false \
    --output_dir=ckpt/tutorial_v2_groot \
    --job_name=tutorial_v2_groot \
    --wandb.enable=false \
    --steps=3000 \
    --policy.chunk_size=20 \
    --policy.n_action_steps=20 \
    --batch_size=32
```

**Key Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy.type` | `groot` | Policy architecture type (Generalist Robot Transformer) |
| `policy.tune_diffusion_model` | `false` | Disable diffusion model fine-tuning |
| `steps` | `3000` | Total training steps |
| `batch_size` | `32` | Batch size for training |
| `chunk_size` | `20` | Action chunk size |
| `n_action_steps` | `20` | Number of action prediction steps |

**Output:**
- Trained model saved to `ckpt/tutorial_v2_groot`
- Optional: Pushed to Hugging Face Hub if configured

**⏱️ Training Time:** ~1-2 hours on single GPU
## 📈 Model Evaluation
### 📊 Pi-0.5 Evaluation
**File:** `3.eval_pi05.ipynb`

Evaluate the trained Pi-0.5 model on your environment.

**Prerequisites:**
```bash
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi
```

**Key Setup:**
```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from src.env.env import RILAB_OMY_ENV

# Load model
repo_id_or_path = 'Jeongeun/tutorial_v2_pi05'
policy = PI05Policy.from_pretrained(repo_id_or_path)
policy.to('cuda')

# Load preprocessor/postprocessor
preprocessor = PolicyProcessorPipeline.from_pretrained(repo_id_or_path, ...)
postprocessor = PolicyProcessorPipeline.from_pretrained(repo_id_or_path, ...)

# Load environment
env_conf = json.load(open('./configs/train.json'))
omy_env = RILAB_OMY_ENV(cfg=env_conf, action_type='joint', obs_type='joint_pos')
```

**Evaluation Configuration:**
```python
TEST_EPISODES = 20
MAX_EPISODE_STEPS = 10_000
```

**Run Evaluation:**
- Loops through episodes
- Captures agent and wrist camera images (256×256)
- Preprocesses observations
- Selects actions via policy
- Postprocesses actions and steps environment
- Reports success rate

**Output:** Average success rate over 20 episodes


## Custom Policy Training and Evaluation
### 🔄 Data Transformation
**File:** `10.transform.ipynb`

Define action and observation spaces, then transform your dataset for training.

**Configuration:**
```python
action_type = 'delta_joint'      # 'joint' | 'delta_joint' | 'delta_eef_pose' | 'eef_pose'
proprio_type = 'eef_pose'        # 'joint' | 'eef_pose'
observation_type = 'image'       # 'image' | 'object_pose'
image_aug_num = 2                # Number of augmented images per original image
transformed_dataset_path = './dataset/transformed_data'
```

**Configuration Details:**
| Parameter | Description | Options |
|-----------|-------------|---------|
| `action_type` | Action representation format | `joint`, `delta_joint`, `eef_pose`, `delta_eef_pose` |
| `proprio_type` | Proprioceptive information representation | `joint`, `eef_pose` |
| `observation_type` | Input modality | `image`, `object_pose` |
| `image_aug_num` | Augmented trajectories for image features | Integer |

**Command Line Usage:**
```bash
python transform.py \
  --action_type delta_eef_pose \
  --proprio_type eef_pose \
  --observation_type image \
  --image_aug_num 2
```

### 🎓 Custom Policy Training
**File:** `11.train_custom.ipynb`

Train MLP or Transformer models with your transformed dataset.

**Configuration Example:**
```python
@PreTrainedConfig.register_subclass("omy_baseline")
@dataclass
class BaselineConfig(PreTrainedConfig):
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 5
    n_action_steps: int = 5

    # Architecture
    backbone: str = 'mlp'  # 'mlp' or 'transformer'
    vision_backbone: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    projection_dim: int = 128
    freeze_backbone: bool = True

    # Model dimensions
    n_hidden_layers: int = 5
    hidden_dim: int = 512

    # Transformer-specific parameters
    n_heads: int = 4
    dim_feedforward: int = 2048
    feedforward_activation: str = "gelu"
    dropout: float = 0.1
    pre_norm: bool = True
    n_encoder_layers: int = 6

    # Training parameters
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6
    lr_warmup_steps: int = 1000
    total_training_steps: int = 500000

# Initialize policy configuration
cfg = BaselineConfig(
    chunk_size=10,
    n_action_steps=10,
    backbone='mlp',
    optimizer_lr=5e-4,
    n_hidden_layers=10,
    hidden_dim=512,
    vision_backbone='facebook/dinov3-vitb16-pretrain-lvd1689m',
    projection_dim=128,
    freeze_backbone=True,
)
```

**Command Line Training:**
```bash
python train_custom.py \
  --dataset_path DATASET_PATH \
  --batch_size BATCH_SIZE \
  --num_epochs NUM_EPOCHS \
  --ckpt_path CKPT_PATH \
  --chunk_size CHUNK_SIZE \
  --n_action_steps N_ACTION_STEPS \
  --learning_rate LEARNING_RATE \
  --backbone BACKBONE \
  --n_hidden_layers N_HIDDEN_LAYERS \
  --hidden_dim HIDDEN_DIM \
  --vision_backbone {facebook/dinov3-vitb16-pretrain-lvd1689m,facebook/dinov2-base} \
  --projection_dim PROJECTION_DIM \
  --freeze_backbone FREEZE_BACKBONE
```

### ✅ Custom Policy Evaluation
**File:** `12.eval_custom.ipynb`

Evaluate your trained policies in simulation environment.

<img src="./media/baseline.gif" width="480" height="360"></img>

---

## 📊 Model Performance

| Model | Clean Image | Noisy Color Image |
|-------|:---:|:---:|
| [🎯 MLP with GT Object Pose](https://huggingface.co/Jeongeun/mlp_obj_deep_learning_2025_joint) | 65% ✅ | 65% ✅ |
| [🖼️ MLP with Image (DINOv3)](https://huggingface.co/Jeongeun/mlp_image_deep_learning_2025_joint) | 50% | 40% |
| [🚀 SmolVLA with Image](https://huggingface.co/Jeongeun/smolvla_deep_learning_2025_joint) | 65% ✅ | 10% ⚠️ |

> **Note:** Action: Target Joint Position | State: Current Joint Position
> 
> ⚠️ Color augmentation was not applied during vision model training.

---

## 🔧 Custom Policy Implementation

👉 Refer to [src/policies/README.md](./src/policies/README.md) for detailed instructions.

### 📝 Training Your Custom Policy

In `11.train_custom.ipynb`, update the **first cell**:
```python
from src.policies.your_policy.configuration import YourPolicyConfig
from src.policies.baseline.processor import make_baseline_pre_post_processors
from src.policies.your_policy.modeling import YourPolicy
```

Update the **third cell** to instantiate your configuration:
```python
cfg = YourPolicyConfig(
    chunk_size=10,
    n_action_steps=10,
    # Your custom parameters
)
```
Update the **fifth cell** to build preprocessor and postprocessor
```python
preprocessor, postprocessor = make_baseline_pre_post_processors(
        config=cfg,
        dataset_stats= ds_meta.stats
    )
```
Update the **sixth cell** to instantiate your policy:
```python
policy = YourPolicy(**kwargs)
```

### 📝 Evaluating Your Custom Policy

In `12.eval_custom.ipynb`, update the **first cell**:
```python
from src.policies.your_policy.modeling import YourPolicy
```

Update the **third cell** to load your trained model:
```python
policy = YourPolicy.from_pretrained(CKPT, **kwargs)
```

---

## 📡 Data Collection with Leader Arm

### ✋ Prerequisites
- ✅ ROS2 installed on your system
- ✅ ROBOTIS Open Manipulator hardware
- ✅ Leader arm setup complete

### 🔧 Procedure

**Terminal 1:** Launch ROS2 hardware driver
```bash
ros2 launch open_manipulator_bringup hardware_y_leader.launch.py
```

**Terminal 2:** Run leader arm interface
```bash
python leader.py
```

**Terminal 3:** Start data collection
```bash
python collect_data.py
```

💾 Your collected data will be saved in the dataset directory!

---

## 💬 Contact

**👤 Jeongeun Park**  
📧 Email: [baro0906@korea.ac.kr](mailto:baro0906@korea.ac.kr)

---

<div align="center">

Made with ❤️ for robot learning research

</div>