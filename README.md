# 🤖 LeRobot MuJoCo Tutorial v2

A comprehensive tutorial for training and evaluating custom robotic manipulation policies using LeRobot and MuJoCo simulation.

## 📋 Table of Contents

- [🤖 LeRobot MuJoCo Tutorial v2](#-lerobot-mujoco-tutorial-v2)
  - [📋 Table of Contents](#-table-of-contents)
  - [🚀 Installation](#-installation)
  - [📁 Project Structure](#-project-structure)
    - [⌨️ Keyboard Teleoperation Demo](#️-keyboard-teleoperation-demo)
    - [📊 Dataset Visualization](#-dataset-visualization)
    - [🏋️ Model Training](#️-model-training)
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

### 🏋️ Model Training
**Files:** 
- `2.train.ipynb` - Baseline model training
- `3.eval_pi05.ipynb` - Pi-0.5 model evaluation
- `4.eval_groot.ipynb` - GRooT model evaluation

Train and evaluate baseline models with your data.

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