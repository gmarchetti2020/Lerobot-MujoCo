from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    'msavchen-nasa/clr_mujoco_dataset', './dataset/clr_teleoperation_dataset'
)
# dataset = LeRobotDataset(
#     'Jeongeun/deep_learning_2025', './dataset/demo_data'
# )
dataset.push_to_hub(
    upload_large_folder=True
)