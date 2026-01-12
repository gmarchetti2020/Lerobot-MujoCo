from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    'Jeongeun/tutorial_v2', './dataset/leader_data'
)
# dataset = LeRobotDataset(
#     'Jeongeun/deep_learning_2025', './dataset/demo_data'
# )
dataset.push_to_hub(
    upload_large_folder=True
)