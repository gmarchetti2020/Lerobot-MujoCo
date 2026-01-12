# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# root = './dataset/leader_data'
# dataset = LeRobotDataset('Jeongeun/tutorial_v2',root = root )
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
ROOT = './dataset/leader_data'
repo_id = 'Jeongeun/tutorial_v2_pi05'
ckpt = 'ckpt/tutorial_v2_pi05'
device = 'cuda'

# dataset_metadata = LeRobotDatasetMetadata("Jeongeun/tutorial_v2", root=ROOT)

policy = PI05Policy.from_pretrained(repo_id, cache_dir=ckpt)
policy.to(device)