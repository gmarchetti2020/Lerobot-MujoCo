# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# root = './dataset/leader_data'
# dataset = LeRobotDataset('Jeongeun/tutorial_v2',root = root )

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
repo_id_or_path = 'Jeongeun/tutorial_v2_pi05' # Use this for loading pretrained model from the hub
device = 'cuda'

# dataset_metadata = LeRobotDatasetMetadata("Jeongeun/tutorial_v2", root=ROOT)

policy = PI05Policy.from_pretrained(repo_id_or_path)
policy.to(device)