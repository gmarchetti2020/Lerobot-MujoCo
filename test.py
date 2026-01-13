
from lerobot.policies.groot.modeling_groot import GrootPolicy
repo_id_or_path = 'ckpt/tutorial_v2_groot' #'Jeongeun/tutorial_v2_groot'
device = 'cuda'

# dataset_metadata = LeRobotDatasetMetadata("Jeongeun/tutorial_v2", root=ROOT)
policy = GrootPolicy.from_pretrained(repo_id_or_path)
policy.to(device)