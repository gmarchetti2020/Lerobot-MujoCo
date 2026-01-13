import argparse

parser = argparse.ArgumentParser(description='Download dataset from Lerobot Hub')
parser.add_argument('--type', type=str, default='dataset', choices=['pi0', 'pi05', 'groot', 'dataset'], required=True, help='Type of policy/dataset')
args = parser.parse_args()

if args.type == 'dataset':
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = './dataset/leader_data'
    dataset = LeRobotDataset('Jeongeun/tutorial_v2',root = root )
elif args.type == 'pi05':
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    repo_id_or_path = 'Jeongeun/tutorial_v2_pi05' 
    policy = PI05Policy.from_pretrained(repo_id_or_path)
elif args.type == 'pi0':
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    repo_id_or_path = 'Jeongeun/tutorial_v2_pi0' # Use this for loading pretrained model from the hub
    policy = PI0Policy.from_pretrained(repo_id_or_path)

elif args.type == 'groot':
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    repo_id_or_path = 'Jeongeun/tutorial_v2_groot' 
    policy = GrootPolicy.from_pretrained(repo_id_or_path)