from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = './dataset/leader_data'
dataset = LeRobotDataset('Jeongeun/tutorial_v2',root = root )
# from lerobot.policies.groot.modeling_groot import GR00TN15
# from lerobot.processor import PolicyAction, PolicyProcessorPipeline
# from lerobot.processor.converters import (
#     batch_to_transition,
#     policy_action_to_transition,
#     transition_to_batch,
#     transition_to_policy_action,
# )
# from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# ROOT = './dataset/leader_data'
# repo_id_or_path = 'Jeongeun/tutorial_v2_groot'
# policy = GR00TN15.from_pretrained(repo_id_or_path)

# kwargs = {}
# preprocessor_overrides = {}
# postprocessor_overrides = {}
# preprocessor_overrides["groot_pack_inputs_v3"] = {
#     "stats": kwargs.get("dataset_stats"),
#     "normalize_min_max": True,
# }

# # Also ensure postprocessing slices to env action dim and unnormalizes with dataset stats
# env_action_dim = policy.config.output_features["action"].shape[0]
# postprocessor_overrides["groot_action_unpack_unnormalize_v1"] = {
#     "stats": kwargs.get("dataset_stats"),
#     "normalize_min_max": True,
#     "env_action_dim": env_action_dim,
# }
# kwargs["preprocessor_overrides"] = preprocessor_overrides
# kwargs["postprocessor_overrides"] = postprocessor_overrides


# preprocessor = PolicyProcessorPipeline.from_pretrained(
#     pretrained_model_name_or_path=repo_id_or_path,
#     config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
#     overrides=kwargs.get("preprocessor_overrides", {}),
#     to_transition=batch_to_transition,
#     to_output=transition_to_batch,
# ),

# postprocessor =  PolicyProcessorPipeline.from_pretrained(
#     pretrained_model_name_or_path=repo_id_or_path,
#     config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
#     overrides=kwargs.get("postprocessor_overrides", {}),
#     to_transition=policy_action_to_transition,
#     to_output=transition_to_policy_action,
# )