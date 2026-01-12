# Try your own policy!
This guide explains how to implement a custom policy in the LeRobot framework.
LeRobot is designed to make it easy to extend existing policies by cleanly separating:

- Configuration (architecture, optimizer, scheduler, normalization)

- Modeling (policy forward logic, inference, and loss computation)

By following this structure, a new policy can be plugged into the existing training and evaluation pipelines without modifying the core framework.

## 📁 File Structure

Each policy lives under src/policies/ and must contain two files:
a configuration file and a modeling file.
```
src
  |- policies
    |- baseline
       |- configuration.py
       |- modeling.py
       |- processor.py
    |[Your policy]
       |- configuration.py
       |- modeling.py
       |- processor.py
```

## 1️⃣ Configuration File

The configuration file defines how the policy is trained and used, including:

- temporal structure (observation and action horizons),
- normalization strategy,
- architecture hyperparameters,
- optimizer and learning-rate scheduler presets.

All policy configs must inherit from PreTrainedConfig and be registered so that LeRobot can automatically discover them.

### ✅ Basic Configuration Template

The configuration file defines how the policy is trained and executed, including:

temporal structure (observation/action horizons),

normalization strategy,

architecture hyperparameters,

optimizer and scheduler presets.

All policy configurations must inherit from PreTrainedConfig and register themselves so LeRobot can automatically discover them.

```python
# src/policies/your_policy_name/configuration.py

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamConfig, AdamWConfig, SGDConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    VQBeTSchedulerConfig,
    DiffuserSchedulerConfig,
)


@PreTrainedConfig.register_subclass("[YOUR POLICY]")
@dataclass
class YourConfig(PreTrainedConfig):
    # --------------------------------------------------
    # Temporal structure
    # --------------------------------------------------
    # Number of observation steps provided to the policy.
    n_obs_steps: int = 1

    # Number of future actions predicted per forward pass.
    chunk_size: int = 5

    # Number of actions actually executed before re-inference.
    n_action_steps: int = 5

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ENVIRONMENT_STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --------------------------------------------------
    # Architecture hyperparameters
    # --------------------------------------------------
    hidden_dim: int = 256
    num_layers: int = 4

    # --------------------------------------------------
    # Optimizer hyperparameters
    # --------------------------------------------------
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6

    def get_optimizer_preset(self):
        """
        Return an optimizer preset.
        Supported defaults: AdamConfig, AdamWConfig, SGDConfig.
        """
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        """
        Return a learning-rate scheduler preset.
        """
        return CosineDecayWithWarmupSchedulerConfig()

```
### ✅ Custom Optimizer / Scheduler (Optional)

LeRobot supports Adam, AdamW, and SGD by default.
If you want to implement a custom optimizer or scheduler, define a new config
and register it using the same policy name.

```python
from dataclasses import dataclass, asdict
import torch
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.optim.optimizers import OptimizerConfig

@LRSchedulerConfig.register_subclass("[YOUR POLICY]")
@dataclass
class MyOwnSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_training_steps: int

@OptimizerConfig.register_subclass("[YOUR POLICY]")
@dataclass
class MyOwnOptimizerConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def build(self, params):
        kwargs = asdict(self)
        return torch.optim.AdamW(params, **kwargs)


```

## 2️⃣ Modeling

The modeling file defines the actual policy behavior:

- input/output normalization,
- model architecture,
- training loss,
- inference-time action selection.

```python
# src/policies/your_policy_name/modeling.py

from collections import deque
import torch
from torch import nn, Tensor

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.constants import ACTION, OBS_STATE, OBS_ENV_STATE
from lerobot.common.preprocessing import Normalize, Unnormalize

from .configuration import YourConfig
'''
Make your Policy
'''
class YourPolicy(PreTrainedPolicy):
    config_class = YourConfig
    name = "[YOUR POLICY]"

    def __init__(
        self,
        config: YourConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    )
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Normalization of input and outputs
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        # Your code Starts Here
        self.model = YOURPOLICYMODEL()
        # Ends here

    def get_optim_params(self) -> dict:
        # return parameters for optimizer
        # You can change if needed
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad
                ]
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

     @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        batch = self.normalize_inputs(batch)
        if len(self._action_queue) == 0:
            # Inference
            actions = self.model(batch)[0][: self.config.n_action_steps, :]
            actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunks given environment observations.

        This method returns the full action chunk predicted by the model for the given batch of observations.
        """
        self.eval()
        if self.config.image_features:
            batch = dict(batch)  
        actions = self.model(batch)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
            batch = self.normalize_inputs(batch)
            """Run the batch through the model and compute the loss for training or validation."""
            batch = self.normalize_targets(batch)
            # Your Code Starts here
            actions = self.model(batch)
            loss = [YOUR LOSS FUNCTION]
            loss_dict = {"anything": loss}
            # Ends here
            return loss, loss_dict


class YOURPOLICYMODEL(nn.Module):
    def __init__(self, config: YourConfig):
        super().__init__()
        '''
        Build Model than returns action
        '''

    def get_optim_params(self) -> dict:
        # return parameters for optimizer
        # You can change if needed

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

     @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunks given environment observations.

        This method returns the full action chunk predicted by the model for the given batch of observations.
        """
        
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        '''
        Input: batch dict with keys OBS_IMAGES, OBS_STATE, OBS_ENV_STATE
        Output: actions [B, chunk_size, action_dim]
        '''
```

## 2️⃣ Preprocessor/Postprocessor
This is for normalizing input and outputs

```python
from typing import Any

import torch

from .configuration import BaselineConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_baseline_pre_post_processors(
    config: BaselineConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

```