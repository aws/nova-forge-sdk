# Copyright 2025 Amazon Inc

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Configuration dataclasses for RFT (Reinforcement Fine-Tuning) methods on Nova 2.0.

This includes both RFT LoRA and RFT Full-rank training configurations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
    to_primitive,
)


class ReasoningEffort(Enum):
    HIGH = "high"
    LOW = "low"


class PeftScheme(Enum):
    LORA = "lora"


class Type(Enum):
    OFF_POLICY_SYNC = "off_policy_async"


@dataclass
class ApiEndpoint:
    lambda_arn: str
    lambda_concurrency_limit: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Rewards:
    api_endpoint: ApiEndpoint

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Generator:
    max_new_tokens: int = 8192
    set_random_seed: bool = True
    temperature: int = 1
    top_k: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class AdvantageStrategy:
    number_generation: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class RolloutStrategy:
    type: Type = Type.OFF_POLICY_SYNC
    age_tolerance: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class OptimConfig:
    lr: float = 7e-7
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class LoraTuning:
    loraplus_lr_ratio: float = 64.0
    alpha: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Peft:
    peft_scheme: PeftScheme
    lora_tuning: Optional[LoraTuning] = None

    def __post_init__(self):
        if self.peft_scheme == PeftScheme.LORA and self.lora_tuning is None:
            self.lora_tuning = LoraTuning()

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Trainer:
    optim_config: OptimConfig
    peft: Optional[Peft]
    max_steps: int = 100
    save_steps: int = 100
    save_top_k: int = 5
    refit_freq: int = 4
    clip_ratio_high: float = 0.2
    ent_coeff: float = 0.001
    loss_scale: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Rollout:
    rollout_strategy: RolloutStrategy
    advantage_strategy: AdvantageStrategy
    generator: Generator
    rewards: Rewards

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Data:
    shuffle: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class RFTTrainingConfig:
    data: Data
    rollout: Rollout
    trainer: Trainer
    global_batch_size: int = 256
    max_length: int = 10240
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class RFTRunConfig(BaseRunConfig):
    reward_lambda_arn: str
    generation_replicas: int = 2
    rollout_worker_replicas: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class RFTRecipeConfig(BaseRecipeConfig):
    run: RFTRunConfig
    training_config: RFTTrainingConfig

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)
