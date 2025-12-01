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
Configuration dataclasses for SFT (Supervised Fine-Tuning) methods on Nova 2.0.

This includes both SFT LoRA and SFT Full-rank training configurations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    to_primitive,
)


class PeftScheme(Enum):
    LORA = "lora"
    NULL = "null"


@dataclass
class LrScheduler:
    warmup_steps: int = 10
    min_lr: float = 1e-6


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class LoraTuningConfig:
    alpha: int = 64
    lora_plus_lr_ratio: float = 64.0

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class Peft:
    peft_scheme: PeftScheme
    lora_tuning: Optional[LoraTuningConfig] = None

    def __post_init__(self):
        if self.peft_scheme == PeftScheme.LORA and self.lora_tuning is None:
            self.lora_tuning = LoraTuningConfig()
        elif self.peft_scheme is None:
            self.peft_scheme = PeftScheme.NULL
            self.lora_tuning = None

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class SFTTrainingConfig:
    lr_scheduler: LrScheduler
    optim_config: OptimConfig
    peft: Peft
    max_steps: int = 100
    save_steps: int = 100
    save_top_k: int = 5
    max_length: int = 32768
    global_batch_size: int = 32
    reasoning_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class SFTRecipeConfig(BaseRecipeConfig):
    training_config: SFTTrainingConfig

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)
