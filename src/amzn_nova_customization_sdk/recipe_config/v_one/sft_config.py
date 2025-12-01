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
Configuration dataclasses for SFT (Supervised Fine-Tuning) methods on Nova 1.0.

This includes both SFT LoRA and SFT Full-rank training configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    to_primitive,
)


class Name(Enum):
    DISTRIBUTED_FUSED_ADAM = "distributed_fused_adam"


class PeftScheme(Enum):
    LORA = "lora"
    NULL = None


@dataclass
class TrainerConfig:
    max_epochs: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class SchedConfig:
    min_lr: float
    warmup_steps: int = 10
    constant_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class OptimConfig:
    lr: float
    sched: SchedConfig
    eps: float = 1e-6
    name: Name = Name.DISTRIBUTED_FUSED_ADAM
    adam_w_mode: bool = True
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class LoraTuningConfig:
    alpha: int = 128
    loraplus_lr_ratio: float = 16.0
    adapter_dropout: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class PeftConfig:
    peft_scheme: PeftScheme
    lora_tuning: Optional[LoraTuningConfig] = None

    def __post_init__(self):
        if self.peft_scheme is None:
            self.peft_scheme = PeftScheme.NULL
            self.lora_tuning = None

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class ModelConfig:
    optim: OptimConfig
    peft: Optional[PeftConfig]
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class SFTTrainingConfig:
    max_length: int
    global_batch_size: int
    trainer: TrainerConfig
    model: ModelConfig

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class SFTRecipeConfig(BaseRecipeConfig):
    training_config: SFTTrainingConfig

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)
