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
Model and training method enumerations for Nova customization.

This module contains shared enums used across recipe building and validation.
"""

from enum import Enum, auto


class Platform(Enum):
    """Supported training platforms."""

    SMTJ = "SMTJ"
    SMHP = "SMHP"


class Version(Enum):
    """Supported Nova Versions (i.e. 1.0, 2.0, etc.)"""

    ONE = auto()
    TWO = auto()


class Model(Enum):
    """Supported Nova models."""

    version: Version
    model_type: str
    model_path: str

    def __new__(cls, value, version: Version, model_type: str, model_path: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.version = version
        obj.model_type = model_type
        obj.model_path = model_path
        return obj

    @classmethod
    def from_model_type(cls, model_type: str) -> "Model":
        for model in cls:
            if model.model_type == model_type:
                return model
        raise ValueError(f"Unknown model_type: {model_type}")

    NOVA_MICRO = (
        "nova_micro",
        Version.ONE,
        "amazon.nova-micro-v1:0:128k",
        "nova-micro/prod",
    )

    NOVA_LITE = (
        "nova_lite",
        Version.ONE,
        "amazon.nova-lite-v1:0:300k",
        "nova-lite/prod",
    )

    NOVA_LITE_2 = (
        "nova_lite_2",
        Version.TWO,
        "amazon.nova-2-lite-v1:0:256k",
        "nova-lite-2/prod",
    )

    NOVA_PRO = (
        "nova_pro",
        Version.ONE,
        "amazon.nova-pro-v1:0:300k",
        "nova-pro/prod",
    )


class TrainingMethod(Enum):
    """Supported training methods."""

    RFT_LORA = "rft_lora"
    RFT = "rft"
    SFT_LORA = "sft_lora"
    SFT_FULLRANK = "sft_fullrank"
    EVALUATION = "evaluation"


class DeployPlatform(Enum):
    """Supported deployment platforms."""

    BEDROCK_OD = "bedrock_od"
    BEDROCK_PT = "bedrock_pt"
