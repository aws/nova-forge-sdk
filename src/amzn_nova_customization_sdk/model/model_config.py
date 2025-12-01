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
Data models for Nova Customization SDK.

This module contains dataclass definitions and constants used across the SDK.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict

from amzn_nova_customization_sdk.model.model_enums import (
    DeployPlatform,
    Platform,
    TrainingMethod,
    Version,
)

REGION_TO_ESCROW_ACCOUNT_MAPPING = {
    "us-east-1": "708977205387",
    "eu-west-2": "470633809225",
}

IMAGE_REPO_REGISTRY = {
    TrainingMethod.RFT: "nova-fine-tune-repo",
    TrainingMethod.RFT_LORA: "nova-fine-tune-repo",
    TrainingMethod.SFT_LORA: "nova-fine-tune-repo",
    TrainingMethod.SFT_FULLRANK: "nova-fine-tune-repo",
    TrainingMethod.EVALUATION: "nova-evaluation-repo",
}

RUNTIME_PREFIX_REGISTRY = {
    "SMTJRuntimeManager": "SM-TJ-",
    "SMHPRuntimeManager": "SM-HP-",
}

METHOD_IMAGE_REGISTRY = {
    Platform.SMTJ: {
        Version.ONE: {
            TrainingMethod.EVALUATION: "Eval-latest",
            TrainingMethod.SFT_FULLRANK: "SFT-latest",
            TrainingMethod.SFT_LORA: "SFT-latest",
        },
        Version.TWO: {
            TrainingMethod.EVALUATION: "Eval-V2-latest",
            TrainingMethod.RFT: "RFT-V2-latest",
            TrainingMethod.RFT_LORA: "RFT-V2-latest",
            TrainingMethod.SFT_FULLRANK: "SFT-V2-latest",
            TrainingMethod.SFT_LORA: "SFT-V2-latest",
        },
    },
    Platform.SMHP: {
        Version.ONE: {
            TrainingMethod.EVALUATION: "Eval-latest",
            TrainingMethod.SFT_FULLRANK: "SFT-latest",
            TrainingMethod.SFT_LORA: "SFT-latest",
        },
        Version.TWO: {
            TrainingMethod.EVALUATION: "Eval-V2-latest",
            TrainingMethod.RFT: "RFT-TRAIN-V2-latest",
            TrainingMethod.RFT_LORA: "RFT-TRAIN-V2-latest",
            TrainingMethod.SFT_FULLRANK: "SFT-V2-latest",
            TrainingMethod.SFT_LORA: "SFT-V2-latest",
        },
    },
}


class ModelConfigDict(TypedDict):
    type: str
    path: str


@dataclass
class ModelArtifacts:
    checkpoint_s3_path: Optional[str]
    output_s3_path: str


@dataclass
class EndpointInfo:
    platform: DeployPlatform
    endpoint_name: str
    uri: str
    model_artifact_path: str


@dataclass
class DeploymentResult:
    endpoint: EndpointInfo
    created_at: datetime
