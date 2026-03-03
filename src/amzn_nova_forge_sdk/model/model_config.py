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
Data models for Nova Forge SDK.

This module contains dataclass definitions and constants used across the SDK.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypedDict

from amzn_nova_forge_sdk.model.model_enums import DeployPlatform, Model

REGION_TO_ESCROW_ACCOUNT_MAPPING = {
    "us-east-1": "708977205387",
    "us-west-2": "176779409107",
    "eu-west-2": "470633809225",
}

# Supported SageMaker Inference configurations per (Model, instance_type).
# Each entry is a list of (max_context_length, max_concurrency) tiers, sorted by context length.
# Source: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-model-sagemaker-inference.html
SUPPORTED_SMI_CONFIGS = {
    (Model.NOVA_MICRO, "ml.g5.12xlarge"): [(4000, 32), (8000, 16)],
    (Model.NOVA_MICRO, "ml.g5.24xlarge"): [(8000, 32)],
    (Model.NOVA_MICRO, "ml.g6.12xlarge"): [(4000, 32), (8000, 16)],
    (Model.NOVA_MICRO, "ml.g6.24xlarge"): [(8000, 32)],
    (Model.NOVA_MICRO, "ml.g6.48xlarge"): [(8000, 32)],
    (Model.NOVA_MICRO, "ml.p5.48xlarge"): [(8000, 32), (16000, 2), (24000, 1)],
    (Model.NOVA_LITE, "ml.g6.48xlarge"): [(4000, 32), (8000, 16)],
    (Model.NOVA_LITE, "ml.p5.48xlarge"): [(8000, 32), (16000, 2), (24000, 1)],
    (Model.NOVA_LITE_2, "ml.p5.48xlarge"): [(8000, 32), (16000, 2), (24000, 1)],
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

    @property
    def status(self):
        from amzn_nova_forge_sdk.util.bedrock import check_deployment_status

        return check_deployment_status(self.endpoint.uri, self.endpoint.platform)
