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

from amzn_nova_customization_sdk.model.model_enums import DeployPlatform
from amzn_nova_customization_sdk.util.bedrock import check_deployment_status

REGION_TO_ESCROW_ACCOUNT_MAPPING = {
    "us-east-1": "708977205387",
    "eu-west-2": "470633809225",
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
        return check_deployment_status(self.endpoint.uri, self.endpoint.platform)
