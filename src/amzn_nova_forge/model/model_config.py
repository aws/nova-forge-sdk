# Copyright Amazon.com, Inc. or its affiliates

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

Re-exported from amzn_nova_forge.core for backward compatibility.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from amzn_nova_forge.core.constants import (
    _BEDROCK_STATUS_MAP,
    ESCROW_URI_TAG_KEY,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_SMI_CONFIGS,
    ModelStatus,
    _escrow_tag_value,
)
from amzn_nova_forge.core.enums import DeployPlatform, Model
from amzn_nova_forge.core.types import (
    DeploymentResult,
    EndpointInfo,
    ModelArtifacts,
    ModelConfigDict,
)
from amzn_nova_forge.util.logging import logger


@dataclass
class ModelDeployResult:
    """Result of creating a Bedrock or SageMaker model.

    Standalone dataclass (not a BaseJobResult subclass) because model creation
    is synchronous — no job lifecycle or status polling.

    Attributes:
        model_arn: The Bedrock custom model ARN or SageMaker model ARN.
        model_name: The model name.
        escrow_uri: S3 artifacts path used to create the model.
        created_at: UTC timestamp when the model was created.
    """

    model_arn: str
    model_name: str
    escrow_uri: str
    created_at: datetime
    _bedrock_client: Optional[Any] = field(default=None, repr=False)
    _sagemaker_client: Optional[Any] = field(default=None, repr=False)

    def _region(self) -> str:
        """Extract AWS region from model ARN."""
        parts = self.model_arn.split(":")
        if len(parts) >= 4:
            return parts[3]
        raise ValueError(f"Cannot extract region from ARN: {self.model_arn}")

    @property
    def status(self) -> ModelStatus:
        """Query current model status via live API call."""
        platform = self._platform(self.model_arn)
        if platform == "bedrock":
            if not self._bedrock_client:
                logger.warning(
                    "ModelDeployResult could not communicate with AWS to check model status. "
                    "Call from_arn() or provide a client to enable status checking."
                )
                return ModelStatus.UNKNOWN
            return self._bedrock_status()
        if platform == "sagemaker":
            if not self._sagemaker_client:
                logger.warning(
                    "ModelDeployResult could not communicate with AWS to check model status. "
                    "Call from_arn() or provide a client to enable status checking."
                )
                return ModelStatus.UNKNOWN
            return self._sagemaker_status()
        return ModelStatus.UNKNOWN

    def _bedrock_status(self) -> ModelStatus:
        assert self._bedrock_client is not None
        resp = self._bedrock_client.get_custom_model(modelIdentifier=self.model_arn)
        raw = resp.get("modelStatus", "")
        return _BEDROCK_STATUS_MAP.get(raw, ModelStatus.UNKNOWN)

    def _sagemaker_status(self) -> ModelStatus:
        assert self._sagemaker_client is not None
        try:
            model_name = self.model_arn.split("/")[-1]
            self._sagemaker_client.describe_model(ModelName=model_name)
            return ModelStatus.ACTIVE
        except ClientError:
            return ModelStatus.FAILED

    def _refresh_clients(self):
        """Create fresh boto3 clients based on ARN's region and platform."""
        try:
            region = self._region()
            platform = self._platform(self.model_arn)
        except ValueError:
            return
        try:
            if platform == "bedrock":
                self._bedrock_client = boto3.client("bedrock", region_name=region)
            elif platform == "sagemaker":
                self._sagemaker_client = boto3.client("sagemaker", region_name=region)
        except Exception as e:
            logger.debug("Could not create AWS client during load: %s", e)

    # Strict ARN patterns: arn:aws:<service>:<region>:<account>:<resource-type>/<resource>
    _BEDROCK_ARN_RE = re.compile(r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:custom-model/")
    _SAGEMAKER_ARN_RE = re.compile(r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:model/")

    @staticmethod
    def _platform(model_arn: str) -> Optional[str]:
        """Return 'bedrock' or 'sagemaker' based on strict ARN parsing, or None."""
        if ModelDeployResult._BEDROCK_ARN_RE.match(model_arn):
            return "bedrock"
        elif ModelDeployResult._SAGEMAKER_ARN_RE.match(model_arn):
            return "sagemaker"
        return None

    @property
    def platform(self) -> Optional[str]:
        """The platform this model was created on, or None if ARN is unrecognized."""
        return self._platform(self.model_arn)

    @classmethod
    def from_arn(
        cls, model_arn: str, bedrock_client=None, sagemaker_client=None
    ) -> "ModelDeployResult":
        """Reconstruct from an existing Bedrock or SageMaker model ARN.

        For Bedrock: calls GetCustomModel. escrow_uri recovered from tags if available.
        For SageMaker: calls DescribeModel. escrow_uri recovered from ListTags.
        """
        plat = cls._platform(model_arn)
        if plat == "sagemaker":
            return cls._from_sagemaker_arn(model_arn, sagemaker_client)
        elif plat == "bedrock":
            return cls._from_bedrock_arn(model_arn, bedrock_client)
        else:
            raise ValueError(
                f"Unrecognized ARN format: {model_arn}. "
                f"Expected a Bedrock ARN (arn:aws:bedrock:<region>:<account>:custom-model/...) "
                f"or SageMaker ARN (arn:aws:sagemaker:<region>:<account>:model/...)."
            )

    @classmethod
    def _from_bedrock_arn(cls, model_arn: str, bedrock_client=None) -> "ModelDeployResult":
        if bedrock_client is None:
            parts = model_arn.split(":")
            region = parts[3] if len(parts) >= 4 else None
            bedrock_client = boto3.client("bedrock", region_name=region)
        client = bedrock_client
        resp = client.get_custom_model(modelIdentifier=model_arn)

        creation_time = resp.get("creationTime")
        if isinstance(creation_time, str):
            created_at = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
        elif isinstance(creation_time, datetime):
            created_at = creation_time
        else:
            created_at = datetime.now(timezone.utc)

        # Best-effort: populate escrow_uri from tags
        escrow_uri = ""
        try:
            tags_resp = client.list_tags_for_resource(resourceARN=resp.get("modelArn", model_arn))
            for tag in tags_resp.get("tags", []):
                if tag.get("key") == ESCROW_URI_TAG_KEY:
                    escrow_uri = tag["value"]
                    break
        except Exception as e:
            logger.debug("Could not read tags for %s: %s", model_arn, e)

        result = cls(
            model_arn=resp.get("modelArn", model_arn),
            model_name=resp.get("modelName", ""),
            escrow_uri=escrow_uri,
            created_at=created_at,
        )
        result._bedrock_client = client
        return result

    @classmethod
    def _from_sagemaker_arn(cls, model_arn: str, sagemaker_client=None) -> "ModelDeployResult":
        if sagemaker_client is None:
            parts = model_arn.split(":")
            region = parts[3] if len(parts) >= 4 else None
            sagemaker_client = boto3.client("sagemaker", region_name=region)
        client = sagemaker_client
        model_name = model_arn.split("/")[-1]
        resp = client.describe_model(ModelName=model_name)

        creation_time = resp.get("CreationTime")
        if isinstance(creation_time, datetime):
            created_at = creation_time
        elif isinstance(creation_time, str):
            created_at = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
        else:
            created_at = datetime.now(timezone.utc)

        escrow_uri = ""
        try:
            tags_resp = client.list_tags(ResourceArn=model_arn)
            for tag in tags_resp.get("Tags", []):
                if tag.get("Key") == ESCROW_URI_TAG_KEY:
                    escrow_uri = tag["Value"]
                    break
        except Exception as e:
            logger.debug("Could not read tags for %s: %s", model_arn, e)

        result = cls(
            model_arn=model_arn,
            model_name=model_name,
            escrow_uri=escrow_uri,
            created_at=created_at,
        )
        result._sagemaker_client = client
        return result

    def _to_dict(self) -> Dict:
        return {
            "model_arn": self.model_arn,
            "model_name": self.model_name,
            "escrow_uri": self.escrow_uri,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def _from_dict(cls, data: Dict) -> "ModelDeployResult":
        return cls(
            model_arn=data["model_arn"],
            model_name=data["model_name"],
            escrow_uri=data.get("escrow_uri", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def dump(self, file_path: Optional[str] = None, file_name: Optional[str] = None) -> Path:
        """Save to JSON file."""
        file_name = file_name or f"{self.model_name}_deploy_result.json"
        full_path = Path(file_path) / file_name if file_path else Path(file_name)
        data = self._to_dict()
        data["__class_name__"] = "ModelDeployResult"
        with open(full_path, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Model deploy result saved to {full_path}")
        return full_path

    @classmethod
    def load(cls, file_path: str) -> "ModelDeployResult":
        """Load from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        data.pop("__class_name__", None)
        result = cls._from_dict(data)
        result._refresh_clients()
        return result
