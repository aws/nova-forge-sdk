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
Shared data classes and type definitions for Nova Forge SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional, TypedDict

from amzn_nova_forge.core.constants import (
    DEFAULT_JOB_CACHE_DIR,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
)
from amzn_nova_forge.core.enums import DeployPlatform, Platform, TrainingMethod

if TYPE_CHECKING:
    from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor


@dataclass
class ForgeConfig:
    """Shared configuration for service classes.

    Holds optional settings that cut across training, evaluation, deployment,
    and inference workflows (e.g. KMS encryption, output paths, validation).

    ``mlflow_monitor`` is passed through to RecipeBuilder for experiment tracking.
    ``kms_key_id`` is used by ForgeDeployer for Bedrock model encryption (not
    sourced from RuntimeManager — ForgeDeployer does not use RuntimeManager).
    """

    kms_key_id: Optional[str] = None
    output_s3_path: Optional[str] = None
    generated_recipe_dir: Optional[str] = None
    validation_config: Optional[Dict[str, bool]] = None
    image_uri: Optional[str] = None
    mlflow_monitor: Optional[MLflowMonitor] = None
    enable_job_caching: bool = False
    job_cache_dir: str = DEFAULT_JOB_CACHE_DIR
    job_caching_config: Optional[Dict[str, Any]] = None


class ModelConfigDict(TypedDict):
    type: str
    path: str


@dataclass
class ModelArtifacts:
    checkpoint_s3_path: Optional[str]
    output_s3_path: str
    output_model_arn: Optional[str] = None  # Model package ARN for SMTJServerless jobs


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
    model_publish: Optional[Any] = None  # ModelDeployResult, Optional to avoid circular import

    @property
    def escrow_uri(self) -> Optional[str]:
        """Convenience: delegates to model_publish.escrow_uri."""
        return self.model_publish.escrow_uri if self.model_publish else None

    _status_checker: ClassVar[Optional[Callable]] = None

    @classmethod
    def _register_status_checker(cls, checker: Callable) -> None:
        """Register the function used to check deployment status.

        Called by util/bedrock.py at import time to wire up the status
        property without core/ needing to import util/.
        """
        cls._status_checker = checker

    @property
    def status(self):
        if DeploymentResult._status_checker is None:
            # Runtime fallback only — core/types.py does NOT import util.bedrock
            # at module load time.  This triggers registration if the caller
            # forgot to import util.bedrock before accessing .status.
            try:
                import amzn_nova_forge.util.bedrock  # noqa: F401
            except ImportError:
                pass
        if DeploymentResult._status_checker is None:
            raise RuntimeError(
                "Status checker not available. Ensure amzn_nova_forge.util.bedrock is imported."
            )
        return DeploymentResult._status_checker(self.endpoint.uri, self.endpoint.platform)


def validate_region(region: str) -> None:
    """Validate that the given AWS region is supported by Forge."""
    if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
        raise ValueError(
            f"Region '{region}' is not supported. "
            f"Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
        )


@dataclass
class JobConfig:
    job_name: str
    image_uri: str
    recipe_path: str
    output_s3_path: Optional[str] = None
    data_s3_path: Optional[str] = None
    input_s3_data_type: Optional[str] = None
    validation_data_s3_path: Optional[str] = None  # Validation data S3 path (for CPT and Bedrock)
    rft_lambda_arn: Optional[str] = None  # RFT Lambda ARN (for RFT jobs on Bedrock)
    mlflow_tracking_uri: Optional[str] = None  # MLflow tracking server ARN
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    method: Optional[TrainingMethod] = None  # Training method (required for Bedrock)
    # TODO: The mlflow config is populated in recipe for both SMTJ and SMHP but will only work for SMHP as SMTJ support for mlflow is only through boto3, fix this with sagemaker 3 update
