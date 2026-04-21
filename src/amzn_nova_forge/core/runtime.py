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
"""Abstract RuntimeManager interface for Nova Forge SDK.

Defines the properties and abstract methods that every runtime manager
must implement.  Concrete methods (deploy_lambda, validate_lambda,
_s3_bucket_arn_from_path, required_calling_role_permissions) live in
the ``manager/`` subclass.

Rule: this module imports nothing outside ``core/``.
"""

from abc import ABC, abstractmethod
from typing import Optional

from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.core.types import JobConfig


class RuntimeManager(ABC):
    """Abstract base for all runtime managers.

    Provides the shared interface (properties + abstract methods) that
    ``util/``, service classes, and ``recipe/`` depend on for type
    annotations and runtime dispatch.
    """

    def __init__(
        self,
        instance_type: Optional[str],
        instance_count: Optional[int],
        kms_key_id: Optional[str],
        rft_lambda: Optional[str] = None,
    ):
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._kms_key_id = kms_key_id
        self._rft_lambda: Optional[str] = None
        self._rft_lambda_arn: Optional[str] = None
        # Use the property setter so subclass overrides are triggered
        self.rft_lambda = rft_lambda

    @property
    def rft_lambda(self) -> Optional[str]:
        """Lambda ARN or local .py file path set on this manager."""
        return self._rft_lambda

    @rft_lambda.setter
    def rft_lambda(self, value: Optional[str]) -> None:
        """Simple assignment — ``manager/`` subclass overrides to add ARN validation."""
        self._rft_lambda = value
        self._rft_lambda_arn = None

    @property
    def rft_lambda_arn(self) -> Optional[str]:
        """Resolved Lambda ARN. Set after deploy_lambda() or immediately if rft_lambda is an ARN."""
        return self._rft_lambda_arn

    @rft_lambda_arn.setter
    def rft_lambda_arn(self, value: Optional[str]) -> None:
        self._rft_lambda_arn = value

    @property
    def instance_type(self) -> Optional[str]:
        """Type of instance (e.g., ml.p5.48xlarge)."""
        return self._instance_type

    @property
    def instance_count(self) -> Optional[int]:
        """Number of instances used."""
        return self._instance_count

    @instance_count.setter
    def instance_count(self, value: Optional[int]) -> None:
        self._instance_count = value

    @property
    def kms_key_id(self) -> Optional[str]:
        """Optional KMS Key Id to use in S3 Bucket encryption, training jobs and deployments."""
        return self._kms_key_id

    @property
    def runtime_name(self) -> str:
        """Human-readable name for this runtime, used in filter operation logging."""
        return type(self).__name__

    @property
    def runtime_config(self) -> str:
        """Human-readable config summary for this runtime instance, used in operation logging."""
        return ""

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The execution platform for this runtime manager."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Prepare environment and dependencies."""
        pass

    @abstractmethod
    def execute(self, job_config: JobConfig) -> str:
        """Launch a job and return a job id."""
        pass

    @abstractmethod
    def cleanup(self, job_id: str) -> None:
        """Tear down or release resources."""
        pass

    # --- Pure utility classmethods (no external dependencies) ---

    @classmethod
    def _s3_bucket_arn_from_path(cls, s3_path):  # type: ignore[no-untyped-def]
        """Extract S3 bucket ARN from a single S3 path."""
        if not s3_path:
            return None
        bucket = s3_path.split("/")[2]
        return f"arn:aws:s3:::{bucket}"

    @classmethod
    def _s3_object_arn_from_path(cls, s3_path):  # type: ignore[no-untyped-def]
        """Extract S3 object ARN from a single S3 path."""
        if not s3_path:
            return None
        bucket = s3_path.split("/")[2]
        if len(s3_path.split("/")) > 3:
            path = "/".join(s3_path.split("/")[3:])
            return f"arn:aws:s3:::{bucket}/{path}*"
        else:
            return f"arn:aws:s3:::{bucket}/*"

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):  # type: ignore[no-untyped-def]
        """Base permissions required by all runtime managers."""
        permissions = []

        bucket_arns = set()
        for s3_path in [data_s3_path, output_s3_path]:
            bucket_arn = cls._s3_bucket_arn_from_path(s3_path)
            if bucket_arn:
                bucket_arns.add(bucket_arn)

        for bucket_arn in bucket_arns:
            permissions.extend(
                [
                    ("s3:CreateBucket", bucket_arn),
                    ("s3:ListBucket", bucket_arn),
                ]
            )

        if data_s3_path:
            data_object_arn = cls._s3_object_arn_from_path(data_s3_path)
            permissions.append(("s3:GetObject", data_object_arn))

        if output_s3_path:
            output_object_arn = cls._s3_object_arn_from_path(output_s3_path)
            permissions.extend(
                [
                    ("s3:GetObject", output_object_arn),
                    ("s3:PutObject", output_object_arn),
                ]
            )

        return permissions
