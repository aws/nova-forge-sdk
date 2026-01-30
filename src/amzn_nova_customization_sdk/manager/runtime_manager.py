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
import json
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3
import sagemaker
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from amzn_nova_customization_sdk.recipe.recipe_config import HYPERPOD_RECIPE_PATH
from amzn_nova_customization_sdk.util.logging import logger


@dataclass
class JobConfig:
    job_name: str
    image_uri: str
    recipe_path: str
    output_s3_path: Optional[str] = None
    data_s3_path: Optional[str] = None
    input_s3_data_type: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None  # MLflow tracking server ARN
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    # TODO: The mlflow config is populated in recipe for both SMTJ and SMHP but will only work fro SMHP as SMTJ support for mlfow is only through boto3, fix this wit sagemaker 3 update


class RuntimeManager(ABC):
    def __init__(
        self, instance_type: str, instance_count: int, kms_key_id: Optional[str]
    ):
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._kms_key_id = kms_key_id

    @property
    def instance_type(self) -> str:
        """Type of instance (e.g., ml.p5.48xlarge)."""
        return self._instance_type

    @property
    def instance_count(self) -> int:
        """Number of instances used."""
        return self._instance_count

    # Needed to update the instance_count if user decides to override its value
    @instance_count.setter
    def instance_count(self, value: int) -> None:
        self._instance_count = value

    @property
    def kms_key_id(self) -> Optional[str]:
        """Optional KMS Key Id to use in S3 Bucket encryption, training jobs and deployments."""
        return self._kms_key_id

    @abstractmethod
    def setup(self) -> None:
        """Prepare environment and dependencies"""
        pass

    @abstractmethod
    def execute(self, job_config: JobConfig) -> str:
        """Launch a job and return a job id."""
        pass

    @abstractmethod
    def cleanup(self, job_id: str) -> None:
        """Tear down or release resources."""
        pass

    @classmethod
    def _s3_bucket_arn_from_path(cls, s3_path):
        """Extract S3 bucket ARN from a single S3 path."""
        if not s3_path:
            return None
        bucket = s3_path.split("/")[2]
        return f"arn:aws:s3:::{bucket}"

    @classmethod
    def _s3_object_arn_from_path(cls, s3_path):
        """Extract S3 object ARN from a single S3 path."""
        if not s3_path:
            return None
        bucket = s3_path.split("/")[2]
        # Allow access to the specific path and subdirectories
        if len(s3_path.split("/")) > 3:
            # Has a path component, use it
            path = "/".join(s3_path.split("/")[3:])
            return f"arn:aws:s3:::{bucket}/{path}*"
        else:
            # Just bucket, allow all objects
            return f"arn:aws:s3:::{bucket}/*"

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Base permissions required by all runtime managers."""
        permissions = []

        # Collect unique bucket ARNs
        bucket_arns = set()
        for s3_path in [data_s3_path, output_s3_path]:
            bucket_arn = cls._s3_bucket_arn_from_path(s3_path)
            if bucket_arn:
                bucket_arns.add(bucket_arn)

        # Add bucket-level permissions
        for bucket_arn in bucket_arns:
            permissions.extend(
                [
                    ("s3:CreateBucket", bucket_arn),
                    ("s3:ListBucket", bucket_arn),
                ]
            )

        # Add input-specific permissions (read-only)
        if data_s3_path:
            data_object_arn = cls._s3_object_arn_from_path(data_s3_path)
            permissions.append(("s3:GetObject", data_object_arn))

        # Add output-specific permissions (read-write)
        if output_s3_path:
            output_object_arn = cls._s3_object_arn_from_path(output_s3_path)
            permissions.extend(
                [
                    ("s3:GetObject", output_object_arn),
                    ("s3:PutObject", output_object_arn),
                ]
            )

        return permissions


class SMTJRuntimeManager(RuntimeManager):
    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        execution_role: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        encrypt_inter_container_traffic: bool = False,
        subnets: Optional[list[str]] = None,
        security_group_ids: Optional[list[str]] = None,
    ):
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role

        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic

        super().__init__(instance_type, instance_count, kms_key_id)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for SMTJ calling role operations and execution role validation."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(
            data_s3_path, output_s3_path
        )

        # Add SMTJ-specific permissions
        permissions.extend(
            [
                ("sagemaker:CreateTrainingJob", "*"),
                ("sagemaker:DescribeTrainingJob", "*"),
                "iam:GetRole",
                "iam:PassRole",
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
                "iam:ListRolePolicies",
                "iam:GetRolePolicy",
                "iam:ListAttachedRolePolicies",
            ]
        )

        return permissions

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = sagemaker.get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

    def execute(self, job_config: JobConfig) -> str:
        from amzn_nova_customization_sdk.validation.validator import Validator

        Validator.validate_job_name(job_name=job_config.job_name)

        try:
            assert job_config.output_s3_path is not None

            tensorboard_output = TensorBoardOutputConfig(
                s3_output_path=job_config.output_s3_path,
            )

            estimator_config = {
                "output_path": job_config.output_s3_path,
                "base_job_name": job_config.job_name,
                "role": self.execution_role,
                "instance_count": self.instance_count,
                "instance_type": self.instance_type,
                "training_recipe": job_config.recipe_path,
                "sagemaker_session": self.sagemaker_session,
                "image_uri": job_config.image_uri,
                "tensorboard_output_config": tensorboard_output,
                "disable_profiler": True,
                "debugger_hook_config": False,
                "encrypt_inter_container_traffic": self.encrypt_inter_container_traffic,
                "security_group_ids": self.security_group_ids,
                "subnets": self.subnets,
                "output_kms_key": self.kms_key_id,
            }

            estimator = PyTorch(**estimator_config)

            # For eval job, the input could be none
            # https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-notebook
            if job_config.data_s3_path:
                train_kwargs: Dict[str, Any] = {
                    "s3_data": job_config.data_s3_path,
                    "distribution": "FullyReplicated",
                }
                if job_config.input_s3_data_type is not None:
                    train_kwargs["s3_data_type"] = job_config.input_s3_data_type

                estimator.fit(
                    inputs={"train": TrainingInput(**train_kwargs)},
                    job_name=job_config.job_name,
                    wait=False,
                )
            else:
                estimator.fit(job_name=job_config.job_name, wait=False)

            return job_config.job_name

        except Exception as e:
            logger.error(f"Failed to start training job: {str(e)}")
            raise

    def cleanup(self, job_name: str) -> None:
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            self.sagemaker_client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_name}: {str(e)}")
            raise


# TODO: Might need to take RIG as input in case of multiple RIGs
class SMHPRuntimeManager(RuntimeManager):
    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        cluster_name: str,
        namespace: str,
        kms_key_id: Optional[str] = None,
    ):
        from amzn_nova_customization_sdk.validation.validator import Validator

        Validator.validate_cluster_name(cluster_name=cluster_name)
        Validator.validate_namespace(namespace=namespace)

        self.cluster_name = cluster_name
        self.namespace = namespace
        super().__init__(instance_type, instance_count, kms_key_id)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for HyperPod operations."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(
            data_s3_path, output_s3_path
        )

        # Add SMHP-specific permissions
        permissions.extend(
            [
                (
                    "sagemaker:DescribeCluster",
                    lambda infra: f"arn:aws:sagemaker:{infra.region}:*:cluster/{infra.cluster_name}",
                ),
                (
                    "eks:DescribeCluster",
                    lambda infra: f"arn:aws:eks:{infra.region}:*:cluster/*",
                ),
                (
                    "eks:ListAddons",
                    lambda infra: f"arn:aws:eks:{infra.region}:*:cluster/{infra.cluster_name}",
                ),
                ("sagemaker:ListClusters", "*"),
            ]
        )

        return permissions

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"

        response = subprocess.run(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                self.cluster_name,
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if response.stderr:
            logger.error(
                f"Unable to connect to HyperPod cluster {self.cluster_name}: {response.stderr}"
            )
            raise RuntimeError(response.stderr)

        logger.info(
            f"Successfully connected to HyperPod cluster '{self.cluster_name}' in namespace '{self.namespace}'."
        )

    def execute(self, job_config: JobConfig) -> str:
        try:
            # Scrub recipe path so that it will be recognized by the HyperPod CLI
            recipe_path = (
                job_config.recipe_path.split(HYPERPOD_RECIPE_PATH, 1)[1]
                .lstrip("/")
                .lstrip("\\")
                .removesuffix(".yaml")
            )

            override_parameters = json.dumps(
                {
                    "instance_type": self.instance_type,
                    "container": job_config.image_uri,
                }
            )
            response = subprocess.run(
                [
                    "hyperpod",
                    "start-job",
                    "--namespace",
                    self.namespace,
                    "--recipe",
                    recipe_path,
                    "--override-parameters",
                    override_parameters,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if matched_job_name := re.search(r"NAME: (\S+)", response.stdout):
                return matched_job_name.group(1)
            raise ValueError(
                f"Could not find job name in output. There may be an issue with the helm installation, "
                f"assumed role permissions to trigger jobs on the cluster, or job specification. Output: {response.stdout}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start HyperPod job: {e.stderr}")
            raise

    def cleanup(self, job_name: str) -> None:
        from amzn_nova_customization_sdk.validation.validator import Validator

        Validator.validate_job_name(job_name=job_name)

        try:
            response = subprocess.run(
                [
                    "hyperpod",
                    "cancel-job",
                    "--job-name",
                    job_name,
                    "--namespace",
                    self.namespace,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if response.stderr:
                logger.error(f"Failed to cleanup HyperPod job: {response.stderr}")

        except Exception as e:
            logger.error(f"Failed to cleanup HyperPod job '{job_name}': {str(e)}")
            raise
