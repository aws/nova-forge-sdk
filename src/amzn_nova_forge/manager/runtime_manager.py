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
import io
import json
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import boto3
import yaml
from botocore.exceptions import ClientError, NoRegionError
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes import (
    OutputDataConfig,
    S3DataSource,
    TensorBoardOutputConfig,
)
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    Networking,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

from amzn_nova_forge.core.constants import (
    BYOD_AVAILABLE_EVAL_TASKS,
    HYPERPOD_RECIPE_PATH,
    SERVERLESS_CUSTOM_SCORER_EVAL_TASKS,
)
from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.runtime import RuntimeManager as RuntimeManagerBase
from amzn_nova_forge.core.types import JobConfig
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter
from amzn_nova_forge.util.bedrock import (
    get_customization_type,
    parse_bedrock_recipe_config,
    resolve_base_model_identifier,
)
from amzn_nova_forge.util.hub_util import get_hub_content
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.reward_verifier import verify_reward_function
from amzn_nova_forge.util.sagemaker import (
    extract_lambda_arn_from_hub_content,
    register_lambda_as_hub_content,
)
from amzn_nova_forge.validation.endpoint_validator import is_sagemaker_arn

# Maps TrainingMethod to (CustomizationTechnique, Peft|None) for ServerlessJobConfig
_METHOD_TO_SERVERLESS_CONFIG: Dict[TrainingMethod, tuple[str, Optional[str]]] = {
    TrainingMethod.SFT_LORA: ("SFT", "LORA"),
    TrainingMethod.SFT_FULL: ("SFT", None),
    TrainingMethod.DPO_LORA: ("DPO", "LORA"),
    TrainingMethod.DPO_FULL: ("DPO", None),
    TrainingMethod.RFT_LORA: ("RLVR", "LORA"),
    # TrainingMethod.RFT_FULL: ("RLVR", None),  # TODO: Add RLVR full support
}
DEFAULT_SMTJ_JOB_MAX_RUNTIME = 86400  # 1 day


def _is_hub_content_arn(arn: Optional[str]) -> bool:
    """Return True if the ARN is a SageMaker hub-content ARN."""
    from amzn_nova_forge.validation.validator import (
        is_hub_content_arn,  # lazy — avoids circular import at module level
    )

    return is_hub_content_arn(arn)


@dataclass
class DataPrepJobConfig(JobConfig):
    """Job configuration for data preparation pipelines.

    Extends JobConfig, mapping:
        - job_name -> Glue job name
        - data_s3_path -> input_path
        - output_s3_path -> output_path

    Additional data-prep-specific fields:
        - input_format / output_format: "parquet" or "jsonl"
        - text_field: Column name containing the text to process
        - extra_args: Additional kwargs forwarded to the pipeline builder.
            Operation-specific keys (e.g. ``pipeline_id``) are passed here.
    """

    input_format: str = "parquet"
    output_format: str = "parquet"
    text_field: str = "text"
    extra_args: Dict[str, Any] = field(default_factory=dict)


_account_id_cache: Optional[str] = None


def _get_caller_account_id(region: str = "us-east-1") -> str:
    """Return the AWS account ID of the caller, cached to avoid redundant STS calls.

    Only caches successful results — transient STS failures return "*" without poisoning
    the cache, so subsequent calls will retry.
    """
    global _account_id_cache
    if _account_id_cache is None:
        try:
            _account_id_cache = boto3.client("sts", region_name=region).get_caller_identity()[
                "Account"
            ]
        except Exception:
            logger.warning(
                "Failed to retrieve caller account ID via STS in region %s; "
                "falling back to wildcard '*'",
                region,
                exc_info=True,
            )
            return "*"
    return _account_id_cache


class RuntimeManager(RuntimeManagerBase):
    """Extends core ABC with ARN validation and concrete methods.

    The ``rft_lambda`` setter adds validation via a lazy import from
    ``validation.validator``.  Concrete helper methods (deploy_lambda,
    validate_lambda) also live here because they depend on modules
    outside ``core/``.
    """

    # Override the parent's rft_lambda setter to add ARN validation.
    # This replaces only the setter while inheriting the getter from
    # the base RuntimeManager.  Python requires referencing the parent
    # property explicitly — using @rft_lambda.setter here would fail
    # because rft_lambda isn't defined on this class.
    @RuntimeManagerBase.rft_lambda.setter  # type: ignore[attr-defined]
    def rft_lambda(self, value: Optional[str]) -> None:
        self._rft_lambda = value
        # Keep the resolved ARN in sync: set immediately when value is already an ARN,
        # clear it when switching to a file path or None so stale ARNs aren't reused.
        from amzn_nova_forge.validation.validator import (
            is_lambda_arn,  # avoid circular import
        )

        if value and is_lambda_arn(value):
            self._rft_lambda_arn = value
        elif value and _is_hub_content_arn(value):
            self._rft_lambda_arn = value
        else:
            self._rft_lambda_arn = None

    @_telemetry_emitter(Feature.INFRA, "deploy_lambda")
    def deploy_lambda(
        self,
        lambda_name: Optional[str] = None,
        execution_role_arn: Optional[str] = None,
    ) -> str:
        """
        Deploy the RFT reward lambda from a local Python file set on this manager.

        Uses self.rft_lambda as the source file. If self.rft_lambda is a Lambda ARN
        rather than a file path, raises ValueError — the lambda is already deployed.

        Args:
            lambda_name: Name for the Lambda function. Defaults to the source filename stem.
            execution_role_arn: IAM role ARN for the Lambda. If not provided, falls
                back to the runtime manager's execution_role attribute.

        Returns:
            The deployed Lambda function ARN.

        Raises:
            ValueError: If rft_lambda is not set, is already an ARN, file is not found,
                or if no execution role can be resolved.
        """
        if not self.rft_lambda:
            raise ValueError(
                "rft_lambda must be set on the runtime manager to a local .py file path before calling deploy_lambda()."
            )
        from amzn_nova_forge.validation.validator import (
            is_lambda_arn,
            validate_rft_lambda_name,  # to avoid circular import error
        )

        if is_lambda_arn(self.rft_lambda):
            raise ValueError(
                f"rft_lambda is already a deployed Lambda ARN ('{self.rft_lambda}'). "
                "deploy_lambda() deploys from a local .py file — there is nothing to deploy."
            )

        lambda_source = self.rft_lambda

        if lambda_name is None:
            lambda_name = os.path.splitext(os.path.basename(lambda_source))[0].replace("_", "-")

        platform = self.platform

        validate_rft_lambda_name(lambda_name, platform)

        role_arn = execution_role_arn or getattr(self, "execution_role", None)
        if not role_arn:
            raise ValueError(
                "An execution_role_arn must be provided to deploy_lambda(), or set as "
                "execution_role on the runtime manager."
            )

        if not os.path.isfile(lambda_source):
            raise ValueError(f"lambda_source file not found: {lambda_source}")

        # Package the .py file into a zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(lambda_source, arcname="lambda_function.py")
        zip_bytes = zip_buffer.getvalue()

        region = getattr(self, "region", None)
        if not region:
            logger.warning(
                "region is not set on the runtime manager; falling back to 'us-east-1'. "
                "This may deploy the Lambda to the wrong region."
            )
            region = "us-east-1"
        lambda_client = boto3.client("lambda", region_name=region)

        try:
            lambda_client.get_function(FunctionName=lambda_name)
            logger.info(f"Lambda '{lambda_name}' already exists — updating function code.")
            response = lambda_client.update_function_code(
                FunctionName=lambda_name,
                ZipFile=zip_bytes,
            )
            lambda_arn = response["FunctionArn"]
        except lambda_client.exceptions.ResourceNotFoundException:
            logger.info(f"Creating Lambda function '{lambda_name}'...")
            response = lambda_client.create_function(
                FunctionName=lambda_name,
                Runtime="python3.12",
                Role=role_arn,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Timeout=300,
            )
            lambda_arn = response["FunctionArn"]

        waiter = lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=lambda_name)
        logger.info(f"Lambda '{lambda_name}' is active. ARN: {lambda_arn}")
        self.rft_lambda_arn = lambda_arn

        return lambda_arn

    @_telemetry_emitter(Feature.INFRA, "validate_lambda")
    def validate_lambda(
        self,
        data_s3_path: str,
        validation_samples: int = 10,
    ) -> None:
        """
        Validate the RFT reward lambda with sample data.

        Resolves the lambda to validate from self.rft_lambda / self.rft_lambda_arn:
        - If self.rft_lambda is an ARN (or self.rft_lambda_arn is set), invokes the
          deployed lambda with samples from data_s3_path.
        - If self.rft_lambda is a local .py path, validates by executing lambda_handler
          directly without deploying.

        Args:
            data_s3_path: S3 path to the training dataset for pulling sample data.
            validation_samples: Number of samples to pull from data_s3_path (default: 10).

        Raises:
            ValueError: If rft_lambda is not set, or if validation fails.
        """
        lambda_arn = self.rft_lambda_arn

        # rft_lambda is a local file only if it's not any kind of ARN
        lambda_source = (
            self.rft_lambda
            if self.rft_lambda and not (self.rft_lambda.startswith("arn:"))
            else None
        )

        if not lambda_arn and not lambda_source:
            raise ValueError(
                "Either lambda_arn or lambda_source must be provided to validate_lambda()."
            )

        region = getattr(self, "region", None)
        if not region:
            logger.warning(
                "region is not set on the runtime manager; falling back to 'us-east-1'. "
                "This may validate the Lambda in the wrong region."
            )
            region = "us-east-1"
        platform = self.platform

        if lambda_arn:
            # Extract function name from ARN and validate naming requirements early
            function_name = lambda_arn.split(":")[-1]
            from amzn_nova_forge.validation.validator import (
                validate_rft_lambda_name,
                verify_rft_lambda,
            )

            validate_rft_lambda_name(function_name, platform)
            verify_rft_lambda(
                lambda_arn=lambda_arn,
                sample_count=validation_samples,
                data_s3_path=data_s3_path,
                region=region,
                platform=platform,
            )
        else:
            # Validate local file without deploying — executes lambda_handler directly
            logger.info(f"Validating local lambda source '{lambda_source}' without deployment...")
            sample_data = []
            if data_s3_path:
                from amzn_nova_forge.validation.validator import _parse_s3_uri

                s3_parts = _parse_s3_uri(data_s3_path)
                if not s3_parts:
                    raise ValueError(
                        f"Invalid S3 path: {data_s3_path}. Expected format: s3://bucket/key"
                    )
                bucket, key = s3_parts
                logger.info(f"Loading up to {validation_samples} sample(s) from {data_s3_path}")
                try:
                    s3_client = boto3.client("s3", region_name=region)
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    for i, line in enumerate(response["Body"].iter_lines()):
                        if i >= validation_samples:
                            break
                        try:
                            sample_data.append(json.loads(line.decode("utf-8")))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON on line {i + 1}: {e}")
                    logger.info(f"Loaded {len(sample_data)} sample(s) from S3")
                except Exception as e:
                    raise ValueError(f"Failed to read samples from {data_s3_path}: {e}") from e

            verify_reward_function(
                reward_function=lambda_source or "",
                sample_data=sample_data,
                validate_format=len(sample_data) > 0,
            )
            logger.info("Local lambda source validation passed.")


class SMTJRuntimeManager(RuntimeManager):
    """Runtime manager for SageMaker Training Jobs.

    Standard SMTJ training manager using the SageMaker SDK ``ModelTrainer``.

    Args:
        instance_type: EC2 instance type.
        instance_count: Number of instances.
        execution_role: IAM role ARN.  Auto-resolved when ``None``.
        kms_key_id: Optional KMS key for encryption.
        encrypt_inter_container_traffic: Encrypt traffic between containers.
        subnets: Optional VPC subnets.
        security_group_ids: Optional VPC security group IDs.
        max_job_runtime: Maximum runtime in seconds.
        rft_lambda: Optional RFT reward Lambda ARN or file path.
    """

    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        execution_role: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        encrypt_inter_container_traffic: bool = False,
        subnets: Optional[list[str]] = None,
        security_group_ids: Optional[list[str]] = None,
        max_job_runtime: Optional[int] = DEFAULT_SMTJ_JOB_MAX_RUNTIME,
        rft_lambda: Optional[str] = None,
    ):
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role

        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.max_job_runtime = max_job_runtime

        super().__init__(instance_type, instance_count, kms_key_id, rft_lambda)

        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for SMTJ calling role operations and execution role validation."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add SMTJ-specific permissions
        permissions.extend(
            [
                ("sagemaker:CreateTrainingJob", "*"),
                ("sagemaker:DescribeTrainingJob", "*"),
                ("sagemaker:StopTrainingJob", "*"),
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

    @property
    def platform(self) -> Platform:
        return Platform.SMTJ

    @property
    def runtime_name(self) -> str:
        return "SageMaker Training Job"

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

    def execute(self, job_config: JobConfig) -> str:
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_job_name(job_name=job_config.job_name)

        try:
            assert job_config.output_s3_path is not None

            tensorboard_output_config = TensorBoardOutputConfig(
                s3_output_path=job_config.output_s3_path,
            )

            compute = Compute(
                instance_count=self.instance_count,
                instance_type=self.instance_type,
            )
            output_data_config = OutputDataConfig(
                s3_output_path=job_config.output_s3_path,
                kms_key_id=self.kms_key_id,
            )
            networking = Networking(
                subnets=self.subnets,
                security_group_ids=self.security_group_ids,
            )
            stopping_condition = StoppingCondition(max_runtime_in_seconds=self.max_job_runtime)

            trainer_config = {
                "training_recipe": job_config.recipe_path,
                "compute": compute,
                "networking": networking,
                "stopping_condition": stopping_condition,
                "output_data_config": output_data_config,
                "base_job_name": job_config.job_name,
                "role": self.execution_role,
                "sagemaker_session": self.sagemaker_session,
                "training_image": job_config.image_uri,
            }
            # For eval job, the input could be none
            # https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-notebook
            if job_config.data_s3_path:
                input_data = InputData(
                    channel_name="train",
                    data_source=S3DataSource(
                        s3_uri=job_config.data_s3_path,
                        s3_data_type=job_config.input_s3_data_type,
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
                trainer_config["input_data_config"] = [input_data]

            model_trainer = ModelTrainer.from_recipe(
                **trainer_config
            ).with_tensorboard_output_config(tensorboard_output_config)
            model_trainer.train(wait=False, logs=False)
            # Added since the job name has an auto generated suffix by sagemaker v3
            sagemaker_client = boto3.client("sagemaker")
            list_jobs_response = sagemaker_client.list_training_jobs(
                NameContains=job_config.job_name,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=1,
            )
            unique_job_name = list_jobs_response["TrainingJobSummaries"][0]["TrainingJobName"]
            return unique_job_name

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
        rft_lambda: Optional[str] = None,
    ):
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_cluster_name(cluster_name=cluster_name)
        Validator.validate_namespace(namespace=namespace)

        self.cluster_name = cluster_name
        self.namespace = namespace
        self.execution_role = None
        super().__init__(instance_type, instance_count, kms_key_id, rft_lambda)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for HyperPod operations."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add SMHP-specific permissions
        permissions.extend(
            [
                (
                    "sagemaker:DescribeCluster",
                    lambda infra: (
                        f"arn:aws:sagemaker:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/{infra.cluster_name}"
                    ),
                ),
                (
                    "eks:DescribeCluster",
                    lambda infra: (
                        f"arn:aws:eks:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/*"
                    ),
                ),
                (
                    "eks:ListAddons",
                    lambda infra: (
                        f"arn:aws:eks:{infra.region}:{_get_caller_account_id(infra.region)}:cluster/{infra.cluster_name}"
                    ),
                ),
                ("sagemaker:ListClusters", "*"),
            ]
        )

        return permissions

    @property
    def platform(self) -> Platform:
        return Platform.SMHP

    @property
    def runtime_name(self) -> str:
        return "SageMaker HyperPod"

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
        from amzn_nova_forge.validation.validator import Validator

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

    def scale_cluster(
        self,
        instance_group_name: str,
        target_instance_count: int,
    ) -> Dict[str, Any]:
        """
        Scale a HyperPod cluster RIG up or down. The scaling is asynchronous.
        The cluster status will change to 'Updating' while scaling, and 'InService' when ready.

        Args:
            instance_group_name: Name of the instance group to scale (e.g., 'worker-group')
            target_instance_count: Desired number of instances for the group

        Returns:
            dict: Response containing:
                - ClusterArn: ARN of the updated cluster
                - InstanceGroupName: Name of the scaled instance group
                - InstanceType: Instance type being scaled
                - PreviousCount: Current instance count
                - TargetCount: Target instance count

        Raises:
            ValueError: If target_instance_count is negative
            ClientError: If scaling fails due to insufficient quota, capacity, or cluster issues.
        """
        if target_instance_count < 0:
            raise ValueError(
                f"target_instance_count must be non-negative, got {target_instance_count}"
            )

        sagemaker_client = boto3.client("sagemaker", region_name=self.region)

        # Get current cluster configuration
        try:
            describe_response = sagemaker_client.describe_cluster(ClusterName=self.cluster_name)
        except ClientError as e:
            logger.error(f"Failed to describe cluster '{self.cluster_name}': {e}")
            raise

        # Check if cluster is in InService state
        cluster_status = describe_response.get("ClusterStatus")
        if cluster_status != "InService":
            error_msg = (
                f"Cluster '{self.cluster_name}' is in '{cluster_status}' state. "
                f"Scaling is only allowed when cluster is in 'InService' state."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Find the Restricted Instance Group (RIG)
        restricted_instance_groups = describe_response.get("RestrictedInstanceGroups", [])
        target_group = None

        for group in restricted_instance_groups:
            if group["InstanceGroupName"] == instance_group_name:
                target_group = group
                break

        if not target_group:
            error_msg = (
                f"Instance group '{instance_group_name}' not found in cluster '{self.cluster_name}'. "
                f"Available groups: {[g['InstanceGroupName'] for g in restricted_instance_groups]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        instance_type = target_group["InstanceType"]
        current_count = target_group["CurrentCount"]

        logger.info(
            f"Scaling instance group '{instance_group_name}' "
            f"({instance_type}) from {current_count} to {target_instance_count} instances"
        )

        # Update the cluster with new instance count.
        try:
            # Optional fields to preserve from describe_cluster response
            optional_fields = [
                "ThreadsPerCore",
                "InstanceStorageConfigs",
                "OnStartDeepHealthChecks",
                "TrainingPlanArn",
                "OverrideVpcConfig",
                "ScheduledUpdateConfig",
                "EnvironmentConfig",
            ]

            all_groups = []
            for group in restricted_instance_groups:
                group_params = {
                    "InstanceCount": (
                        target_instance_count
                        if group["InstanceGroupName"] == instance_group_name
                        else group["CurrentCount"]
                    ),
                    "InstanceGroupName": group["InstanceGroupName"],
                    "InstanceType": group["InstanceType"],
                    "ExecutionRole": group["ExecutionRole"],
                }

                # Copy optional fields if present
                for field in optional_fields:
                    if field in group:
                        if field == "EnvironmentConfig":
                            env_config = {}
                            # Only pass FSxLustreConfig which is accepted by the UpdateCluster API
                            if "FSxLustreConfig" in group["EnvironmentConfig"]:
                                env_config["FSxLustreConfig"] = group["EnvironmentConfig"][
                                    "FSxLustreConfig"
                                ]
                            group_params[field] = env_config
                        else:
                            group_params[field] = group[field]

                all_groups.append(group_params)

            update_response = sagemaker_client.update_cluster(
                ClusterName=self.cluster_name,
                RestrictedInstanceGroups=all_groups,
            )

            logger.info(f"Successfully initiated scaling for cluster '{self.cluster_name}'. ")

            return {
                "ClusterArn": update_response["ClusterArn"],
                "InstanceGroupName": instance_group_name,
                "InstanceType": instance_type,
                "PreviousCount": current_count,
                "TargetCount": target_instance_count,
            }
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"Failed to scale cluster: {error_code} - {error_message}")
            raise

    def get_instance_groups(self) -> List[Dict[str, Any]]:
        """
        Get a list of available Restricted Instance Groups (RIGs) in the cluster.

        Returns:
            list: List of dicts containing instance group information:
                - InstanceGroupName: Name of the instance group
                - InstanceType: EC2 instance type (e.g., 'ml.p5.48xlarge')
                - CurrentCount: Current number of instances in the group

        Raises:
            ClientError: If unable to describe the cluster
        """
        sagemaker_client = boto3.client("sagemaker", region_name=self.region)

        # Get current cluster configuration
        try:
            describe_response = sagemaker_client.describe_cluster(ClusterName=self.cluster_name)
        except ClientError as e:
            logger.error(f"Failed to describe cluster '{self.cluster_name}': {e}")
            raise

        # Get the RIGs and extract necessary information
        instance_groups = describe_response.get("RestrictedInstanceGroups", [])

        rig_output = [
            {
                "InstanceGroupName": group["InstanceGroupName"],
                "InstanceType": group["InstanceType"],
                "CurrentCount": group["CurrentCount"],
            }
            for group in instance_groups
        ]

        # Log output to terminal
        logger.info(f"Found {len(rig_output)} instance group(s) in cluster '{self.cluster_name}':")
        for group in rig_output:
            logger.info(
                f"  - {group['InstanceGroupName']}: {group['InstanceType']} "
                f"(Current: {group['CurrentCount']} instances)"
            )

        return rig_output


class BedrockRuntimeManager(RuntimeManager):
    """
    Runtime manager for AWS Bedrock model customization jobs.
    """

    def __init__(
        self,
        execution_role: str,
        base_model_identifier: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        vpc_config: Optional[Dict[str, list[str]]] = None,
        rft_lambda: Optional[str] = None,
    ):
        # Store Bedrock-specific configuration
        self.execution_role = execution_role
        self.base_model_identifier = base_model_identifier
        self.vpc_config = vpc_config

        # Calls constructor with None for instance_type and instance_count
        # since Bedrock manages compute resources automatically
        super().__init__(
            instance_type=None,
            instance_count=None,
            kms_key_id=kms_key_id,
            rft_lambda=rft_lambda,
        )

        self.setup()

    @property
    def platform(self) -> Platform:
        return Platform.BEDROCK

    @property
    def runtime_name(self) -> str:
        return "Bedrock"

    def setup(self) -> None:
        """Initialize Bedrock client and session.

        Creates a boto3 session, extracts the region, and initializes the Bedrock client.
        Falls back to us-east-1 if no region is configured.

        Raises:
            Exception: If Bedrock client initialization fails
        """
        try:
            boto_session = boto3.session.Session()
            self.region = boto_session.region_name or "us-east-1"
            self.bedrock_client = boto3.client("bedrock", region_name=self.region)
            logger.info(f"Successfully initialized Bedrock client in region {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def execute(self, job_config: JobConfig) -> str:

        try:
            # Validate job name
            from amzn_nova_forge.validation.validator import Validator

            Validator.validate_job_name(job_name=job_config.job_name)
            logger.info(f"Starting Bedrock customization job: {job_config.job_name}")

            # Get training method from job_config (passed via nova_model_customizer)
            # For Bedrock, we need the method to determine customization type
            if job_config.method is None:
                raise ValueError(
                    "Training method must be provided in job_config for Bedrock jobs. "
                    "This should be set automatically by NovaModelCustomizer."
                )
            method = job_config.method

            # Extract hyperparameters from recipe if provided
            hyperparameters: Dict[str, str] = {}
            rft_hyperparameters: Dict[
                str, Any
            ] = {}  # RFT hyperparameters use native types (int, float, str)
            if job_config.recipe_path is not None:
                recipe_data = parse_bedrock_recipe_config(job_config.recipe_path, method)
                hyperparameters = recipe_data["hyperparameters"]
                rft_hyperparameters = recipe_data["rft_hyperparameters"]
            else:
                logger.info("No recipe provided - Bedrock will use default hyperparameters")

            # Get customization type from training method
            customization_type = get_customization_type(method)

            # Resolve base model identifier
            if job_config.recipe_path is not None:
                base_model_identifier = resolve_base_model_identifier(
                    job_config.recipe_path, self.base_model_identifier, self.region
                )
            elif self.base_model_identifier is not None:
                base_model_identifier = self.base_model_identifier
            else:
                raise ValueError(
                    "base_model_identifier must be provided either in BedrockRuntimeManager "
                    "constructor or via recipe_path. Cannot determine which model to use."
                )

            # Use job_name as custom_model_name
            custom_model_name = job_config.job_name

            # Build API request dictionary with required fields
            api_request: Dict[str, Any] = {
                "jobName": job_config.job_name,
                "customModelName": custom_model_name,
                "roleArn": self.execution_role,
                "baseModelIdentifier": base_model_identifier,
                "customizationType": customization_type,
            }

            # Add hyperParameters only if provided (optional parameter)
            if hyperparameters:
                api_request["hyperParameters"] = hyperparameters

            # Add trainingDataConfig (REQUIRED by Bedrock API)
            if job_config.data_s3_path is None:
                raise ValueError(
                    "data_s3_path is required for Bedrock customization jobs. "
                    "Bedrock API requires trainingDataConfig.s3Uri to be provided."
                )

            training_data_config: Dict[str, str] = {"s3Uri": job_config.data_s3_path}
            api_request["trainingDataConfig"] = training_data_config

            # Add outputDataConfig with output_s3_path (required)
            if job_config.output_s3_path is None:
                raise ValueError("output_s3_path is required for Bedrock customization jobs")

            output_data_config: Dict[str, str] = {"s3Uri": job_config.output_s3_path}

            # Add KMS key to outputDataConfig only if provided (optional)
            if self.kms_key_id is not None:
                output_data_config["kmsKeyId"] = self.kms_key_id

            api_request["outputDataConfig"] = output_data_config

            # Add validationDataConfig only if validation_data_s3_path is provided (optional)
            if job_config.validation_data_s3_path is not None:
                # Warn that Nova Lite 2 doesn't support validation datasets
                if "nova-2-lite" in base_model_identifier or "nova-lite-2" in base_model_identifier:
                    logger.warning(
                        "Validation datasets are not supported for Nova Lite 2 models on Bedrock. "
                        "The validation_data_s3_path parameter will be ignored. "
                        "To use validation datasets, please use a different model (Nova Micro, Nova Lite, or Nova Pro)."
                    )
                else:
                    validation_data_config: Dict[str, List[Dict[str, str]]] = {
                        "validators": [{"s3Uri": job_config.validation_data_s3_path}]
                    }
                    api_request["validationDataConfig"] = validation_data_config

            # Add vpcConfig only if provided (optional)
            if self.vpc_config is not None:
                vpc_config: Dict[str, List[str]] = {}
                if "subnet_ids" in self.vpc_config:
                    vpc_config["subnetIds"] = self.vpc_config["subnet_ids"]
                if "security_group_ids" in self.vpc_config:
                    vpc_config["securityGroupIds"] = self.vpc_config["security_group_ids"]

                if vpc_config:  # Only add if we have at least one field
                    api_request["vpcConfig"] = vpc_config

            # Add customizationConfig for RFT jobs only if Lambda ARN provided (optional but required for RFT)
            if customization_type == "REINFORCEMENT_FINE_TUNING":
                if not job_config.rft_lambda_arn:
                    raise ValueError(
                        "rft_lambda_arn is required for RFT (Reinforcement Fine-Tuning) jobs. "
                        "Please provide the Lambda ARN in train() method."
                    )

                rft_config: Dict[str, Any] = {
                    "graderConfig": {"lambdaGrader": {"lambdaArn": job_config.rft_lambda_arn}}
                }

                # Add RFT hyperparameters if provided (native types: int, float, string)
                if rft_hyperparameters:
                    rft_config["hyperParameters"] = rft_hyperparameters

                customization_config: Dict[str, Any] = {"rftConfig": rft_config}
                api_request["customizationConfig"] = customization_config

            # Call Bedrock API to create customization job
            response = self.bedrock_client.create_model_customization_job(**api_request)

            # Extract and return job ARN
            job_arn = response["jobArn"]

            return job_arn

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Bedrock API error ({error_code}): {error_message}")

            if error_code == "ValidationException":
                raise ValueError(f"Invalid Bedrock job configuration: {error_message}") from e
            elif error_code == "AccessDeniedException":
                raise PermissionError(f"Bedrock access denied: {error_message}") from e
            elif error_code == "ResourceNotFoundException":
                raise ValueError(f"Bedrock resource not found: {error_message}") from e
            elif error_code == "ServiceQuotaExceededException":
                raise RuntimeError(f"Bedrock service quota exceeded: {error_message}") from e
            elif error_code == "ThrottlingException":
                raise RuntimeError(f"Bedrock API throttled: {error_message}") from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_message}") from e

        except ValueError as e:
            # Re-raise ValueError (from validation, unsupported methods, etc.)
            logger.error(f"Validation error: {e}")
            raise

        except FileNotFoundError as e:
            # Re-raise FileNotFoundError (from recipe parsing)
            logger.error(f"Recipe file not found: {e}")
            raise

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error creating Bedrock customization job: {e}")
            raise RuntimeError(f"Failed to create Bedrock customization job: {e}") from e

    def cleanup(self, job_id: str) -> None:

        try:
            logger.info(f"Stopping Bedrock customization job: {job_id}")

            # Validate job_id format (should be a Bedrock job ARN)
            if not job_id or not isinstance(job_id, str):
                raise ValueError(f"Invalid job_id: must be a non-empty string, got {type(job_id)}")

            if not job_id.startswith("arn:aws:bedrock:"):
                logger.warning(
                    f"job_id does not appear to be a valid Bedrock ARN: {job_id}. "
                    "Expected format: arn:aws:bedrock:region:account:model-customization-job/job-id"
                )

            # Call Bedrock API to stop the customization job
            self.bedrock_client.stop_model_customization_job(jobIdentifier=job_id)
            logger.info(f"Successfully stopped Bedrock customization job: {job_id}")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Bedrock API error while stopping job ({error_code}): {error_message}")

            if error_code == "ValidationException":
                raise ValueError(f"Invalid job ARN: {error_message}") from e
            elif error_code == "ResourceNotFoundException":
                raise ValueError(f"Job not found: {error_message}") from e
            elif error_code == "AccessDeniedException":
                raise PermissionError(
                    f"Insufficient permissions to stop job: {error_message}"
                ) from e
            elif error_code == "ConflictException":
                # Job may already be stopped or in a terminal state
                logger.warning(f"Job may already be stopped: {error_message}")
            else:
                raise RuntimeError(f"Failed to stop Bedrock job: {error_message}") from e

        except Exception as e:
            logger.error(f"Unexpected error stopping Bedrock job: {e}")
            raise RuntimeError(f"Failed to stop Bedrock customization job: {e}") from e

        finally:
            # Close bedrock_client connection
            try:
                if hasattr(self, "bedrock_client") and self.bedrock_client is not None:
                    self.bedrock_client.close()
                    logger.info("Closed Bedrock client connection")
            except Exception as e:
                logger.warning(f"Error closing Bedrock client: {e}")

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Return required IAM permissions for Bedrock operations."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

        # Add Bedrock-specific permissions
        permissions.extend(
            [
                ("bedrock:CreateModelCustomizationJob", "*"),
                ("bedrock:StopModelCustomizationJob", "*"),
                ("bedrock:GetModelCustomizationJob", "*"),
                "iam:PassRole",
            ]
        )

        return permissions


class SMTJServerlessRuntimeManager(RuntimeManager):
    def __init__(
        self,
        model_package_group_name: str,
        execution_role: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        encrypt_inter_container_traffic: bool = False,
        subnets: Optional[list[str]] = None,
        security_group_ids: Optional[list[str]] = None,
        max_job_runtime: Optional[int] = DEFAULT_SMTJ_JOB_MAX_RUNTIME,  # 1 day
        rft_lambda: Optional[str] = None,
        evaluator_name: Optional[str] = None,
    ):
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role
        self.model_package_group_name = model_package_group_name
        self.evaluator_name = evaluator_name
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.max_job_runtime = max_job_runtime

        super().__init__(None, None, kms_key_id=kms_key_id, rft_lambda=rft_lambda)
        self.setup()

    @classmethod
    def required_calling_role_permissions(cls, data_s3_path=None, output_s3_path=None):
        """Required permissions for SMTJ calling role operations and execution role validation."""
        # Start with base S3 permissions
        permissions = super().required_calling_role_permissions(data_s3_path, output_s3_path)

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

    @property
    def platform(self) -> Platform:
        return Platform.SMTJServerless

    @property
    def runtime_name(self) -> str:
        return "SageMaker Serverless"

    def setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

        # Create model package group if it doesn't exist
        try:
            resp = self.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
            self.model_package_group_arn = resp["ModelPackageGroupArn"]
        except self.sagemaker_client.exceptions.ClientError:
            resp = self.sagemaker_client.create_model_package_group(
                ModelPackageGroupName=self.model_package_group_name
            )
            self.model_package_group_arn = resp["ModelPackageGroupArn"]

    def _resolve_base_model_arn(self, model: Model) -> str:
        """Resolve the BaseModelArn from SageMaker Hub for the given model."""
        hub_content = get_hub_content(
            hub_name="SageMakerPublicHub",
            hub_content_name=model.hub_content_name,
            hub_content_type="Model",
            region=self.region,
        )
        return hub_content["HubContentArn"]

    def _register_lambda_as_hub_content(self, lambda_arn: str) -> str:
        """Register a Lambda ARN as a JsonDoc hub-content and return the hub-content ARN."""
        hub_name = re.sub(r"[^a-zA-Z0-9-]", "-", self.model_package_group_name)[:63]
        return register_lambda_as_hub_content(
            lambda_arn=lambda_arn,
            hub_name=hub_name,
            sagemaker_client=self.sagemaker_client,
            evaluator_name=self.evaluator_name,
        )

    @_telemetry_emitter(Feature.INFRA, "validate_lambda")
    def validate_lambda(
        self,
        data_s3_path: str,
        validation_samples: int = 10,
    ) -> None:
        """Override to support hub-content ARNs by extracting the Lambda ARN first."""
        lambda_arn = self.rft_lambda_arn
        if lambda_arn and _is_hub_content_arn(lambda_arn):
            extracted = extract_lambda_arn_from_hub_content(lambda_arn, self.sagemaker_client)
            if not extracted:
                logger.warning(
                    f"Hub-content document does not contain a valid Lambda ARN. "
                    "Skipping validation."
                )
                return
            logger.info(f"Extracted Lambda ARN from hub-content: {extracted}")
            # Temporarily set rft_lambda_arn to the extracted Lambda ARN for validation
            original_arn = self._rft_lambda_arn
            self._rft_lambda_arn = extracted
            try:
                super().validate_lambda(data_s3_path, validation_samples)
            finally:
                self._rft_lambda_arn = original_arn
        else:
            super().validate_lambda(data_s3_path, validation_samples)

    def _extract_hyperparameters(
        self, recipe: Dict[str, Any], method: Optional[TrainingMethod] = None
    ) -> Dict[str, str]:
        """Extract hyperparameters from recipe as string key-value pairs.

        Args:
            recipe: The rendered recipe dict.
            method: Training method — determines the HyperParameter key for max_length.
                    RFT uses "max_length"; all others use "max_context_length".
        """
        is_rft = method in (TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL)
        max_length_hp_key = "max_length" if is_rft else "max_context_length"
        # Maps recipe key → HyperParameter key sent to the serverless API.
        hyperparameter_map = {
            # --- Shared across all models/methods ---
            "global_batch_size": "global_batch_size",
            "warmup_steps": "warmup_steps",
            "reasoning_effort": "reasoning_effort",
            "top_logprobs": "top_logprobs",
            "name": "name",
            # --- Learning rate: all recipes use "lr" as the recipe key ---
            "lr": "learning_rate",
            # --- LoRA ratio: v1 uses "loraplus_lr_ratio", v2 uses "lora_plus_lr_ratio" ---
            "loraplus_lr_ratio": "learning_rate_ratio",  # v1 (MICRO, LITE, PRO) recipe key
            "lora_plus_lr_ratio": "learning_rate_ratio",  # v2 (LITE_2) SFT/RFT recipe key
            # --- LoRA alpha: all recipes use "alpha" as the recipe key ---
            "alpha": "lora_alpha",
            # --- Sequence length: recipe key is "max_length" for all methods.
            #     HyperParameter key differs: RFT uses "max_length", SFT/DPO use "max_context_length" ---
            "max_length": max_length_hp_key,
            # --- Training duration: v1 uses "max_epochs", v2 uses "max_steps" ---
            "max_epochs": "max_epochs",  # v1 SFT/DPO
            "max_steps": "max_steps",  # v2 SFT/RFT
            # --- v2 SFT only ---
            "reasoning_enabled": "reasoning_enabled",
            # --- DPO only: recipe key is "beta" (under dpo_cfg.beta) ---
            "beta": "adam_beta",
        }

        result: Dict[str, str] = {}

        def _extract(obj: Dict[str, Any]) -> None:
            for k, v in obj.items():
                if k in hyperparameter_map and v is not None:
                    result[hyperparameter_map[k]] = str(v)
                elif isinstance(v, dict):
                    _extract(v)

        _extract(recipe)
        logger.info(f"HyperParameters used for the job: {result}")
        return result

    def _build_serverless_job_config(
        self,
        method: TrainingMethod,
        base_model_arn: str,
        eval_task: Optional[str] = None,
        evaluator_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build ServerlessJobConfig for create_training_job."""
        logger.warning("Accepting End-User License Agreement by using SMTJ Serverless")
        config: Dict[str, Any] = {
            "BaseModelArn": base_model_arn,
            "AcceptEula": True,
        }
        if method == TrainingMethod.EVALUATION:
            config["JobType"] = "Evaluation"
            # Map eval task to API EvaluationType enum
            # Mapping based on serverless API contract:
            # BenchmarkEvaluation: built-in benchmarks + llm_judge/rubric_llm_judge
            if eval_task in SERVERLESS_CUSTOM_SCORER_EVAL_TASKS:
                config["EvaluationType"] = "CustomScorerEvaluation"
            else:
                config["EvaluationType"] = "BenchmarkEvaluation"
            # EvaluatorArn is NOT used for eval — lambda goes in HyperParameters instead
        else:
            if method not in _METHOD_TO_SERVERLESS_CONFIG:
                raise ValueError(
                    f"{method.value} is not supported on SMTJServerless. "
                    + (
                        "Use RFT_LORA instead."
                        if method == TrainingMethod.RFT_FULL
                        else f"Supported methods: {list(_METHOD_TO_SERVERLESS_CONFIG.keys())}"
                    )
                )
            technique, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            config["JobType"] = "FineTuning"
            config["CustomizationTechnique"] = technique
            if peft:
                config["Peft"] = peft
            # EvaluatorArn: reward function hub-content ARN for RLVR training.
            # Lambda ARNs are passed via HyperParameters (reward_lambda_arn) instead.
            if _is_hub_content_arn(evaluator_arn):
                config["EvaluatorArn"] = evaluator_arn

        return config

    def execute(self, job_config: JobConfig) -> str:
        from amzn_nova_forge.validation.validator import Validator

        Validator.validate_job_name(job_name=job_config.job_name)

        try:
            assert job_config.output_s3_path is not None
            assert job_config.method is not None

            with open(job_config.recipe_path) as f:
                recipe = yaml.safe_load(f)

            model_type = recipe["run"]["model_type"]
            model = Model.from_model_type(model_type)

            # Resolve base model ARN from SageMaker Hub
            base_model_arn = self._resolve_base_model_arn(model)

            # Resolve the reward ARN only for RFT and eval jobs — not SFT/DPO.
            # For training (RLVR): auto-register Lambda ARN as hub-content for EvaluatorArn.
            # For eval: Lambda ARN goes directly into HyperParameters — no hub-content needed.
            _is_rft_or_eval = job_config.method in (
                TrainingMethod.RFT_LORA,
                TrainingMethod.RFT_FULL,
                TrainingMethod.EVALUATION,
            )
            evaluator_arn = (
                (
                    recipe.get("rl_env", {}).get("reward_lambda_arn")
                    or recipe.get("run", {}).get("reward_lambda_arn")
                    or self.rft_lambda_arn
                )
                if _is_rft_or_eval
                else None
            )
            from amzn_nova_forge.validation.validator import is_lambda_arn

            if (
                evaluator_arn
                and is_lambda_arn(evaluator_arn)
                and job_config.method != TrainingMethod.EVALUATION
            ):
                evaluator_arn = self._register_lambda_as_hub_content(evaluator_arn)

            hyperparams = self._extract_hyperparameters(recipe, method=job_config.method)
            # For eval jobs with a reward lambda, the API requires lambda_arn + lambda_type
            # in HyperParameters. EvaluatorArn is NOT used for eval jobs.
            if job_config.method == TrainingMethod.EVALUATION and evaluator_arn:
                # Resolve the actual Lambda ARN — evaluator_arn may be a hub-content ARN
                lambda_arn_for_params = evaluator_arn
                if not is_lambda_arn(evaluator_arn):
                    extracted = extract_lambda_arn_from_hub_content(
                        evaluator_arn, self.sagemaker_client
                    )
                    if extracted:
                        lambda_arn_for_params = extracted
                    else:
                        logger.warning(
                            "Could not extract Lambda ARN from hub-content for eval. "
                            "Skipping lambda_arn in HyperParameters."
                        )
                        lambda_arn_for_params = None
                if lambda_arn_for_params:
                    hyperparams["lambda_arn"] = lambda_arn_for_params
                    hyperparams["lambda_type"] = "rft"

            # Build create_training_job request
            create_params: Dict[str, Any] = {
                "TrainingJobName": job_config.job_name,
                "RoleArn": self.execution_role,
                "OutputDataConfig": {
                    "S3OutputPath": job_config.output_s3_path,
                    "CompressionType": "NONE",
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": self.max_job_runtime,
                },
                "HyperParameters": hyperparams,
                "ServerlessJobConfig": self._build_serverless_job_config(
                    job_config.method,
                    base_model_arn,
                    eval_task=recipe.get("evaluation", {}).get("task"),
                    evaluator_arn=evaluator_arn,
                ),
                "ModelPackageConfig": {
                    "ModelPackageGroupArn": self.model_package_group_arn,
                    # model_name_or_path is a model package ARN for iterative training
                    **(
                        {"SourceModelPackageArn": recipe["run"]["model_name_or_path"]}
                        if is_sagemaker_arn(recipe.get("run", {}).get("model_name_or_path", ""))
                        else {}
                    ),
                },
            }

            if self.kms_key_id:
                create_params["OutputDataConfig"]["KmsKeyId"] = self.kms_key_id

            # Input data via DataSet
            # Tasks in BYOD_AVAILABLE_EVAL_TASKS require input data; built-in benchmarks don't
            is_no_input_eval = (
                job_config.method == TrainingMethod.EVALUATION
                and recipe.get("evaluation", {}).get("task") not in BYOD_AVAILABLE_EVAL_TASKS
            )
            if job_config.data_s3_path and not is_no_input_eval:
                from sagemaker.ai_registry.dataset import (
                    DataSet,  # Import DataSet at call site since it crashes without AWS_DEFAULT_REGION
                )

                dataset_name = f"{job_config.job_name[:51]}-train-input"
                input_dataset = DataSet.create(dataset_name, job_config.data_s3_path)
                create_params["InputDataConfig"] = [
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "DatasetSource": {"DatasetArn": input_dataset.arn},
                        },
                        "CompressionType": "None",
                        "RecordWrapperType": "None",
                    }
                ]
            if self.subnets or self.security_group_ids:
                create_params["VpcConfig"] = {
                    "SecurityGroupIds": self.security_group_ids or [],
                    "Subnets": self.subnets or [],
                }
            # MLflow config from recipe
            run_section = recipe.get("run", {})
            mlflow_uri = run_section.get("mlflow_tracking_uri")
            if mlflow_uri:
                create_params["MlflowConfig"] = {
                    "MlflowResourceArn": mlflow_uri,
                    "MlflowExperimentName": run_section.get("mlflow_experiment_name", ""),
                    "MlflowRunName": run_section.get("mlflow_run_name", ""),
                }

            result = self.sagemaker_client.create_training_job(**create_params)
            job_arn = result["TrainingJobArn"]
            return job_arn.split("/")[-1]

        except Exception as e:
            if isinstance(e, NoRegionError):
                logger.error(
                    "Could not connect to SageMaker. Please set a region with `export AWS_DEFAULT_REGION=<your-region>`"
                )
            logger.error(f"Failed to start training job: {str(e)}")
            raise

    def cleanup(self, job_name: str) -> None:
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            self.sagemaker_client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_name}: {str(e)}")
            raise
