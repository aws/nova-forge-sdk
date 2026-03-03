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
"""Base infrastructure classes and utilities for RFT Multiturn."""

import json
import os
import signal
import tarfile
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Dict, List, Optional

import boto3
import sagemaker

from amzn_nova_forge_sdk.util.logging import logger
from amzn_nova_forge_sdk.validation.rft_multiturn_validator import (
    validate_stack_name,
)

from .constants import (
    ECR_REPO_NAME,
    IAM_PROPAGATION_WAIT_TIME,
    RFT_EVAL_LOG,
    RFT_EXECUTION_ROLE_NAME,
    RFT_POLICY_NAME,
    RFT_SAM_LOG,
    RFT_TRAIN_LOG,
    STACK_NAME_SUFFIX,
    STARTER_KIT_S3,
)
from .utils import build_duplicate_job_error_message, validate_starter_kit_path


def _wait_for_iam_propagation(
    iam_client,
    role_name: str,
    policy_arn: str,
    timeout: int = IAM_PROPAGATION_WAIT_TIME,
) -> None:
    """

    Wait for IAM policy attachment to propagate

    """
    logger.info(f"Waiting for IAM policy propagation for role {role_name}...")
    wait_interval = 5
    elapsed = 0

    while elapsed < timeout:
        time.sleep(wait_interval)
        elapsed += wait_interval

        try:
            response = iam_client.list_attached_role_policies(RoleName=role_name)
            attached_policies = [p["PolicyArn"] for p in response["AttachedPolicies"]]

            if policy_arn in attached_policies:
                logger.info("IAM policy propagation completed successfully")
                return
        except Exception:
            pass

    logger.warning(
        f"IAM propagation check timed out after {timeout}s, proceeding anyway"
    )


class EnvType(Enum):
    """
    Environment types for training, evaluation, and SAM deployment
    """

    TRAIN = "train"
    EVAL = "eval"
    SAM = "sam"


LOG_FILES = {
    EnvType.TRAIN: RFT_TRAIN_LOG,
    EnvType.EVAL: RFT_EVAL_LOG,
    EnvType.SAM: RFT_SAM_LOG,
}


class VFEnvId(str, Enum):
    """
    Built-in verifier environment IDs
    """

    WORDLE = "wordle"
    TERMINAL_BENCH = "terminalbench_env"


def _load_rft_policies() -> dict:
    """
    Load RFT policies from JSON file.
    """
    policy_file = resources.files("amzn_nova_forge_sdk.rft_multiturn").joinpath(
        "rft_multiturn_policies.json"
    )
    with policy_file.open() as f:  # type: ignore
        return json.load(f)


def _build_combined_policy_document(
    region: str, account_id: str, custom_policy_path: Optional[str] = None
) -> dict:
    """
    Build combined RFT policy document from JSON file.

    Args:
        region: AWS region
        account_id: AWS account ID
        custom_policy_path: Optional path to custom policy JSON file. If provided, uses as-is (must have actual ARNs).

    Returns:
        Combined policy document as dict
    """

    # Load policies from custom file or SDK default
    if custom_policy_path:
        with open(custom_policy_path) as f:
            policies = json.load(f)
    else:
        policies = _load_rft_policies()

        # Only replace placeholders for SDK default policy
        replacements = {
            "CLOUDFORMATION_RESOURCE_PLACEHOLDER": [
                f"arn:aws:cloudformation:{region}:{account_id}:stack/*{STACK_NAME_SUFFIX}*",
                f"arn:aws:cloudformation:{region}:{account_id}:stack/aws-sam-cli-managed-*",
                f"arn:aws:cloudformation:{region}:aws:transform/Serverless-2016-10-31",
            ],
            "DYNAMODB_RESOURCE_PLACEHOLDER": f"arn:aws:dynamodb:{region}:{account_id}:table/*{STACK_NAME_SUFFIX}*",
            "IAM_RESOURCE_PLACEHOLDER": [
                f"arn:aws:iam::{account_id}:role/{RFT_POLICY_NAME}",
                f"arn:aws:iam::{account_id}:role/*SageMaker*",
                f"arn:aws:iam::{account_id}:role/*Nova*",
            ],
            "LAMBDA_RESOURCE_PLACEHOLDER": [
                f"arn:aws:lambda:{region}:{account_id}:function:*{STACK_NAME_SUFFIX}*",
                f"arn:aws:lambda:{region}:{account_id}:function:*{STACK_NAME_SUFFIX}-SageMaker-*",
                f"arn:aws:lambda:{region}:{account_id}:event-source-mapping:*",
            ],
            "SQS_RESOURCE_PLACEHOLDER": [
                f"arn:aws:sqs:{region}:{account_id}:*{STACK_NAME_SUFFIX}*",
                f"arn:aws:sqs:{region}:{account_id}:*{STACK_NAME_SUFFIX}-SageMaker-*",
            ],
            "LOGS_RESOURCE_PLACEHOLDER": [
                f"arn:aws:logs:{region}:{account_id}:log-group:/aws/lambda/*{STACK_NAME_SUFFIX}*",
                f"arn:aws:logs:{region}:{account_id}:log-group:/ecs/*{STACK_NAME_SUFFIX}*:*",
            ],
            "ECR_RESOURCE_PLACEHOLDER": f"arn:aws:ecr:{region}:{account_id}:repository/{ECR_REPO_NAME}",
            "ECS_RESOURCE_PLACEHOLDER": [
                f"arn:aws:ecs:{region}:{account_id}:cluster/*",
                f"arn:aws:ecs:{region}:{account_id}:task-definition/*{STACK_NAME_SUFFIX}*",
                f"arn:aws:ecs:{region}:{account_id}:task-definition/*{STACK_NAME_SUFFIX}*:*",
            ],
            "S3_RESOURCE_PLACEHOLDER": [
                "arn:aws:s3:::aws-sam-cli-managed-*",
                "arn:aws:s3:::aws-sam-cli-managed-*/*",
                "arn:aws:s3:::nova-rft-starter-kit-*",
                "arn:aws:s3:::nova-rft-starter-kit-*/*",
                f"arn:aws:s3:::sagemaker-{region}-{account_id}",
                f"arn:aws:s3:::sagemaker-{region}-{account_id}/*",
            ],
        }

        # Replace placeholders in SDK default policy
        for policy_key, policy in policies.items():
            if policy_key == "trust_policy":
                continue
            for statement in policy.get("Statement", []):
                if "Resource" in statement:
                    resource = statement["Resource"]
                    if isinstance(resource, str) and resource in replacements:
                        statement["Resource"] = replacements[resource]

    # Validate all resources
    def validate_resource(resource: str, policy_name: str):
        if not isinstance(resource, str):
            raise ValueError(
                f"Invalid resource in {policy_name}: must be string, got {type(resource)}"
            )
        if not (resource.startswith("arn:") or resource == "*"):
            raise ValueError(
                f"Invalid resource in {policy_name}: '{resource}' must be ARN or '*'"
            )

    for policy_key, policy in policies.items():
        if policy_key == "trust_policy":
            continue
        for statement in policy.get("Statement", []):
            if "Resource" in statement:
                resources_to_validate = (
                    statement["Resource"]
                    if isinstance(statement["Resource"], list)
                    else [statement["Resource"]]
                )
                for r in resources_to_validate:
                    validate_resource(r, policy_key)

    # Combine all statements
    combined_statements = []
    for policy_key in [
        "cloudformation_policy",
        "dynamodb_policy",
        "iam_policy",
        "lambda_policy",
        "sqs_policy",
        "logs_policy",
        "ecr_policy",
        "ecs_policy",
        "s3_policy",
    ]:
        combined_statements.extend(policies[policy_key]["Statement"])

    return {"Version": "2012-10-17", "Statement": combined_statements}


def create_rft_execution_role(
    region: str = "us-east-1",
    role_name: Optional[str] = None,
    custom_policy_path: Optional[str] = None,
) -> str:
    """
    Creates a new IAM Role for RFT Multiturn infrastructure access.

    Args:
        region: AWS region for the RFT infrastructure (default: us-east-1).
        role_name: Optional name of the role to create. Defaults to RFT_EXECUTION_ROLE_NAME.
        custom_policy_path: Optional path to custom policy JSON file. If not provided, uses SDK default.

    Returns:
        str: The ARN of the created/existing role

    Raises:
        Exception: If it fails at creating the new role or attaching policies.
    """

    iam_client = boto3.client("iam")
    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]
    role_name = role_name or RFT_EXECUTION_ROLE_NAME

    # Load trust policy from JSON file
    policies = _load_rft_policies()

    # Create the execution role
    try:
        # Check if the role exists already
        rft_execution_role = iam_client.get_role(RoleName=role_name)
        logger.info(
            f"The {role_name} role already exists,"
            f" if you want to overwrite with new permissions delete the old {role_name}Policy policy"
        )

    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        rft_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for RFT Multiturn infrastructure access.",
        )
        logger.info(f"Created role {role_name}")
    except Exception as e:
        raise Exception(
            f"Failed to create the RFT execution role {role_name}: {str(e)}"
        )

    # Get combined policy document
    policy_document = _build_combined_policy_document(
        region, account_id, custom_policy_path
    )
    policy_name = f"{role_name}Policy"
    policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"

    # Create and attach the combined policy
    try:
        iam_client.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document),
            Description="Combined RFT Multiturn infrastructure policy",
        )
        logger.info(f"Created policy {policy_name}")
    except iam_client.exceptions.EntityAlreadyExistsException:
        logger.info(f"Policy {policy_name} already exists")

    # Check if already attached
    try:
        attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
        if not any(
            p["PolicyArn"] == policy_arn for p in attached_policies["AttachedPolicies"]
        ):
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            logger.info(f"Attached policy {policy_name} to role {role_name}")
        else:
            logger.info(f"Policy {policy_name} already attached to role {role_name}")
    except Exception as e:
        raise Exception(f"Failed to attach policy {policy_name}: {str(e)}")

    # Wait for IAM propagation
    _wait_for_iam_propagation(iam_client, role_name, policy_arn)

    return rft_execution_role["Role"]["Arn"]


@dataclass
class StackOutputs:
    """
    Parsed outputs from CloudFormation stack
    """

    rollout_request_arn: str
    rollout_response_sqs_url: str
    rollout_request_queue_url: str
    generate_request_sqs_url: str
    generate_response_sqs_url: str
    proxy_function_url: str
    dynamo_table_name: str


class BaseRFTInfrastructure:
    """
    Base class for platform-specific RFT infrastructure
    """

    def __init__(
        self,
        region: str,
        stack_name: str,
        rft_role_name: str,
        custom_policy_path: Optional[str] = None,
        starter_kit_path: Optional[str] = None,
    ):
        # Validate stack name early (before any AWS operations)
        validate_stack_name(stack_name)

        self.region = region
        self.rft_role_name = rft_role_name
        self.custom_policy_path = custom_policy_path
        self.starter_kit_path = starter_kit_path
        self._starter_kit_s3_uri: Optional[str] = None
        self.cfn_client = boto3.client("cloudformation", region_name=region)

        # Check if stack name has the required suffix
        if not stack_name.endswith(STACK_NAME_SUFFIX):
            # Check if stack already exists
            if self._check_stack_exists(stack_name):
                # Existing stack without suffix - warn user
                logger.warning(
                    f"Stack name '{stack_name}' does not end with '{STACK_NAME_SUFFIX}'. "
                    f"This stack was not created by the Nova SDK. "
                    f"Ensure your IAM role has permissions to access stack resources.\n"
                    f"You can create a role with required permissions using create_rft_execution_role() "
                    f"or manually add permissions to your existing role.\n"
                    f"Recommended: Use stack names ending with '{STACK_NAME_SUFFIX}' for better IAM management."
                )
                self.stack_name = stack_name
                self.is_sdk_managed = False
            else:
                # New stack - auto-append suffix
                self.stack_name = f"{stack_name}-{STACK_NAME_SUFFIX}"
                self.is_sdk_managed = True
        else:
            self.stack_name = stack_name
            self.is_sdk_managed = True

        self.sqs_client = boto3.client("sqs", region_name=region)
        self.logs_client = boto3.client("logs", region_name=region)
        self.iam_client = boto3.client("iam", region_name=region)
        self.sts_client = boto3.client("sts", region_name=region)

    @property
    def rft_policy_name(self) -> str:
        """
        Get the RFT policy name based on role name
        """
        return f"{self.rft_role_name}Policy"

    def _check_stack_exists(self, stack_name: str) -> bool:
        """
        Check if CloudFormation stack exists
        """
        try:
            self.cfn_client.describe_stacks(StackName=stack_name)
            return True
        except self.cfn_client.exceptions.ClientError:
            return False

    def get_account_id(self) -> str:
        """
        Get AWS account ID
        """
        return self.sts_client.get_caller_identity()["Account"]

    def get_current_role_name(self) -> Optional[str]:
        """
        Get the current assumed role name (for local/EC2)
        """
        try:
            identity = self.sts_client.get_caller_identity()
            arn = identity["Arn"]

            # Extract role name from ARN
            # Format: arn:aws:sts::account:assumed-role/role-name/session-name
            if "assumed-role" in arn:
                role_name = arn.split("/")[1]
                return role_name
            # Format: arn:aws:iam::account:role/role-name
            elif ":role/" in arn:
                role_name = arn.split("/")[-1]
                return role_name
        except Exception as e:
            logger.warning(f"Could not determine current role: {e}")

        return None

    def attach_rft_policy_to_role(self, role_name: str):
        """
        Attach RFT policy to a specified role.
        Creates the policy if it doesn't exist, then attaches it to the role.

        Args:
            role_name: Name of the IAM role to attach the policy to
        """
        account_id = self.get_account_id()
        policy_arn = f"arn:aws:iam::{account_id}:policy/{self.rft_policy_name}"

        # Check if policy already attached
        try:
            attached_policies = self.iam_client.list_attached_role_policies(
                RoleName=role_name
            )
            if any(
                p["PolicyArn"] == policy_arn
                for p in attached_policies["AttachedPolicies"]
            ):
                logger.info(f"RFT policy already attached to role {role_name}")
                return
        except Exception as e:
            logger.warning(f"Could not check attached policies: {str(e)}")

        # Get combined policy document
        policy_document = _build_combined_policy_document(
            self.region, account_id, self.custom_policy_path
        )

        try:
            # Try to create the managed policy
            try:
                self.iam_client.create_policy(
                    PolicyName=self.rft_policy_name,
                    PolicyDocument=json.dumps(policy_document),
                    Description="Combined RFT Multiturn infrastructure policy",
                )
                logger.info(f"Created managed policy {self.rft_policy_name}")
            except self.iam_client.exceptions.EntityAlreadyExistsException:
                logger.info(f"Managed policy {self.rft_policy_name} already exists")

            # Attach the policy to the role
            self.iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            logger.info(f"Attached RFT policy to role {role_name}")
            _wait_for_iam_propagation(self.iam_client, role_name, policy_arn)

        except Exception as e:
            error_msg = (
                f"Failed to attach RFT policy to role '{role_name}'. "
                f"Please manually attach the policy '{self.rft_policy_name}' to your role.\n"
                f"Error: {str(e)}"
            )
            raise RuntimeError(error_msg) from e

    def ensure_rft_policy_on_current_role(self):
        """
        Ensure RFT policy is attached to the current assumed role
        """

        role_name = self.get_current_role_name()
        if not role_name:
            logger.warning(
                "Could not determine current role, skipping policy attachment"
            )
            return

        self.attach_rft_policy_to_role(role_name)

    def validate_starter_kit_access(self):
        """
        Validate access to the starter kit S3 bucket
        """
        s3_client = boto3.client("s3", region_name=self.region)
        bucket_name = STARTER_KIT_S3.split("/")[2]
        prefix = "/".join(STARTER_KIT_S3.split("/")[3:])

        try:
            # Check for HEAD file (git repository marker)
            s3_client.head_object(Bucket=bucket_name, Key=f"{prefix}/HEAD")
            logger.info("Successfully validated access to starter kit repository")
        except s3_client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["403", "Forbidden"]:
                raise RuntimeError(
                    f"Access denied to starter kit repository at {STARTER_KIT_S3}. "
                    "This repository requires Nova Forge subscription. "
                    "Please ensure your AWS account is subscribed to Nova Forge. "
                    "See: https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access"
                ) from e
            elif error_code in ["404", "NoSuchKey", "NoSuchBucket"]:
                raise RuntimeError(
                    f"Starter kit repository not found at {STARTER_KIT_S3}. "
                    "Please verify the repository location."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to access starter kit repository at {STARTER_KIT_S3}: {str(e)}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error validating starter kit access: {str(e)}"
            ) from e

    def check_queue_messages(self, queue_url: str) -> Dict[str, int]:
        """
        Check message counts in SQS queue
        """
        response = self.sqs_client.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
                "LastModifiedTimestamp",
            ],
        )
        attrs = response.get("Attributes", {})
        return {
            "visible": int(attrs.get("ApproximateNumberOfMessages", 0)),
            "in_flight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
            "last_receive_timestamp": int(float(attrs.get("LastModifiedTimestamp", 0))),
        }

    def flush_queue(self, queue_url: str):
        """
        Purge all messages from queue
        """
        self.sqs_client.purge_queue(QueueUrl=queue_url)

    # Abstract methods - must be implemented by subclasses
    def validate_platform(self):
        """
        Validate platform-specific requirements
        """
        raise NotImplementedError

    def deploy_sam_stack(self, s3_bucket: Optional[str] = None):
        """
        Deploy SAM stack for Lambda/SQS/DynamoDB

        Args:
            s3_bucket: Optional S3 bucket for starter kit upload
        """
        raise NotImplementedError

    def start_environment(
        self,
        env_type: EnvType,
        vf_env_id: str,
        vf_env_args: Dict,
        stack_outputs: StackOutputs,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Start environment using unified environment client.

        This method uses environment_client.py which provides:
        - Simplified concurrency control (single max_concurrent_rollouts parameter)
        - Per-rollout timeout handling (prevents stuck rollouts from blocking others)
        - Built-in logging and metrics (automatic config snapshots and runtime stats)
        - Duplicate rollout prevention (automatic deduplication by sample_id)
        - YAML configuration support (version-controlled configs with env var interpolation)

        Args:
            env_type: Environment type (EnvType.TRAIN or EnvType.EVAL) for log file naming
            vf_env_id: Verifier environment identifier
            vf_env_args: Environment arguments dictionary
            stack_outputs: Stack outputs containing URLs and queue information
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
                - Replaces old groups_per_batch × max_concurrent_batches × max_workers
                - For migration: use your old max_workers value
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
                - Prevents a single stuck rollout from blocking others
            completion_poll_timeout: Completion polling timeout in seconds (default: 600)
            completion_poll_interval: Completion poll interval in seconds (default: 0.5)
            rollout_poll_interval: SQS polling interval in seconds (default: 1.0)
                - Controls SQS message polling rate
            log_output_directory: Directory for logs and metrics (optional)
                - Writes config snapshots and runtime metrics to JSON files
            config_name: Use YAML config instead of CLI flags (optional)
                - If provided, loads config from configs/{config_name}.yaml
                - When using config_name, other CLI parameters are ignored
            config_path: Custom config directory path (optional)
                - Overrides default "configs/" directory
                - Use with config_name to load from custom location

        Example (passing overrides):
            rft_infra.start_environment(
                env_type=EnvType.TRAIN,
                vf_env_id="wordle",
                vf_env_args={"max_examples": 1, "max_turns": 4},
                max_concurrent_rollouts=40,
                log_output_directory="/opt/ml/output/logs"
            )

        Example (passing config file):
            rft_infra.start_environment(
                env_type=EnvType.TRAIN,
                vf_env_id="wordle",
                vf_env_args={},
                config_name="wordle_training",
                config_path="/path/to/my/configs"
            )
        """
        raise NotImplementedError

    def get_logs(
        self,
        env_type: EnvType,
        limit: int,
        start_from_head: bool,
        log_stream_name: Optional[str],
        tail: bool = False,
    ) -> list:
        """
        Get logs from environment
        """
        raise NotImplementedError

    def kill_task(self, env_type: EnvType):
        """
        Stop running task
        """
        raise NotImplementedError

    def cleanup(self, cleanup_environment: bool = False):
        """
        Clean up platform resources
        """
        raise NotImplementedError

    def get_state(self) -> Dict:
        """
        Get platform-specific state for serialization.

        Returns:
            Dict containing platform-specific state that can be JSON serialized
        """
        raise NotImplementedError

    def restore_state(self, state: Dict):
        """
        Restore platform-specific state after deserialization.

        Args:
            state: Platform-specific state dict from get_state()
        """
        raise NotImplementedError

    def _upload_starter_kit_to_s3(
        self, local_path: str, s3_bucket: str, region: str
    ) -> str:
        """
        Package and upload local starter kit to S3.

        Args:
            local_path: Local path to starter kit directory
            s3_bucket: S3 bucket name
            region: AWS region

        Returns:
            S3 URI of uploaded starter kit

        Raises:
            ValueError: If path is invalid or not a valid starter kit
        """
        expanded_path = os.path.abspath(os.path.expanduser(local_path))

        # Validate it's a starter kit directory
        validate_starter_kit_path(expanded_path)

        # Create tarball
        tar_name = "v1.tar.gz"
        tar_path = os.path.join(tempfile.gettempdir(), tar_name)

        logger.info(f"Packaging starter kit from {expanded_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(expanded_path, arcname="v1")

        # Upload to S3
        s3_key = f"rft-starter-kits/{self.stack_name}/{tar_name}"
        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        logger.info(f"Uploading starter kit to {s3_uri}")
        s3_client = boto3.client("s3", region_name=region)
        s3_client.upload_file(tar_path, s3_bucket, s3_key)

        os.remove(tar_path)
        logger.info("Starter kit uploaded successfully")

        return s3_uri

    def _ensure_starter_kit_available(self, s3_bucket: Optional[str] = None):
        """
        Ensure starter kit is available for remote deployment.
        Auto-uploads if local path is provided.

        Args:
            s3_bucket: S3 bucket for upload (optional, uses SageMaker default if not provided)

        Note:
            This is the base implementation. EC2 overrides this to also run setup commands.
        """
        if not self.starter_kit_path:
            # No custom path - use default starter kit from AWS S3
            return

        # If it's an S3 URI, use it directly
        if self.starter_kit_path.startswith("s3://"):
            self._starter_kit_s3_uri = self.starter_kit_path
            self.starter_kit_s3 = (
                self.starter_kit_path
            )  # Update the attribute used by _build_setup_commands
            logger.info(f"Using starter kit from S3: {self._starter_kit_s3_uri}")
            return

        # It's a local path - validate and upload
        expanded_path = os.path.abspath(os.path.expanduser(self.starter_kit_path))

        # Validate the path exists
        if not os.path.exists(expanded_path):
            raise ValueError(
                f"Starter kit path does not exist: {expanded_path}\n"
                f"Please provide a valid path to an existing starter kit directory."
            )

        # Validate it's a valid starter kit
        validate_starter_kit_path(expanded_path)

        logger.info(f"Auto-uploading starter kit from: {expanded_path}")

        # Determine S3 bucket
        if not s3_bucket:
            # Try to get from custom_env if available
            if (
                hasattr(self, "custom_env")
                and self.custom_env
                and self.custom_env.s3_uri
            ):
                s3_bucket = self.custom_env.s3_uri.split("/")[2]
                logger.info(f"Using bucket from custom_env: {s3_bucket}")
            else:
                # Use SageMaker default bucket
                try:
                    if sagemaker is None:
                        raise ImportError("sagemaker package not installed")
                    session = sagemaker.Session(
                        boto_session=boto3.Session(region_name=self.region)
                    )
                    s3_bucket = session.default_bucket()
                    logger.info(f"Using SageMaker default bucket: {s3_bucket}")
                except ImportError:
                    raise ValueError(
                        "Cannot auto-upload starter kit: s3_bucket parameter required in setup() "
                        "when sagemaker package is not installed"
                    )

        # Upload
        self._starter_kit_s3_uri = self._upload_starter_kit_to_s3(
            expanded_path, s3_bucket, self.region
        )
        self.starter_kit_s3 = (
            self._starter_kit_s3_uri
        )  # Update the attribute used by _build_setup_commands
        logger.info(f"Starter kit will be downloaded from: {self.starter_kit_s3}")

    def _get_base_logs_dir(self) -> str:
        """
        Get base logs directory path for the platform.
        Must be implemented by subclasses to return platform-specific path.

        Returns:
            Base directory path where logs, PID files, and status files are stored
        """
        raise NotImplementedError

    def _get_log_file_path(self, session_id: str, env_type: EnvType) -> str:
        """
        Get log file path for a session and environment type.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)

        Returns:
            Full path to log file
        """
        return f"{self._get_base_logs_dir()}/{session_id}_{env_type.value}.log"

    def _get_pid_file_path(self, session_id: str, env_type: EnvType) -> str:
        """
        Get PID file path for a session and environment type.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)

        Returns:
            Full path to PID file
        """
        return f"{self._get_base_logs_dir()}/{session_id}_{env_type.value}.pid"

    def _get_status_file_path(self, session_id: str, env_type: EnvType) -> str:
        """
        Get status file path for a session and environment type.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)

        Returns:
            Full path to status file
        """
        return f"{self._get_base_logs_dir()}/{session_id}_{env_type.value}.status"

    def _create_pid_file(self, session_id: str, env_type: EnvType, pid: int):
        """
        Create PID file for tracking process.
        Platform-specific implementation required for file creation.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)
            pid: Process ID to store
        """
        raise NotImplementedError

    def _read_pid_file(self, session_id: str, env_type: EnvType) -> Optional[int]:
        """
        Read PID from PID file.
        Platform-specific implementation required for file reading.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)

        Returns:
            Process ID if file exists and is valid, None otherwise
        """
        raise NotImplementedError

    def _create_status_file(self, session_id: str, env_type: EnvType, status: str):
        """
        Create or update status file.
        Platform-specific implementation required for file creation.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)
            status: Status string to store
        """
        raise NotImplementedError

    def _read_status_file(self, session_id: str, env_type: EnvType) -> Optional[str]:
        """
        Read status from status file.
        Platform-specific implementation required for file reading.

        Args:
            session_id: Session identifier
            env_type: Environment type (TRAIN, EVAL, SAM)

        Returns:
            Status string if file exists, None otherwise
        """
        raise NotImplementedError

    def _is_process_running(self, pid: int) -> bool:
        """
        Check if process with given PID is running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, False otherwise
        """
        try:
            # Send signal 0 to check if process exists
            # This doesn't actually send a signal, just checks permissions
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _check_for_running_jobs_on_stack(self, stack_name: str) -> Optional[str]:
        """
        Check if any jobs are running for this stack (across all sessions).

        Args:
            stack_name: Stack name to check

        Returns:
            Error message if running jobs found, None otherwise

        Note:
            Must be implemented by subclasses for platform-specific checking.
        """
        raise NotImplementedError

    @staticmethod
    def _extract_base_stack_name(stack_name: str) -> str:
        """
        Extract base stack name by removing NovaForgeSDK suffix.

        Args:
            stack_name: Full stack name

        Returns:
            Base stack name without suffix
        """
        return stack_name.replace("-NovaForgeSDK", "").replace("NovaForgeSDK", "")
