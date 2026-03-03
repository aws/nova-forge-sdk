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
Main user-facing RFT Multiturn Infrastructure API.
Delegates to platform-specific implementations while preserving UX.
"""

import glob
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_forge_sdk.model.model_enums import TrainingMethod
from amzn_nova_forge_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_forge_sdk.util.logging import logger

from .base_infra import (
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
    VFEnvId,
    create_rft_execution_role,
)
from .constants import RFT_EXECUTION_ROLE_NAME, STACK_NAME_SUFFIX
from .custom_environment import CustomEnvironment
from .ec2_infra import EC2RFTInfrastructure
from .ecs_infra import ECSRFTInfrastructure
from .local_infra import LocalRFTInfrastructure


def list_rft_stacks(region: str = "us-east-1", all_stacks: bool = False) -> List[str]:
    """
    List CloudFormation stacks in the region.

    Args:
        region: AWS region to list stacks from (default: us-east-1)
        all_stacks: If True, list all stacks. If False, only list stacks ending with NovaSDK suffix.

    Returns:
        List of stack names

    Example:
        >>> from amzn_nova_forge_sdk.rft_multiturn import list_rft_stacks
        >>> # List only Nova SDK stacks
        >>> nova_stacks = list_rft_stacks(region="us-east-1")
        >>> # List all stacks
        >>> all_stacks = list_rft_stacks(region="us-east-1", all_stacks=True)
    """
    cfn_client = boto3.client("cloudformation", region_name=region)

    try:
        paginator = cfn_client.get_paginator("list_stacks")
        stack_names = []

        for page in paginator.paginate(
            StackStatusFilter=[
                "CREATE_COMPLETE",
                "UPDATE_COMPLETE",
                "UPDATE_ROLLBACK_COMPLETE",
            ]
        ):
            for stack in page["StackSummaries"]:
                stack_name = stack["StackName"]
                if all_stacks or stack_name.endswith(STACK_NAME_SUFFIX):
                    stack_names.append(stack_name)

        if not all_stacks:
            logger.info(
                f"Found {len(stack_names)} Nova Forge SDK stack(s) in region {region}"
            )
        else:
            logger.info(f"Found {len(stack_names)} stack(s) in region {region}")

        return sorted(stack_names)

    except Exception as e:
        raise RuntimeError(f"Failed to list stacks in region {region}: {str(e)}") from e


class RFTMultiturnInfrastructure:
    """
    High-level orchestration with CloudFormation stack management.
    Manages infrastructure and environment for RFT multiturn training.
    Supports LOCAL, EC2, and ECS deployment platforms.
    """

    def __init__(
        self,
        stack_name: str,
        region: str = "us-east-1",
        vf_env_id: Optional[VFEnvId] = None,
        custom_env: Optional[CustomEnvironment] = None,
        infrastructure_arn: Optional[str] = None,
        python_venv_name: Optional[str] = None,
        starter_kit_path: Optional[str] = None,
        vpc_config: Optional[Dict[str, Any]] = None,
        cpu: Optional[str] = None,
        memory: Optional[str] = None,
        rft_role_name: Optional[str] = None,
        custom_policy_path: Optional[str] = None,
    ):
        """
        Initialize RFT multiturn infrastructure.

        Args:
            stack_name: CloudFormation stack name for Lambda infrastructure
            region: AWS region (default: us-east-1)
            vf_env_id: Built-in environment ID (VFEnvId.WORDLE or VFEnvId.TERMINAL_BENCH)
            custom_env: CustomEnvironment instance (mutually exclusive with vf_env_id)
            infrastructure_arn: Optional infrastructure ARN
                - None: Runs locally on your machine
                - "i-xxx" or "arn:aws:ec2:...": Uses EC2 instance
                - "arn:aws:ecs:...": Uses ECS cluster
            python_venv_name: Name for Python virtual environment (required for LOCAL/EC2, optional for ECS)
            starter_kit_path: Optional custom starter kit path
                - For LOCAL: Local file path (e.g., "~/my-starter-kit" or "/path/to/v1")
                - For EC2/ECS: Local path (auto-uploaded to S3) or S3 URI (e.g., "s3://bucket/path/v1.tar.gz")
                - If None: Uses default starter kit from AWS
            vpc_config: VPC configuration for ECS (ignored for LOCAL/EC2). Dict with keys:
                - subnets: List[str] - Subnet IDs
                - security_groups: List[str] - Security group IDs
            cpu: CPU units for ECS tasks (e.g., "2048"). Ignored for LOCAL/EC2.
            memory: Memory in MB for ECS tasks (e.g., "4096"). Ignored for LOCAL/EC2.
            rft_role_name: Optional IAM role name for RFT infrastructure. If not provided, uses default.
                If role doesn't exist, it will be created automatically.
            custom_policy_path: Optional path to custom policy JSON file. If not provided, uses SDK default.
                Must follow the same structure as rft_multiturn_policies.json with placeholder strings.

        Example:
            # Built-in environment (LOCAL)
            >>> RFTMultiturnInfrastructure(
            ...     stack_name="my-stack",
            ...     python_venv_name="my_rft_venv",
            ...     vf_env_id=VFEnvId.WORDLE
            ... )

            # Custom environment (LOCAL)
            >>> custom_env = CustomEnvironment(
            ...     env_id="my-custom-env",
            ...     local_path="~/custom_envs/my-custom-env"
            ... )
            >>> RFTMultiturnInfrastructure(
            ...     stack_name="my-stack",
            ...     python_venv_name="my_rft_venv",
            ...     custom_env=custom_env
            ... )

            # EC2 deployment
            >>> RFTMultiturnInfrastructure(
            ...     stack_name="my-stack",
            ...     python_venv_name="my_rft_venv",
            ...     infrastructure_arn="i-1234567890abcdef0",
            ...     vf_env_id=VFEnvId.WORDLE
            ... )

            # ECS deployment with VPC config and custom resources
            >>> RFTMultiturnInfrastructure(
            ...     stack_name="my-stack",
            ...     infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
            ...     vf_env_id=VFEnvId.WORDLE,
            ...     vpc_config={
            ...         "subnets": ["subnet-12345", "subnet-67890"],
            ...         "security_groups": ["sg-12345"]
            ...     },
            ...     cpu="4096",
            ...     memory="8192"
            ... )
        """
        self.region = region

        # Validate environment parameters
        if vf_env_id and custom_env:
            raise ValueError("Cannot specify both vf_env_id and custom_env")

        if not vf_env_id and not custom_env:
            raise ValueError("Please specify one of the vf_env_id and custom_env")

        # Set environment ID and type
        self.custom_env: Optional[CustomEnvironment]
        if custom_env:
            self.env_id = custom_env.env_id
            self.is_custom_env = True
            self.custom_env = custom_env
        else:
            if isinstance(vf_env_id, VFEnvId):
                self.env_id = vf_env_id.value
            elif isinstance(vf_env_id, str):
                try:
                    self.env_id = VFEnvId(vf_env_id).value
                except ValueError:
                    valid_values = [e.value for e in VFEnvId]
                    raise ValueError(
                        f"Invalid vf_env_id: '{vf_env_id}'. "
                        f"Must be one of: {valid_values}"
                    )
            else:
                raise TypeError(
                    f"vf_env_id must be VFEnvId or str, got {type(vf_env_id)}"
                )
            self.is_custom_env = False
            self.custom_env = None

        self.infrastructure_arn = infrastructure_arn
        self.platform = self.detect_platform(infrastructure_arn)

        # Validate custom environment for platform
        if self.is_custom_env and custom_env:
            if self.platform == "local" and not custom_env.local_path:
                raise ValueError(
                    "CustomEnvironment.local_path required for LOCAL platform"
                )
            elif self.platform in ["ec2", "ecs"] and not custom_env.s3_uri:
                raise ValueError(
                    f"CustomEnvironment.s3_uri required for {self.platform.upper()} platform. "
                    f"Please call package_and_upload() on your CustomEnvironment object."
                )

        self.workspace_dir = os.getcwd()
        self.stack_outputs: Optional[StackOutputs] = None
        self.recipe_cache_dir = os.path.join(self.workspace_dir, ".nova_rft_recipes")

        # Initialize platform-specific infrastructure
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        # Validate and set python_venv_name based on platform
        if self.platform in ["local", "ec2"]:
            if not python_venv_name:
                raise ValueError(
                    f"python_venv_name is required for {self.platform.upper()} platform"
                )
        else:  # ecs
            python_venv_name = python_venv_name or "rft_nova_venv"  # Default for ECS

        # Handle RFT role name
        self.rft_role_name = rft_role_name or RFT_EXECUTION_ROLE_NAME
        self.custom_policy_path = custom_policy_path

        # Generate timestamp-based session ID for this instance
        # Format: {stack_name}_{YYYY-MM-DD-HHMMSS}
        # Must be set before creating infrastructure objects (EC2 needs it)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self._session_id = f"{stack_name}_{timestamp}"
        self._state_file_path: Optional[Path] = None

        if self.platform == "local":
            self.infra = LocalRFTInfrastructure(
                region,
                stack_name,
                self.workspace_dir,
                python_venv_name,
                self.rft_role_name,
                self.custom_policy_path,
                starter_kit_path,
                self._session_id,
            )
        elif self.platform == "ec2":
            if not infrastructure_arn:
                raise ValueError("infrastructure_arn required for EC2 platform")
            self.infra = EC2RFTInfrastructure(
                region,
                stack_name,
                infrastructure_arn,
                python_venv_name,
                self.rft_role_name,
                self.custom_policy_path,
                starter_kit_path,
                self.session_id,
            )  # type: ignore[assignment]
        else:  # ecs
            if not infrastructure_arn:
                raise ValueError("infrastructure_arn required for ECS platform")
            self.infra = ECSRFTInfrastructure(
                region,
                stack_name,
                infrastructure_arn,
                account_id,
                python_venv_name,
                self.rft_role_name,
                self.custom_policy_path,
                starter_kit_path,
                vpc_config=vpc_config,
                cpu=cpu,
                memory=memory,
            )  # type: ignore[assignment]

        # Use the prefixed stack_name from infra
        self.stack_name = self.infra.stack_name

        # Pass custom environment to infrastructure
        if self.is_custom_env and custom_env:
            self.infra.custom_env = custom_env  # type: ignore[attr-defined]

        # Check if stack exists (use self.stack_name which includes the suffix)
        self._stack_exists = self._check_stack_exists(self.stack_name)
        if self._stack_exists:
            logger.info(f"Using existing CloudFormation stack: '{self.stack_name}'")
        else:
            logger.info(
                f"CloudFormation stack with name '{self.stack_name}' will be created"
            )

        logger.info(f"RFT Infrastructure session ID: {self._session_id}")

    def _ensure_rft_role_exists(self):
        """Ensure RFT execution role exists, create if it doesn't"""
        iam_client = boto3.client("iam", region_name=self.region)

        try:
            # Check if role exists
            iam_client.get_role(RoleName=self.rft_role_name)
            logger.info(f"Using existing RFT role: {self.rft_role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            # Role doesn't exist, create it
            create_rft_execution_role(
                region=self.region,
                role_name=self.rft_role_name,
                custom_policy_path=self.custom_policy_path,
            )
            logger.info(f"Successfully created RFT role: {self.rft_role_name}")

    def detect_platform(self, infrastructure_arn: Optional[str]) -> str:
        """Auto-detect platform from infrastructure ARN"""
        if infrastructure_arn is None:
            return "local"
        if infrastructure_arn.startswith(
            "arn:aws:ec2:"
        ) or infrastructure_arn.startswith("i-"):
            return "ec2"
        elif infrastructure_arn.startswith("arn:aws:ecs:"):
            return "ecs"
        else:
            raise ValueError(f"Unknown infrastructure ARN format: {infrastructure_arn}")

    def setup(self, s3_bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Setup RFT multiturn infrastructure and deploy CloudFormation stack.

        Args:
            s3_bucket: Optional S3 bucket for starter kit upload (EC2/ECS only).
                If not provided, uses SageMaker default bucket or bucket from custom_env.

        This method performs the following actions in your AWS environment:

        **IAM Resources Created/Modified:**
        - Creates IAM execution role (if not exists) with name from `rft_role_name` parameter
        - Attaches policy with permissions for:
          - CloudFormation (create/update/delete stacks)
          - DynamoDB (create/manage tables for task tracking)
          - Lambda (create/invoke functions for reward computation)
          - SQS (create/manage queues for task distribution)
          - S3 (read starter kit, write artifacts)
          - CloudWatch Logs (create log groups/streams)
          - ECR (for ECS: pull container images)
          - ECS (for ECS: run tasks)

        **Infrastructure Deployed (via CloudFormation):**
        - Lambda functions: Proxy function for task management, rollout function for reward computation
        - SQS queues (FIFO): Request/response queues for generation and rollout tasks
        - DynamoDB table: Task state tracking
        - CloudWatch log groups: Lambda function logs

        **Local/Remote Setup (platform-specific):**
        - Downloads RFT starter kit from S3 to your environment
        - Creates Python virtual environment at specified location
        - Installs dependencies: boto3, git-remote-s3, aws-sam-cli (for SAM deployment)
        - Installs starter kit packages and custom environments

        **For LOCAL platform:**
        - Downloads to: `~/.nova-rft-workspace/<stack_name>/v1/`
        - Creates venv: `~/.nova-rft-workspace/<stack_name>/v1/<python_venv_name>/`

        **For EC2 platform:**
        - Downloads to: `/home/ec2-user/v1/`
        - Creates venv: `/home/ec2-user/v1/<python_venv_name>/`
        - Executes via SSM commands on the specified EC2 instance

        **For ECS platform:**
        - Downloads to: `/root/v1/` (inside container)
        - Creates venv: `/root/v1/<python_venv_name>/`
        - Runs as Fargate tasks in the specified ECS cluster

        **Validation Checks:**
        - Verifies IAM permissions for current user
        - Validates access to Nova Forge starter kit S3 bucket
        - If stack exists: Checks that all SQS queues are empty (prevents conflicts)

        Returns:
            Dict[str, Any]: Configuration dictionary containing:
                - stack_name: CloudFormation stack name
                - region: AWS region
                - platform: Deployment platform (local/ec2/ecs)
                - stack_outputs: Lambda URLs, SQS queue URLs, DynamoDB table name

        Raises:
            RuntimeError: If existing stack has non-empty queues (call flush_all_queues() first)
            ClientError: If IAM permissions are insufficient or S3 access is denied

        Example:
            >>> rft_infra = RFTMultiturnInfrastructure(
            ...     stack_name="my-rft-stack",
            ...     region="us-east-1",
            ...     python_venv_name="my_venv"
            ... )
            >>> config = rft_infra.setup()  # Deploys all infrastructure
            >>> print(config['stack_outputs']['proxy_function_url'])
        """
        # Step 1: Ensure RFT execution role exists
        self._ensure_rft_role_exists()

        # Step 2: Validate IAM permissions and Forge access
        logger.info("Validating IAM permissions and Forge access...")
        self.infra.ensure_rft_policy_on_current_role()
        self.infra.validate_starter_kit_access()

        if self._stack_exists:
            self._load_existing_stack(self.stack_name)
            self._configure_lambda_url(self.stack_name)

            # Ensure starter kit is available even for existing stacks (in case starter_kit_path was added/changed)
            if self.platform != "local":
                # Just ensure S3 upload - setup commands run in start_environment()
                self.infra._ensure_starter_kit_available(s3_bucket)

            # Check if queues have messages
            queue_status = self.check_all_queues()
            non_empty_queues = []
            current_time = int(time.time())

            for queue_name, counts in queue_status.items():
                total_messages = counts["visible"] + counts["in_flight"]
                if total_messages > 0:
                    last_receive = counts.get("last_receive_timestamp", 0)
                    if last_receive > 0:
                        seconds_ago = current_time - last_receive
                        if seconds_ago < 60:
                            time_str = f"{seconds_ago}s ago"
                        elif seconds_ago < 3600:
                            time_str = f"{seconds_ago // 60}m ago"
                        else:
                            time_str = f"{seconds_ago // 3600}h ago"
                        non_empty_queues.append(
                            f"{queue_name} ({counts['visible']} visible, {counts['in_flight']} in-flight, last message {time_str})"
                        )
                    else:
                        non_empty_queues.append(
                            f"{queue_name} ({counts['visible']} visible, {counts['in_flight']} in-flight)"
                        )

            if non_empty_queues:
                raise RuntimeError(
                    f"Cannot use stack - queues are not empty:\n"
                    + "\n".join(f"  - {q}" for q in non_empty_queues)
                    + "\n\nPlease run flush_all_queues() to clear the queues"
                )

            logger.info(f"Using stack '{self.stack_name}'")
            return self.get_configuration()

        # Stack doesn't exist - deploy it
        # Step 3: Deploy stack
        logger.info(f"Deploying new stack '{self.stack_name}'...")

        if self.platform == "local":
            self.infra.setup_local(self.workspace_dir)
        else:
            self.infra.validate_platform()

        self.infra.deploy_sam_stack(s3_bucket)

        self._load_existing_stack(self.stack_name)
        self._configure_lambda_url(self.stack_name)
        self._stack_exists = True

        logger.info(f"Stack '{self.stack_name}' deployed successfully")
        return self.get_configuration()

    def start_environment(
        self,
        env_type: EnvType,
        vf_env_args: Optional[Dict] = None,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
        queue_url: Optional[str] = None,
    ):
        """Start environment using unified environment client.

        This method uses the unified client (environment_client.py) which provides:
        - Simplified concurrency control (single max_concurrent_rollouts parameter)
        - Per-rollout timeout handling (prevents stuck rollouts from blocking others)
        - Built-in logging and metrics (automatic config snapshots and runtime stats)
        - Duplicate rollout prevention (automatic deduplication by sample_id)
        - YAML configuration support (version-controlled configs with env var interpolation)

        Args:
            env_type: Environment type (EnvType.TRAIN or EnvType.EVAL)
            vf_env_args: Environment-specific arguments (optional, defaults to {"use_think": False})
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
            completion_poll_timeout: Completion polling timeout in seconds (default: 600)
            completion_poll_interval: Completion poll interval in seconds (default: 0.5)
            rollout_poll_interval: SQS polling interval in seconds (default: 1.0)
            log_output_directory: Directory for logs and metrics (optional)
                - Recommended: "/opt/ml/output/logs" for SageMaker, "./logs" for local
            config_name: Use YAML config instead of CLI flags (optional)
                - If provided, loads config from configs/{config_name}.yaml
            config_path: Custom config directory path (optional, use with config_name)
            queue_url: SQS queue URL (optional, defaults to training queue from stack)

        Example (CLI mode):
            rft_infra.start_environment(
                env_type=EnvType.TRAIN,
                vf_env_args={"max_examples": 1, "max_turns": 4},
                max_concurrent_rollouts=40,
                log_output_directory="/opt/ml/output/logs"
            )

        Example (YAML mode):
            rft_infra.start_environment(
                env_type=EnvType.TRAIN,
                config_name="wordle_training"
            )

        Example (custom config path):
            rft_infra.start_environment(
                env_type=EnvType.TRAIN,
                config_name="wordle_training",
                config_path="/custom/configs"
            )
        """
        if not self.stack_outputs:
            raise RuntimeError("Stack not deployed. Run setup() first.")

        # Default to training queue if not provided
        if not queue_url:
            queue_url = self.stack_outputs.rollout_request_queue_url
            logger.info(f"Using queue: {queue_url}")

        vf_env_args = vf_env_args or {"use_think": False}

        if self.platform == "local":
            # Ensure starter kit is set up for local platform
            self.infra.setup_local(self.workspace_dir)
            self.infra.install_local_environment(self.env_id)
        elif self.platform == "ec2":
            # EC2 setup happens automatically in _execute_training_or_eval via _build_setup_commands
            pass
        elif self.platform == "ecs":
            # ECS doesn't need setup - environment is in the container image
            pass

        self.infra.start_environment(
            env_type=env_type,
            vf_env_id=self.env_id,
            vf_env_args=vf_env_args,
            stack_outputs=self.stack_outputs,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_rollout_timeout=max_rollout_timeout,
            completion_poll_timeout=completion_poll_timeout,
            completion_poll_interval=completion_poll_interval,
            rollout_poll_interval=rollout_poll_interval,
            log_output_directory=log_output_directory,
            config_name=config_name,
            config_path=config_path,
        )

        logger.info(f"Environment started on {self.platform}")

    def start_training_environment(
        self,
        vf_env_args: Optional[Dict] = None,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
        queue_url: Optional[str] = None,
    ):
        """DEPRECATED: Use start_environment(env_type=EnvType.TRAIN, ...) instead.

        This method is deprecated and will be removed in a future version.
        Please use start_environment(env_type=EnvType.TRAIN, ...) instead.

        This is now a simple wrapper that calls start_environment with env_type=EnvType.TRAIN.
        All parameters are passed through directly.

        Args:
            vf_env_args: Environment-specific arguments (optional, defaults to {"use_think": False})
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
            completion_poll_timeout: Completion polling timeout in seconds (default: 600)
            completion_poll_interval: Completion poll interval in seconds (default: 0.5)
            rollout_poll_interval: SQS polling interval in seconds (default: 1.0)
            log_output_directory: Directory for logs and metrics (optional)
            config_name: Use YAML config instead of CLI flags (optional)
            config_path: Custom config directory path (optional, use with config_name)
            queue_url: SQS queue URL (optional, defaults to training queue from stack)
        """
        logger.warning(
            "start_training_environment() is deprecated and will be removed in a future version. "
            "Please use start_environment(env_type=EnvType.TRAIN, ...) instead.",
        )

        return self.start_environment(
            env_type=EnvType.TRAIN,
            vf_env_args=vf_env_args,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_rollout_timeout=max_rollout_timeout,
            completion_poll_timeout=completion_poll_timeout,
            completion_poll_interval=completion_poll_interval,
            rollout_poll_interval=rollout_poll_interval,
            log_output_directory=log_output_directory,
            config_name=config_name,
            config_path=config_path,
            queue_url=queue_url,
        )

    def start_evaluation_environment(
        self,
        vf_env_args: Optional[Dict] = None,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
        queue_url: Optional[str] = None,
    ):
        """DEPRECATED: Use start_environment(env_type=EnvType.EVAL, ...) instead.

        This method is deprecated and will be removed in a future version.
        Please use start_environment(env_type=EnvType.EVAL, ...) instead.

        This is now a simple wrapper that calls start_environment with env_type=EnvType.EVAL.
        All parameters are passed through directly.

        Args:
            vf_env_args: Environment-specific arguments (optional, defaults to {"use_think": False})
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
            completion_poll_timeout: Completion polling timeout in seconds (default: 600)
            completion_poll_interval: Completion poll interval in seconds (default: 0.5)
            rollout_poll_interval: SQS polling interval in seconds (default: 1.0)
            log_output_directory: Directory for logs and metrics (optional)
            config_name: Use YAML config instead of CLI flags (optional)
            config_path: Custom config directory path (optional, use with config_name)
            queue_url: SQS queue URL (optional, defaults to training queue from stack)
        """
        logger.warning(
            "start_evaluation_environment() is deprecated and will be removed in a future version. "
            "Please use start_environment(env_type=EnvType.EVAL, ...) instead."
        )

        return self.start_environment(
            env_type=EnvType.EVAL,
            vf_env_args=vf_env_args,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_rollout_timeout=max_rollout_timeout,
            completion_poll_timeout=completion_poll_timeout,
            completion_poll_interval=completion_poll_interval,
            rollout_poll_interval=rollout_poll_interval,
            log_output_directory=log_output_directory,
            config_name=config_name,
            config_path=config_path,
            queue_url=queue_url,
        )

    def get_logs(
        self,
        env_type: Optional[EnvType] = None,
        limit: int = 100,
        start_from_head: bool = False,
        log_stream_name: Optional[str] = None,
        tail: bool = False,
    ) -> list[str]:
        """Get logs from environment

        Args:
            env_type: Environment type (TRAIN, EVAL, SAM). Defaults to TRAIN.
            limit: Maximum number of log lines to return
            start_from_head: If True, start from beginning of logs
            log_stream_name: Optional specific log stream name (ECS only)
            tail: If True, continuously stream logs in real-time (blocks until Ctrl+C)

        Returns:
            List of log lines (empty if tail=True)
        """
        if env_type is None:
            env_type = EnvType.TRAIN

        return self.infra.get_logs(
            env_type, limit, start_from_head, log_stream_name, tail
        )

    def kill_task(self, env_type: Optional[EnvType] = None, **kwargs):
        """
        Kill training or evaluation task.

        Args:
            env_type: Type of environment (TRAIN or EVAL). Defaults to TRAIN if not specified.
            **kwargs: Additional platform-specific parameters:
                - kill_all_for_stack (bool): If True, kills ALL jobs of this type for the stack (cross-session).
                                            If False, only kills jobs from current session. (All platforms)
                - task_id (str): Optional task ID for ECS (ECS only)
                - deregister_task_def (bool): If True, deregister task definition (ECS only)
        """
        if env_type is None:
            env_type = EnvType.TRAIN

        self.infra.kill_task(env_type, **kwargs)
        logger.info(f"Killed {env_type.value} task on {self.platform}")

    def cleanup(self, delete_stack: bool = False, cleanup_environment: bool = False):
        """Clean up resources

        Args:
            delete_stack: If True, delete the CloudFormation stack
            cleanup_environment: If True, clean up environment resources:
                - LOCAL/EC2: Delete virtual environment and starter kit directories
                - ECS: Deregister task definitions
        """
        self.infra.cleanup(cleanup_environment=cleanup_environment)

        if delete_stack and self.stack_outputs:
            self._delete_stack(self.stack_name)

    def check_all_queues(self) -> Dict[str, Dict[str, int]]:
        """Check message counts in all queues"""
        if not self.stack_outputs:
            raise RuntimeError("Stack not deployed. Run setup() first.")

        queues = self._get_queue_urls()
        return {
            name: self.infra.check_queue_messages(url) for name, url in queues.items()
        }

    def flush_all_queues(self):
        """Flush all messages from all queues"""
        if not self.stack_outputs:
            raise RuntimeError("Stack not deployed. Run setup() first.")

        for name, url in self._get_queue_urls().items():
            self.infra.flush_queue(url)
            logger.info(f"Flushed {name} queue")

    def get_configuration(self) -> Dict[str, Any]:
        """Get complete configuration"""
        config = {
            "stack_name": self.stack_name,
            "region": self.region,
            "platform": self.platform,
            "infrastructure_arn": self.infrastructure_arn,
        }

        if self.stack_outputs:
            config["stack_outputs"] = {  # type: ignore[assignment]
                "proxy_function_url": self.stack_outputs.proxy_function_url,
                "rollout_request_queue_url": self.stack_outputs.rollout_request_queue_url,
                "rollout_response_sqs_url": self.stack_outputs.rollout_response_sqs_url,
                "generate_request_sqs_url": self.stack_outputs.generate_request_sqs_url,
                "generate_response_sqs_url": self.stack_outputs.generate_response_sqs_url,
            }

        return config

    def get_recipe_path(self, method) -> str:
        """Download and cache specific recipe file from S3"""
        from amzn_nova_forge_sdk.model.model_enums import TrainingMethod
        from amzn_nova_forge_sdk.recipe.recipe_config import EvaluationTask

        recipe_map = {
            TrainingMethod.RFT_MULTITURN_LORA: "fine-tuning/nova/forge/nova_2_0/nova_lite/RFT/nova_lite_2_0_p5_gpu_lora_rft_byoo.yaml",
            TrainingMethod.RFT_MULTITURN_FULL: "fine-tuning/nova/forge/nova_2_0/nova_lite/RFT/nova_lite_2_0_p5_gpu_rft_byoo.yaml",
            EvaluationTask.RFT_MULTITURN_EVAL: "evaluation/nova/forge/nova_2_0/nova_lite/nova_lite_2_0_p5_gpu_rft_long_process_eval.yaml",
        }

        s3_key = recipe_map.get(method)
        if not s3_key:
            raise ValueError(f"No recipe mapping found for method: {method}")

        os.makedirs(self.recipe_cache_dir, exist_ok=True)
        local_path = os.path.join(self.recipe_cache_dir, os.path.basename(s3_key))

        if not os.path.exists(local_path):
            logger.info(f"Downloading recipe from S3: {s3_key}")
            s3_client = boto3.client("s3", region_name=self.region)
            s3_client.download_file(
                "nova-forge-c7363-206080352451-us-east-1",
                f"v1/src/hyperpod_cli/sagemaker_hyperpod_recipes/recipes_collection/recipes/{s3_key}",
                local_path,
            )
            logger.info(f"Recipe cached at: {local_path}")
        else:
            logger.info(f"Using cached recipe: {local_path}")

        return local_path

    def get_recipe_overrides(self) -> Dict[str, Any]:
        """Get recipe parameter overrides for RFT multiturn jobs"""
        if not self.stack_outputs:
            raise RuntimeError("Stack outputs not loaded. Run setup() first.")

        if not self.stack_outputs:
            raise RuntimeError("Stack outputs not available. Deploy SAM stack first.")

        return {
            "rollout_request_arn": self.stack_outputs.rollout_request_arn,
            "rollout_response_sqs_url": self.stack_outputs.rollout_response_sqs_url,
            "rollout_request_queue_url": self.stack_outputs.rollout_request_queue_url,
            "generate_request_sqs_url": self.stack_outputs.generate_request_sqs_url,
            "generate_response_sqs_url": self.stack_outputs.generate_response_sqs_url,
        }

    def _get_queue_urls(self) -> Dict[str, str]:
        """Get all queue URLs as a dict"""
        if not self.stack_outputs:
            raise RuntimeError("Stack outputs not available. Deploy SAM stack first.")

        return {
            "rollout_request": self.stack_outputs.rollout_request_queue_url,
            "rollout_response": self.stack_outputs.rollout_response_sqs_url,
            "generate_request": self.stack_outputs.generate_request_sqs_url,
            "generate_response": self.stack_outputs.generate_response_sqs_url,
        }

    def _check_stack_exists(self, stack_name: str) -> bool:
        """Check if CloudFormation stack exists"""
        cfn_client = boto3.client("cloudformation", region_name=self.region)
        try:
            cfn_client.describe_stacks(StackName=stack_name)
            return True
        except cfn_client.exceptions.ClientError:
            return False

    def _load_existing_stack(self, stack_name: str):
        """Load outputs from existing stack"""
        cfn_client = boto3.client("cloudformation", region_name=self.region)
        response = cfn_client.describe_stacks(StackName=stack_name)
        outputs = response["Stacks"][0]["Outputs"]

        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]
        rollout_function_name = self._get_output(outputs, "RolloutFunctionName")

        self.stack_outputs = StackOutputs(
            rollout_request_arn=f"arn:aws:lambda:{self.region}:{account_id}:function:{rollout_function_name}",
            rollout_response_sqs_url=self._get_output(
                outputs, "RolloutFinishedQueueUrl"
            ),
            rollout_request_queue_url=self._get_output(outputs, "RolloutQueueUrl"),
            generate_request_sqs_url=self._get_output(outputs, "RequestQueueUrl"),
            generate_response_sqs_url=self._get_output(
                outputs, "RequestResponseQueueUrl"
            ),
            proxy_function_url=self._get_output(outputs, "ProxyFunctionUrl"),
            dynamo_table_name=self._get_output(outputs, "TasksTableName"),
        )

    def _configure_lambda_url(self, stack_name: str):
        """Configure Lambda URL for IAM authentication"""
        lambda_client = boto3.client("lambda", region_name=self.region)
        function_name = f"{stack_name}-SageMaker-proxy"

        lambda_client.update_function_url_config(
            FunctionName=function_name, AuthType="AWS_IAM"
        )

    def _delete_stack(self, stack_name: str):
        """Delete CloudFormation stack"""
        cfn_client = boto3.client("cloudformation", region_name=self.region)
        cfn_client.delete_stack(StackName=stack_name)

    def _get_output(self, outputs: list, key: str) -> str:
        """Extract output value from CloudFormation outputs"""
        for output in outputs:
            if output["OutputKey"] == key:
                return output["OutputValue"]
        raise ValueError(f"Output {key} not found in stack")

    @property
    def session_id(self) -> str:
        """Get the unique session identifier for this infrastructure instance"""
        return self._session_id

    def dump(
        self,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        include_session_id: bool = True,
    ) -> Path:
        """
        Save infrastructure state to file for session recovery.

        Args:
            file_path: Directory path to save the result. Saves to current directory if not provided
            file_name: File name (if None, auto-generates with session_id)
            include_session_id: If True, includes session_id in filename

        Returns:
            Path to saved state file

        Examples:
            >>> rft_infra.dump()
            # Saves to current directory: rft_state_my-stack_a1b2c3d4.json

            >>> rft_infra.dump(file_path="/home/user/workspace")
            # Saves to: /home/user/workspace/rft_state_my-stack_a1b2c3d4.json

            >>> rft_infra.dump(file_name="my_state.json", include_session_id=False)
            # Saves to current directory: my_state.json
        """
        if file_name is None:
            if include_session_id:
                file_name = f"rft_state_{self.stack_name}_{self._session_id}.json"
            else:
                file_name = f"rft_state_{self.stack_name}.json"

        if file_path is None:
            full_path = Path(file_name)
        else:
            full_path = Path(file_path) / file_name

        state = {
            "__class_name__": self.__class__.__name__,
            "session_id": self._session_id,
            "platform": self.platform,
            "region": self.region,
            "stack_name": self.stack_name,
            "env_id": self.env_id,
            "workspace_dir": self.workspace_dir,
            "rft_role_name": self.rft_role_name,
            "infrastructure_arn": self.infrastructure_arn,
            "is_custom_env": self.is_custom_env,
            "custom_env": (
                {
                    "env_id": self.custom_env.env_id,
                    "local_path": self.custom_env.local_path,
                    "s3_uri": self.custom_env.s3_uri,
                    "output_dir": self.custom_env.output_dir,
                    "env_type": self.custom_env.env_type,
                }
                if self.custom_env
                else None
            ),
            "stack_outputs": (
                asdict(self.stack_outputs) if self.stack_outputs else None
            ),
            "infra_state": self.infra.get_state(),
            "dumped_at": datetime.now().isoformat(),
        }

        with open(full_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        self._state_file_path = full_path
        logger.info(f"Infrastructure state saved to {full_path}")
        return full_path

    @classmethod
    def load(
        cls, file_path: str, auto_reconnect: bool = True
    ) -> "RFTMultiturnInfrastructure":
        """
        Load infrastructure state from file and reconnect to running processes.

        Args:
            file_path: Path to state file
            auto_reconnect: If True, verify and reconnect to running processes

        Returns:
            Reconstructed RFTMultiturnInfrastructure instance

        Example:
            >>> # After notebook restart
            >>> rft_infra = RFTMultiturnInfrastructure.load(
            ...     ".rft_state_my-stack_a1b2.json"
            ... )
            >>> print(f"Reconnected to session: {rft_infra.session_id}")
        """
        with open(file_path, "r") as f:
            state = json.load(f)

        platform = state["platform"]
        infra_state = state["infra_state"]

        custom_env = None
        if state.get("is_custom_env") and state.get("custom_env"):
            custom_env_data = state["custom_env"]
            custom_env = CustomEnvironment(
                env_id=custom_env_data["env_id"],
                local_path=custom_env_data.get("local_path"),
                s3_uri=custom_env_data.get("s3_uri"),
                output_dir=custom_env_data.get("output_dir", "~/custom_envs"),
                env_type=custom_env_data.get("env_type", "single_turn"),
            )

        if platform == "local":
            infra = cls(
                stack_name=state["stack_name"],
                region=state["region"],
                vf_env_id=state["env_id"] if not state["is_custom_env"] else None,
                custom_env=custom_env,
                infrastructure_arn=None,
                python_venv_name=infra_state["python_venv_name"],
                rft_role_name=state["rft_role_name"],
            )
        elif platform == "ec2":
            instance_id = infra_state["instance_id"]
            infra = cls(
                stack_name=state["stack_name"],
                region=state["region"],
                vf_env_id=state["env_id"] if not state["is_custom_env"] else None,
                custom_env=custom_env,
                infrastructure_arn=instance_id,
                python_venv_name=infra_state["python_venv_name"],
                rft_role_name=state["rft_role_name"],
            )

            if isinstance(infra.infra, EC2RFTInfrastructure):
                infra.infra.session_id = state.get("session_id", str(uuid.uuid4())[:8])
        elif platform == "ecs":
            infra = cls(
                stack_name=state["stack_name"],
                region=state["region"],
                vf_env_id=state["env_id"] if not state["is_custom_env"] else None,
                custom_env=custom_env,
                infrastructure_arn=infra_state["cluster_arn"],
                python_venv_name=infra_state["python_venv_name"],
                rft_role_name=state["rft_role_name"],
            )
        else:
            raise ValueError(f"Unknown platform: {platform}")

        infra._session_id = state.get("session_id", str(uuid.uuid4())[:8])
        infra._state_file_path = Path(file_path)

        if state["stack_outputs"]:
            infra.stack_outputs = StackOutputs(**state["stack_outputs"])
            infra._stack_exists = True

        if auto_reconnect:
            infra.infra.restore_state(infra_state)

        logger.info(f"Infrastructure state loaded from {file_path}")
        logger.info(f"Session ID: {infra._session_id}")
        return infra
