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
"""ECS platform implementation for RFT Multiturn infrastructure."""

import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_forge_sdk.util.logging import logger
from amzn_nova_forge_sdk.validation.rft_multiturn_validator import (
    validate_ecs_cluster_arn,
)

from .base_infra import (
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
    create_rft_execution_role,
)
from .common_infra_commands import CommonInfraCommands
from .constants import ECR_REPO_NAME, SAM_WAIT_TIME, STARTER_KIT_S3
from .utils import build_duplicate_job_error_message

STARTER_KIT_PATH_ECS = "/root/v1"
IMAGE_URI_FOR_TASKS = "public.ecr.aws/docker/library/python:3.12-slim"


class ECSRFTInfrastructure(CommonInfraCommands, BaseRFTInfrastructure):
    """
    ECS platform implementation
    """

    def __init__(
        self,
        region: str,
        stack_name: str,
        cluster_arn: str,
        account_id: str,
        python_venv_name: str,
        rft_role_name: str,
        custom_policy_path: Optional[str] = None,
        starter_kit_path: Optional[str] = None,
        vpc_config: Optional[Dict[str, Any]] = None,
        cpu: Optional[str] = None,
        memory: Optional[str] = None,
    ):
        """
        Initialize ECS RFT Infrastructure.

        Args:
            region: AWS region
            stack_name: CloudFormation stack name
            cluster_arn: ECS cluster ARN
            account_id: AWS account ID
            python_venv_name: Python virtual environment name
            rft_role_name: RFT execution role name
            custom_policy_path: Optional path to custom policy JSON file
            starter_kit_path: Optional custom starter kit path (local or S3 URI)
            vpc_config: Optional VPC configuration dict with keys:
                - subnets: List[str] - Subnet IDs
                - security_groups: List[str] - Security group IDs
            cpu: Optional CPU units (e.g., "2048").
            memory: Optional memory in MB (e.g., "4096").
        """
        super().__init__(
            region,
            stack_name,
            rft_role_name,
            custom_policy_path=custom_policy_path,
            starter_kit_path=starter_kit_path,
        )
        validate_ecs_cluster_arn(cluster_arn)
        self.cluster_arn = cluster_arn
        self.account_id = account_id
        self.python_venv_name = python_venv_name
        self.user_vpc_config = vpc_config
        self.user_cpu = cpu
        self.user_memory = memory
        self.ecs_client = boto3.client("ecs", region_name=region)
        self.ecr_client = boto3.client("ecr", region_name=region)
        self.ec2_client = boto3.client(
            "ec2", region_name=region
        )  # Needed for VPC discovery

        # Use self.stack_name which already has NovaSDK suffix from BaseRFTInfrastructure
        self.task_family = f"{self.stack_name}-task"
        self.log_group = f"/ecs/{self.stack_name}"

        self.ecs_roles: Optional[Dict[str, str]] = None
        self.vpc_config: Optional[Dict[str, Any]] = None
        self.latest_train_task_id: Optional[str] = None
        self.latest_eval_task_id: Optional[str] = None
        self.latest_sam_task_id: Optional[str] = None

        self.base_path = STARTER_KIT_PATH_ECS
        self.starter_kit_s3 = STARTER_KIT_S3
        self.sam_base_dir = "/root"

        # If starter_kit_path is an S3 URI, use it immediately
        if self.starter_kit_path and self.starter_kit_path.startswith("s3://"):
            self.starter_kit_s3 = self.starter_kit_path
            logger.info(f"Using custom starter kit from S3: {self.starter_kit_s3}")

    def _get_package_install_cmd(self) -> List[str]:
        return ["apt-get update", "apt-get install -y git awscli"]

    def _build_sam_deploy_command(
        self, custom_starter_kit_s3: Optional[str] = None
    ) -> List[str]:
        """
        Build SAM deployment command for ECS

        Args:
            custom_starter_kit_s3: Optional custom starter kit S3 URI
        """
        commands = self._build_sam_deploy_commands(custom_starter_kit_s3)
        return ["/bin/bash", "-c", " && ".join(commands)]

    def validate_platform(self):
        """
        Validate ECS cluster exists and is active
        """
        response = self.ecs_client.describe_clusters(clusters=[self.cluster_arn])
        if not response["clusters"]:
            raise ValueError(f"Cluster {self.cluster_arn} not found")

        cluster = response["clusters"][0]
        if cluster["status"] != "ACTIVE":
            raise ValueError(f"Cluster {self.cluster_arn} is not active")

    def get_cluster_vpc_config(self) -> Dict[str, Any]:
        """
        Get VPC configuration - use user-provided config or discover from cluster
        """
        # Use user-provided VPC config if available
        if self.user_vpc_config:
            logger.info("Using user-provided VPC configuration")
            return self.user_vpc_config

        # Otherwise, discover from cluster
        logger.info("Discovering VPC configuration from cluster")
        services = self.ecs_client.list_services(cluster=self.cluster_arn, maxResults=1)

        if services["serviceArns"]:
            service_detail = self.ecs_client.describe_services(
                cluster=self.cluster_arn, services=[services["serviceArns"][0]]
            )
            network_config = service_detail["services"][0].get(
                "networkConfiguration", {}
            )
            awsvpc_config = network_config.get("awsvpcConfiguration", {})

            return {
                "subnets": awsvpc_config.get("subnets", []),
                "security_groups": awsvpc_config.get("securityGroups", []),
            }

        vpcs = self.ec2_client.describe_vpcs(
            Filters=[{"Name": "isDefault", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError("No VPC configuration found for cluster")

        vpc_id = vpcs["Vpcs"][0]["VpcId"]
        subnets = self.ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )

        return {
            "subnets": [s["SubnetId"] for s in subnets["Subnets"][:2]],
            "security_groups": [],
        }

    def get_or_create_ecs_roles(self) -> Dict[str, str]:
        """
        Get or create ECS task role - used for both execution and task (merged)
        """
        iam_client = boto3.client("iam", region_name=self.region)

        # Use the RFT execution role name
        role_name = self.rft_role_name

        try:
            role = iam_client.get_role(RoleName=role_name)
            role_arn = role["Role"]["Arn"]
            logger.info(f"Using existing role: {role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            # Role doesn't exist, create it
            logger.info(f"Creating RFT role: {role_name}")
            role_arn = create_rft_execution_role(
                region=self.region, role_name=role_name
            )
            logger.info(f"Created role: {role_arn}")

        # Ensure the combined policy is attached using shared method
        self.attach_rft_policy_to_role(role_name)

        return {
            "execution_role_arn": role_arn,  # Same role for both
            "task_role_arn": role_arn,
        }

    def create_log_group(self):
        """
        Create CloudWatch log group
        """
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group)
            logger.info(f"Created log group: {self.log_group}")
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass

    def _setup_ecr_image(self, repo_name: str = ECR_REPO_NAME) -> str:
        """
        Setup ECR repository and push Python 3.12 image
        """
        try:
            response = self.ecr_client.describe_repositories(
                repositoryNames=[repo_name]
            )
            repository_uri = response["repositories"][0]["repositoryUri"]
            logger.info(f"ECR repository exists: {repository_uri}")
        except self.ecr_client.exceptions.RepositoryNotFoundException:
            response = self.ecr_client.create_repository(repositoryName=repo_name)
            repository_uri = response["repository"]["repositoryUri"]
            logger.info(f"Created ECR repository: {repository_uri}")

        image_uri = f"{repository_uri}:latest"

        try:
            self.ecr_client.describe_images(
                repositoryName=repo_name, imageIds=[{"imageTag": "latest"}]
            )
            logger.info(f"Image already exists in ECR: {image_uri}")
            return image_uri
        except self.ecr_client.exceptions.ImageNotFoundException:
            pass

        subprocess.run(
            ["docker", "pull", f"{IMAGE_URI_FOR_TASKS}"],
            check=True,
        )

        # Get ECR login password and pipe to docker login
        password_result = subprocess.run(
            ["aws", "ecr", "get-login-password", "--region", self.region],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "docker",
                "login",
                "--username",
                "AWS",
                "--password-stdin",
                f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com",
            ],
            input=password_result.stdout,
            text=True,
            check=True,
        )

        subprocess.run(
            [
                "docker",
                "tag",
                f"{IMAGE_URI_FOR_TASKS}",
                image_uri,
            ],
            check=True,
        )

        subprocess.run(["docker", "push", image_uri], check=True)

        logger.info(f"Successfully pushed image to {image_uri}")
        return image_uri

    def _get_s3_etag(self, s3_uri: str) -> Optional[str]:
        """
        Get ETag (hash) of S3 object to detect changes.
        """
        if not s3_uri or not s3_uri.startswith("s3://"):
            return None

        try:
            s3_client = boto3.client("s3", region_name=self.region)

            parts = s3_uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            response = s3_client.head_object(Bucket=bucket, Key=key)
            return response.get("ETag", "").strip('"')
        except Exception as e:
            logger.warning(f"Could not get S3 ETag for {s3_uri}: {e}")
            return None

    def _extract_s3_uri_from_command(
        self, command: List[str], uri_type: str = "custom_env"
    ) -> Optional[str]:
        """
        Extract S3 URI from command based on type.

        Args:
            command: Command list to extract from
            uri_type: Type of URI to extract:
                - "custom_env": Extract custom environment .tar.gz URI
                - "starter_kit": Extract starter kit URI (git clone or tarball)

        Returns:
            S3 URI if found, None otherwise
        """
        if len(command) < 3:
            return None

        cmd_string = command[2] if len(command) > 2 else ""

        if uri_type == "custom_env":
            # Look for s3:// URIs in command
            pattern = r"s3://[^\s]+"
            matches = re.findall(pattern, cmd_string)

            # Find any .tar.gz file (custom env)
            for s3_uri in matches:
                if ".tar.gz" in s3_uri:
                    return s3_uri
            return None

        elif uri_type == "starter_kit":
            # Look for git clone s3://... pattern
            git_pattern = r"git clone (s3://[^\s]+)"
            match = re.search(git_pattern, cmd_string)
            if match:
                return match.group(1)

            # Look for aws s3 cp s3://.../starter_kit.tar.gz pattern
            tarball_pattern = r"aws s3 cp (s3://[^\s]+/starter_kit\.tar\.gz)"
            match = re.search(tarball_pattern, cmd_string)
            if match:
                return match.group(1)
            return None

        return None

    def _extract_params_from_command(self, command: List[str]) -> Dict[str, str]:
        """
        Extract runtime parameters from command for comparison.

        Used to determine if an ECS task definition needs to be updated when
        runtime parameters change (e.g., max_concurrent_rollouts, timeouts).
        """
        if len(command) < 3:
            return {}

        # Command is ["/bin/bash", "-c", "actual_command_string"]
        cmd_string = command[2] if len(command) > 2 else ""

        params = {}

        # Unified client parameters (environment_client.py)
        unified_param_patterns = [
            "environment-id",
            "environment-args",
            "rollout-request-sqs-url",
            "completion-lambda-url",
            "max-concurrent-rollouts",
            "max-rollout-timeout",
            "completion-poll-interval",
            "completion-poll-timeout",
            "rollout-poll-interval",
            "aws-region",
            "log-output-directory",
            "config-name",
            "config-path",
        ]

        for param in unified_param_patterns:
            # Look for --param value or --param=value patterns
            pattern = rf"--{param}[=\s]+([^\s]+)"
            match = re.search(pattern, cmd_string)
            if match:
                params[param] = match.group(1)

        return params

    def _normalize_task_def_for_comparison(self, task_def: Dict) -> Dict:
        """
        Normalize task definition for comparison
        """
        normalized = {
            "cpu": task_def.get("cpu"),
            "memory": task_def.get("memory"),
            "executionRoleArn": task_def.get("executionRoleArn"),
            "taskRoleArn": task_def.get("taskRoleArn"),
            "networkMode": task_def.get("networkMode"),
            "requiresCompatibilities": task_def.get("requiresCompatibilities"),
        }

        # Normalize container definitions
        if "containerDefinitions" in task_def:
            container = task_def["containerDefinitions"][0]
            command = container.get("command", [])

            # Extract S3 custom env URI and get its ETag
            s3_uri = self._extract_s3_uri_from_command(command, uri_type="custom_env")
            s3_etag = self._get_s3_etag(s3_uri) if s3_uri else None

            # Extract starter kit S3 URI
            starter_kit_s3 = self._extract_s3_uri_from_command(
                command, uri_type="starter_kit"
            )

            normalized["container"] = {
                "image": container.get("image"),
                "logConfiguration": container.get("logConfiguration"),
                # Extract runtime params from command instead of comparing full command
                "runtime_params": self._extract_params_from_command(command),
                # Add S3 custom env file hash
                "custom_env_s3_etag": s3_etag,
                # Add starter kit S3 URI for comparison
                "starter_kit_s3": starter_kit_s3,
            }

        return normalized

    def _get_or_register_task_definition(self, task_def: Dict) -> str:
        """
        Get existing task definition if it matches current config, otherwise register new one.
        Compares infrastructure (CPU, memory, roles, image) AND runtime parameters.
        """
        try:
            # Get latest task definition for this family
            response = self.ecs_client.describe_task_definition(
                taskDefinition=self.task_family
            )
            existing_task_def = response["taskDefinition"]

            # Normalize both for comparison
            existing_normalized = self._normalize_task_def_for_comparison(
                existing_task_def
            )
            new_normalized = self._normalize_task_def_for_comparison(task_def)

            # Log S3 ETags for debugging
            existing_etag = existing_normalized.get("container", {}).get(
                "custom_env_s3_etag"
            )
            new_etag = new_normalized.get("container", {}).get("custom_env_s3_etag")

            # Compare all relevant fields
            if existing_normalized == new_normalized:
                task_def_arn = existing_task_def["taskDefinitionArn"]
                logger.info(f"Reusing existing task definition: {task_def_arn}")
                return task_def_arn
            else:
                logger.info(f"Task definition changed, creating new revision")

        except self.ecs_client.exceptions.ClientException:
            # Task definition family doesn't exist yet
            pass

        # Register new task definition
        response = self.ecs_client.register_task_definition(**task_def)
        task_def_arn = response["taskDefinition"]["taskDefinitionArn"]
        logger.info(f"Registered new task definition: {task_def_arn}")
        return task_def_arn

    def _run_one_time_task(
        self, env_type: EnvType, command: List[str], log_prefix: str
    ) -> Optional[str]:
        """
        Run a one-time ECS task (not a service)
        """
        if not self.ecs_roles:
            self.ecs_roles = self.get_or_create_ecs_roles()
        if not self.vpc_config:
            self.vpc_config = self.get_cluster_vpc_config()

        self.create_log_group()
        ecr_image = self._setup_ecr_image()

        # Use user-provided CPU/memory or defaults
        cpu = self.user_cpu or "2048"
        memory = self.user_memory or "4096"

        task_def = {
            "family": self.task_family,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": cpu,
            "memory": memory,
            "executionRoleArn": self.ecs_roles["execution_role_arn"],
            "taskRoleArn": self.ecs_roles["task_role_arn"],
            "containerDefinitions": [
                {
                    "name": f"{self.stack_name}-container",
                    "image": ecr_image,
                    "essential": True,
                    "command": command,
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": self.log_group,
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": log_prefix,
                        },
                    },
                }
            ],
        }

        task_def_arn = self._get_or_register_task_definition(task_def)

        run_response = self.ecs_client.run_task(
            cluster=self.cluster_arn,
            taskDefinition=task_def_arn,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self.vpc_config["subnets"],
                    "securityGroups": self.vpc_config["security_groups"],
                    "assignPublicIp": "ENABLED",
                }
            },
        )

        if run_response["tasks"]:
            task_id = run_response["tasks"][0]["taskArn"].split("/")[-1]

            # Track task ID for log retrieval
            if env_type == EnvType.TRAIN:
                self.latest_train_task_id = task_id
            elif env_type == EnvType.EVAL:
                self.latest_eval_task_id = task_id
            elif env_type == EnvType.SAM:
                self.latest_sam_task_id = task_id

            logger.info(
                f"{env_type.value.capitalize()} task started with ID: {task_id}"
            )
            return task_id

        logger.warning(f"Failed to start {env_type.value} task")
        return None

    def deploy_sam_stack(self, s3_bucket: Optional[str] = None):
        """
        Deploy SAM stack via ECS

        Args:
            s3_bucket: S3 bucket for starter kit upload (if needed)
        """
        # Ensure starter kit is available
        self._ensure_starter_kit_available(s3_bucket)

        command = self._build_sam_deploy_command(
            custom_starter_kit_s3=self._starter_kit_s3_uri
        )
        task_id = self._run_one_time_task(EnvType.SAM, command, "sam-deploy")
        if task_id:
            self.latest_sam_task_id = task_id
            self.wait_for_sam_task(task_id, self.cluster_arn, self.region)

    def wait_for_sam_task(self, task_id: str, cluster_arn: str, region: str):
        """
        Wait for SAM deployment task to complete
        """
        task_arn = f"arn:aws:ecs:{region}:{self.account_id}:task/{cluster_arn.split('/')[-1]}/{task_id}"

        logger.info(f"Waiting for SAM deployment task {task_id} to complete...")
        start_time = time.time()

        while time.time() - start_time < SAM_WAIT_TIME:
            response = self.ecs_client.describe_tasks(
                cluster=cluster_arn, tasks=[task_arn]
            )

            if not response["tasks"]:
                raise RuntimeError(f"Task {task_id} not found")

            task = response["tasks"][0]
            status = task["lastStatus"]

            if status == "STOPPED":
                exit_code = task["containers"][0].get("exitCode", 1)
                if exit_code == 0:
                    logger.info("SAM deployment completed successfully")
                    return
                else:
                    raise RuntimeError(
                        f"SAM deployment failed with exit code {exit_code}"
                    )

            time.sleep(10)

        raise TimeoutError("SAM deployment timed out after 10 minutes")

    def _is_task_running(self, task_id: str) -> bool:
        """
        Check if a task is currently running.

        Args:
            task_id: ECS task ID to check

        Returns:
            True if task is running, False otherwise
        """
        if not task_id:
            return False

        try:
            response = self.ecs_client.describe_tasks(
                cluster=self.cluster_arn, tasks=[task_id]
            )

            if not response.get("tasks"):
                return False

            task = response["tasks"][0]
            last_status = task.get("lastStatus", "")
            return last_status == "RUNNING"

        except Exception as e:
            logger.debug(f"Error checking task status: {e}")
            return False

    def _check_for_running_jobs_on_stack(self, stack_name: str) -> Optional[str]:
        """
        Check if any jobs are running for this stack on ECS (across all sessions).

        Args:
            stack_name: Stack name to check

        Returns:
            Error message if running jobs found, None otherwise
        """
        try:
            base_stack = self._extract_base_stack_name(stack_name)

            response = self.ecs_client.list_tasks(
                cluster=self.cluster_arn, desiredStatus="RUNNING"
            )

            if not response.get("taskArns"):
                return None

            tasks_response = self.ecs_client.describe_tasks(
                cluster=self.cluster_arn, tasks=response["taskArns"]
            )

            train_jobs = []
            eval_jobs = []

            for task in tasks_response.get("tasks", []):
                task_def_arn = task.get("taskDefinitionArn", "")
                task_arn = task.get("taskArn", "")
                task_id = task_arn.split("/")[-1]

                if base_stack not in task_def_arn:
                    continue

                try:
                    task_def_response = self.ecs_client.describe_task_definition(
                        taskDefinition=task_def_arn
                    )
                    task_def = task_def_response.get("taskDefinition", {})
                    container_defs = task_def.get("containerDefinitions", [])

                    for container_def in container_defs:
                        log_config = container_def.get("logConfiguration", {})
                        log_options = log_config.get("options", {})
                        log_prefix = log_options.get("awslogs-stream-prefix", "")

                        if "training" in log_prefix.lower():
                            train_jobs.append(f"task: {task_id}")
                            break
                        elif "evaluation" in log_prefix.lower():
                            eval_jobs.append(f"task: {task_id}")
                            break

                except Exception as e:
                    logger.debug(f"Error fetching task definition for {task_id}: {e}")

            if train_jobs or eval_jobs:
                error_msg = build_duplicate_job_error_message(
                    stack_name, train_jobs, eval_jobs, "on ECS cluster"
                )
                logger.warning(error_msg)
                return error_msg

            return None

        except Exception as e:
            logger.warning(f"Could not check for running jobs on ECS: {e}")
            return None

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
        Start environment on ECS using unified environment client.

        Args:
            env_type: Environment type (EnvType.TRAIN or EnvType.EVAL) for task naming
            vf_env_id: Verifier environment identifier
            vf_env_args: Environment arguments
            stack_outputs: Stack outputs containing URLs and queue information
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
            completion_poll_timeout: Completion polling timeout (default: 600)
            completion_poll_interval: Completion poll interval (default: 0.5)
            rollout_poll_interval: SQS polling interval (default: 1.0)
            log_output_directory: Directory for logs and metrics (optional)
            config_name: Use YAML config instead of CLI flags (optional)
            config_path: Custom config directory path (optional, use with config_name)

        Raises:
            RuntimeError: If any task (train or eval) is already running for this stack
        """
        # Check if ANY job is running for this stack (across all sessions)
        error_msg = self._check_for_running_jobs_on_stack(self.stack_name)
        if error_msg:
            raise RuntimeError(error_msg)

        # Build unified client command
        command = self._build_container_command(
            vf_env_id=vf_env_id,
            vf_env_args=vf_env_args,
            lambda_url=stack_outputs.proxy_function_url,
            queue_url=stack_outputs.rollout_request_queue_url,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_rollout_timeout=max_rollout_timeout,
            completion_poll_timeout=completion_poll_timeout,
            completion_poll_interval=completion_poll_interval,
            rollout_poll_interval=rollout_poll_interval,
            log_output_directory=log_output_directory,
            config_name=config_name,
            config_path=config_path,
        )

        # Use appropriate task name based on env_type
        task_name = "training" if env_type == EnvType.TRAIN else "evaluation"
        self._run_one_time_task(env_type, command, task_name)

    def _build_container_command(
        self,
        vf_env_id: str,
        vf_env_args: Dict,
        lambda_url: str,
        queue_url: str,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> List[str]:
        """
        Build container command for unified client.

        Downloads and sets up the environment at container startup,
        then runs the unified client.

        Returns command as list for ECS task definition.
        """
        # First, run setup commands to download starter kit and install environment
        base_setup = self._build_setup_commands(vf_env_id)

        # Then build the unified client command
        env_cmd = self._build_unified_client_command(
            vf_env_id=vf_env_id,
            vf_env_args=vf_env_args,
            lambda_url=lambda_url,
            queue_url=queue_url,
            max_concurrent_rollouts=max_concurrent_rollouts,
            max_rollout_timeout=max_rollout_timeout,
            completion_poll_timeout=completion_poll_timeout,
            completion_poll_interval=completion_poll_interval,
            rollout_poll_interval=rollout_poll_interval,
            log_output_directory=log_output_directory,
            config_name=config_name,
            config_path=config_path,
        )

        # Replace newlines with && for proper bash chaining
        env_cmd = env_cmd.replace("\n", " && ")
        base_setup.append(env_cmd)

        # Return as list for ECS container command
        return ["/bin/bash", "-c", " && ".join(base_setup)]

    def get_logs(
        self,
        env_type: EnvType,
        limit: int,
        start_from_head: bool,
        log_stream_name: Optional[str],
        tail: bool = False,
    ) -> list:
        """
        Get logs from ECS
        """
        if log_stream_name:
            stream_name = log_stream_name
            logger.info(f"Reading logs from specified stream: {stream_name}")
        else:
            prefixes = {
                EnvType.TRAIN: "training",
                EnvType.EVAL: "evaluation",
                EnvType.SAM: "sam-deploy",
            }
            prefix = prefixes[env_type]

            tracked_task_id = None
            if env_type == EnvType.TRAIN:
                tracked_task_id = self.latest_train_task_id
            elif env_type == EnvType.EVAL:
                tracked_task_id = self.latest_eval_task_id
            elif env_type == EnvType.SAM:
                tracked_task_id = self.latest_sam_task_id

            if not tracked_task_id:
                logger.info(f"No {env_type.value} task started in current session")
                return []

            # All containers use the same naming pattern
            container_name = f"{self.stack_name}-container"
            stream_name = f"{prefix}/{container_name}/{tracked_task_id}"
            logger.info(f"Reading logs from stream: {stream_name}")

        if tail:
            logger.info(
                f"Tailing CloudWatch logs from {stream_name} (Press Ctrl+C to stop)"
            )
            try:
                next_token = None
                while True:
                    kwargs = {
                        "logGroupName": self.log_group,
                        "logStreamName": stream_name,
                        "limit": 50,
                        "startFromHead": False,
                    }
                    if next_token:
                        kwargs["nextToken"] = next_token

                    try:
                        response = self.logs_client.get_log_events(**kwargs)
                        for event in response.get("events", []):
                            logger.info(event["message"])
                        next_token = response.get("nextForwardToken")
                    except self.logs_client.exceptions.ResourceNotFoundException:
                        logger.warning(f"Log stream not found: {stream_name}")
                        time.sleep(2)
                        continue

                    time.sleep(2)
            except KeyboardInterrupt:
                logger.info("Stopped tailing logs")
            return []

        try:
            response = self.logs_client.get_log_events(
                logGroupName=self.log_group,
                logStreamName=stream_name,
                limit=limit,
                startFromHead=start_from_head,
            )
            return [event["message"] for event in response.get("events", [])]
        except self.logs_client.exceptions.ResourceNotFoundException:
            logger.warning(f"Log stream not found: {stream_name}")
            return []

    def list_log_streams_ecs(self, env_type: Optional[EnvType] = None) -> list[str]:
        """
        List log streams in the log group
        """
        kwargs = {"logGroupName": self.log_group, "descending": True, "limit": 50}

        if env_type:
            prefixes = {
                EnvType.TRAIN: "training",
                EnvType.EVAL: "evaluation",
                EnvType.SAM: "sam-deploy",
            }
            kwargs["logStreamNamePrefix"] = prefixes[env_type]
        else:
            kwargs["orderBy"] = "LastEventTime"

        response = self.logs_client.describe_log_streams(**kwargs)
        return [stream["logStreamName"] for stream in response.get("logStreams", [])]

    def kill_task(
        self,
        env_type: EnvType,
        task_id: Optional[str] = None,
        deregister_task_def: bool = False,
        kill_all_for_stack: bool = False,
    ):
        """
        Stop running ECS task and optionally deregister its task definition.

        Args:
            env_type: Type of task to kill
            task_id: Optional task ID if session was lost
            deregister_task_def: If True, deregister the task definition after stopping,
            the task definition can be deleted after waiting 10 minutes post-deregistration
            kill_all_for_stack: If True, kills ALL tasks of this type for the stack (cross-session).
                               If False, only kills task from current session.
        """
        if kill_all_for_stack:
            base_stack = self._extract_base_stack_name(self.stack_name)
            logger.info(
                f"Killing ALL {env_type.value} tasks for stack '{self.stack_name}' on ECS"
            )

            try:
                response = self.ecs_client.list_tasks(
                    cluster=self.cluster_arn, desiredStatus="RUNNING"
                )

                if not response.get("taskArns"):
                    logger.info("No running tasks found")
                    return

                tasks_response = self.ecs_client.describe_tasks(
                    cluster=self.cluster_arn, tasks=response["taskArns"]
                )

                killed_count = 0
                env_type_str = "training" if env_type == EnvType.TRAIN else "evaluation"

                for task in tasks_response.get("tasks", []):
                    task_def_arn = task.get("taskDefinitionArn", "")
                    task_arn = task["taskArn"]
                    task_id = task_arn.split("/")[-1]

                    if base_stack not in task_def_arn:
                        continue

                    try:
                        task_def_response = self.ecs_client.describe_task_definition(
                            taskDefinition=task_def_arn
                        )
                        task_def = task_def_response.get("taskDefinition", {})
                        container_defs = task_def.get("containerDefinitions", [])

                        matches_env_type = False
                        for container_def in container_defs:
                            log_config = container_def.get("logConfiguration", {})
                            log_options = log_config.get("options", {})
                            log_prefix = log_options.get("awslogs-stream-prefix", "")

                            if env_type_str in log_prefix.lower():
                                matches_env_type = True
                                break

                        if not matches_env_type:
                            continue

                        self.ecs_client.stop_task(
                            cluster=self.cluster_arn,
                            task=task_arn,
                            reason=f"Stopped by NovaCustomizationSDK",
                        )
                        killed_count += 1

                        if deregister_task_def:
                            try:
                                self.ecs_client.deregister_task_definition(
                                    taskDefinition=task_def_arn
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not deregister task definition: {e}"
                                )

                    except Exception as e:
                        logger.warning(f"Could not process task {task_id}: {e}")

                logger.info(f"Killed {killed_count} {env_type.value} task(s)")

            except Exception as e:
                logger.warning(f"Error killing tasks: {e}")

            return

        # Current session mode: Kill only task from this session
        # Use provided task_id or fall back to cached ID
        if not task_id:
            if env_type == EnvType.TRAIN:
                task_id = self.latest_train_task_id
            elif env_type == EnvType.EVAL:
                task_id = self.latest_eval_task_id

        if not task_id:
            logger.warning(
                f"No {env_type.value} task found from current session. Provide task_id if you wish to kill specific task"
            )
            return

        task_arn = f"arn:aws:ecs:{self.region}:{self.account_id}:task/{self.cluster_arn.split('/')[-1]}/{task_id}"
        task_def_arn = None

        try:
            # Get task definition ARN before stopping
            if deregister_task_def:
                response = self.ecs_client.describe_tasks(
                    cluster=self.cluster_arn, tasks=[task_arn]
                )
                if response.get("tasks"):
                    task_def_arn = response["tasks"][0].get("taskDefinitionArn")

            # Stop the task
            self.ecs_client.stop_task(
                cluster=self.cluster_arn,
                task=task_arn,
                reason=f"Stopped by NovaCustomizationSDK cleanup",
            )
            logger.info(f"Stopped {env_type.value} task: {task_id}")

            # Deregister task definition if requested
            if deregister_task_def and task_def_arn:
                try:
                    self.ecs_client.deregister_task_definition(
                        taskDefinition=task_def_arn
                    )
                    logger.info(f"Deregistered task definition: {task_def_arn}")
                except Exception as e:
                    logger.warning(
                        f"Could not deregister task definition {task_def_arn}: {e}"
                    )

        except self.ecs_client.exceptions.ClientException as e:
            logger.warning(f"Could not stop task {task_id}: {e}")

    def cleanup(self, cleanup_environment: bool = False):
        """
        Stop running ECS tasks and optionally clean up environment.

        Args:
            cleanup_environment: If True, deregister task definitions (ECS environment cleanup)
        """
        for env_type in [EnvType.TRAIN, EnvType.EVAL]:
            self.kill_task(env_type, deregister_task_def=cleanup_environment)
        logger.info("ECS cleanup complete")

    def get_state(self) -> Dict:
        """Get ECS platform state for serialization"""
        return {
            "cluster_arn": self.cluster_arn,
            "account_id": self.account_id,
            "python_venv_name": self.python_venv_name,
            "latest_train_task_id": self.latest_train_task_id,
            "latest_eval_task_id": self.latest_eval_task_id,
            "latest_sam_task_id": self.latest_sam_task_id,
            "starter_kit_s3": self.starter_kit_s3,
        }

    def restore_state(self, state: Dict):
        """Restore ECS platform state after deserialization"""
        self.latest_train_task_id = state.get("latest_train_task_id")
        self.latest_eval_task_id = state.get("latest_eval_task_id")
        self.latest_sam_task_id = state.get("latest_sam_task_id")
        # Restore custom starter kit S3 URI if it was set
        if "starter_kit_s3" in state:
            self.starter_kit_s3 = state["starter_kit_s3"]

        # Verify tasks are still running
        for task_id, env_type in [
            (self.latest_train_task_id, "train"),
            (self.latest_eval_task_id, "eval"),
            (self.latest_sam_task_id, "sam"),
        ]:
            if task_id:
                try:
                    response = self.ecs_client.describe_tasks(
                        cluster=self.cluster_arn, tasks=[task_id]
                    )
                    if response["tasks"]:
                        status = response["tasks"][0]["lastStatus"]
                        logger.info(f"ECS {env_type} task {task_id} status: {status}")
                except Exception as e:
                    logger.warning(f"Could not verify ECS {env_type} task: {e}")
