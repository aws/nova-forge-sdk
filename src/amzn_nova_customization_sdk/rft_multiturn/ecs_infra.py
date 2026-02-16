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

import subprocess
import time
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
    validate_ecs_cluster_arn,
)

from .base_infra import (
    ECR_REPO_NAME,
    SAM_WAIT_TIME,
    STARTER_KIT_S3,
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
    create_rft_execution_role,
)
from .common_infra_commands import CommonInfraCommands

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
            vpc_config: Optional VPC configuration dict with keys:
                - subnets: List[str] - Subnet IDs
                - security_groups: List[str] - Security group IDs
            cpu: Optional CPU units (e.g., "2048").
            memory: Optional memory in MB (e.g., "4096").
        """
        super().__init__(region, stack_name, rft_role_name, custom_policy_path)
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

    def _get_package_install_cmd(self) -> List[str]:
        return ["apt-get update", "apt-get install -y git awscli"]

    def _build_container_command(
        self,
        environment_type: str,
        vf_env_id: str,
        vf_env_args: Dict,
        lambda_url: str,
        queue_url: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Build container startup command for ECS
        """
        base_setup = self._build_setup_commands(vf_env_id)

        mode = "eval" if environment_type == "eval" else "train"
        env_cmd = self._build_command(
            mode, vf_env_id, vf_env_args, lambda_url, queue_url, **kwargs
        )

        # Replace newlines with && for proper bash chaining
        env_cmd = env_cmd.replace("\n", " && ")
        base_setup.append(env_cmd)
        return ["/bin/bash", "-c", " && ".join(base_setup)]

    def _build_sam_deploy_command(self) -> List[str]:
        """
        Build SAM deployment command for ECS
        """
        commands = self._build_sam_deploy_commands()
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
            import boto3

            s3_client = boto3.client("s3", region_name=self.region)

            parts = s3_uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            response = s3_client.head_object(Bucket=bucket, Key=key)
            return response.get("ETag", "").strip('"')
        except Exception as e:
            logger.warning(f"Could not get S3 ETag for {s3_uri}: {e}")
            return None

    def _extract_s3_uri_from_command(self, command: List[str]) -> Optional[str]:
        """
        Extract custom env S3 URI from command.
        """
        if len(command) < 3:
            return None

        cmd_string = command[2] if len(command) > 2 else ""

        # Look for s3:// URIs in command
        import re

        pattern = r"s3://[^\s]+"
        matches = re.findall(pattern, cmd_string)

        # Find any .tar.gz file (custom env)
        for s3_uri in matches:
            if ".tar.gz" in s3_uri:
                return s3_uri

        return None

    def _extract_params_from_command(self, command: List[str]) -> Dict[str, str]:
        """
        Extract runtime parameters from command for comparison
        """
        if len(command) < 3:
            return {}

        # Command is ["/bin/bash", "-c", "actual_command_string"]
        cmd_string = command[2] if len(command) > 2 else ""

        params = {}
        # Extract key parameters from command string
        param_patterns = [
            "client_poll_interval",
            "client_timeout",
            "max_messages_per_poll",
            "groups_per_batch",
            "max_concurrent_batches",
            "max_workers",
            "num_examples",
            "rollouts_per_example",
            "max_concurrent",
        ]

        for param in param_patterns:
            # Look for --param value or --param=value patterns
            import re

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
            s3_uri = self._extract_s3_uri_from_command(command)
            s3_etag = self._get_s3_etag(s3_uri) if s3_uri else None

            normalized["container"] = {
                "image": container.get("image"),
                "logConfiguration": container.get("logConfiguration"),
                # Extract runtime params from command instead of comparing full command
                "runtime_params": self._extract_params_from_command(command),
                # Add S3 custom env file hash
                "custom_env_s3_etag": s3_etag,
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

    def deploy_sam_stack(self):
        """
        Deploy SAM stack via ECS
        """
        command = self._build_sam_deploy_command()
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

    def start_training_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start training environment on ECS
        """
        command = self._build_container_command(
            "train",
            vf_env_id,
            vf_env_args,
            stack_outputs.proxy_function_url,
            stack_outputs.rollout_request_queue_url,
            **kwargs,
        )
        self._run_one_time_task(EnvType.TRAIN, command, "training")

    def start_evaluation_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start evaluation environment on ECS
        """
        command = self._build_container_command(
            "eval",
            vf_env_id,
            vf_env_args,
            stack_outputs.proxy_function_url,
            None,
            **kwargs,
        )
        self._run_one_time_task(EnvType.EVAL, command, "evaluation")

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
    ):
        """
        Stop running ECS task and optionally deregister its task definition.

        Args:
            env_type: Type of task to kill
            task_id: Optional task ID if session was lost
            deregister_task_def: If True, deregister the task definition after stopping,
            the task definition can be deleted after waiting 10 minutes post-deregistration
        """
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
