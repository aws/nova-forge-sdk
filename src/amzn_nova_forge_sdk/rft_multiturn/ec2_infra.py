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
"""EC2 platform implementation for RFT Multiturn infrastructure."""

import json
import os
import time
from typing import Dict, List, Optional

import boto3

from amzn_nova_forge_sdk.util.logging import logger
from amzn_nova_forge_sdk.validation.rft_multiturn_validator import (
    validate_amazon_linux_ami,
    validate_ec2_instance_identifier,
)

from .base_infra import (
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
)
from .common_infra_commands import BASE_PYTHON_COMMAND, CommonInfraCommands
from .constants import (
    JOB_STATUS_KILLED,
    JOB_STATUS_RUNNING,
    RFT_SAM_LOG,
    SAM_WAIT_TIME,
    SDK_RFT_LOGS_DIR,
    SDK_RFT_SCRIPTS_DIR,
    SSM_COMMAND_MAX_POLL_ATTEMPTS,
    SSM_COMMAND_POLL_INTERVAL,
    STARTER_KIT_S3,
)
from .utils import build_duplicate_job_error_message

STARTER_KIT_PATH_EC2 = "/home/ec2-user/v1"
EC2_BASE_PATH = "/home/ec2-user"
EC2_LOGS_PATH = f"{EC2_BASE_PATH}/{SDK_RFT_LOGS_DIR}"
EC2_SCRIPTS_PATH = f"{EC2_BASE_PATH}/{SDK_RFT_SCRIPTS_DIR}"


class EC2RFTInfrastructure(CommonInfraCommands, BaseRFTInfrastructure):
    """
    EC2 platform implementation
    """

    def __init__(
        self,
        region: str,
        stack_name: str,
        instance_arn: str,
        python_venv_name: str,
        rft_role_name: str,
        custom_policy_path: Optional[str] = None,
        starter_kit_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        super().__init__(
            region,
            stack_name,
            rft_role_name,
            custom_policy_path=custom_policy_path,
            starter_kit_path=starter_kit_path,
        )
        self.instance_id = self._extract_instance_id(instance_arn)
        self.python_venv_name = python_venv_name
        self.session_id = session_id or "default"
        self.ec2_client = boto3.client("ec2", region_name=region)
        self.ssm_client = boto3.client("ssm", region_name=region)

        self.base_path = STARTER_KIT_PATH_EC2
        self.starter_kit_s3 = STARTER_KIT_S3
        self.sam_base_dir = EC2_BASE_PATH

        # If starter_kit_path is an S3 URI, use it immediately
        if self.starter_kit_path and self.starter_kit_path.startswith("s3://"):
            self.starter_kit_s3 = self.starter_kit_path
            logger.info(f"Using custom starter kit from S3: {self.starter_kit_s3}")

        # Validate Amazon Linux AMI early during initialization
        validate_amazon_linux_ami(self.ec2_client, self.instance_id)

    def _get_base_logs_dir(self) -> str:
        """Get base logs directory path for EC2"""
        return EC2_LOGS_PATH

    def _get_script_file_path(self, session_id: str, env_type: EnvType) -> str:
        """Get script file path for EC2"""
        return f"{EC2_SCRIPTS_PATH}/{session_id}_{env_type.value}.sh"

    def _create_pid_file(self, session_id: str, env_type: EnvType, pid: int):
        """Create PID file on EC2 via SSM"""
        pid_file = self._get_pid_file_path(session_id, env_type)
        cmd = f"mkdir -p {EC2_LOGS_PATH} && echo {pid} > {pid_file}"
        self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
        )

    def _read_pid_file(self, session_id: str, env_type: EnvType) -> Optional[int]:
        """Read PID from PID file on EC2 via SSM"""
        pid_file = self._get_pid_file_path(session_id, env_type)
        cmd = f"cat {pid_file} 2>/dev/null || echo ''"

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
        )

        # Wait for command to complete
        command_id = response["Command"]["CommandId"]
        try:
            self._wait_for_ssm_command(command_id, timeout=10)
            output = self.ssm_client.get_command_invocation(
                CommandId=command_id, InstanceId=self.instance_id
            )
            pid_str = output["StandardOutputContent"].strip()
            return int(pid_str) if pid_str else None
        except Exception:
            return None

    def _create_status_file(self, session_id: str, env_type: EnvType, status: str):
        """Create or update status file on EC2 via SSM"""
        status_file = self._get_status_file_path(session_id, env_type)
        cmd = f"mkdir -p {EC2_LOGS_PATH} && echo {status} > {status_file}"
        self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
        )

    def _read_status_file(self, session_id: str, env_type: EnvType) -> Optional[str]:
        """Read status from status file on EC2 via SSM"""
        status_file = self._get_status_file_path(session_id, env_type)
        cmd = f"cat {status_file} 2>/dev/null || echo ''"

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
        )

        # Wait for command to complete
        command_id = response["Command"]["CommandId"]
        try:
            self._wait_for_ssm_command(command_id, timeout=10)
            output = self.ssm_client.get_command_invocation(
                CommandId=command_id, InstanceId=self.instance_id
            )
            status = output["StandardOutputContent"].strip()
            return status if status else None
        except Exception:
            return None

    def _check_for_running_jobs_on_stack(self, stack_name: str) -> Optional[str]:
        """
        Check if any jobs are running for this stack on EC2 (across all sessions).

        Args:
            stack_name: Stack name to check

        Returns:
            Error message if running jobs found, None otherwise
        """
        base_stack = self._extract_base_stack_name(stack_name)

        check_cmd = (
            f"echo 'TRAIN:'; "
            f"(pgrep -f '{stack_name}_.*_train\\.sh' 2>/dev/null || true; "
            f"pgrep -f '{base_stack}_.*_train\\.sh' 2>/dev/null || true) | sort -u || echo 'none'; "
            f"echo 'EVAL:'; "
            f"(pgrep -f '{stack_name}_.*_eval\\.sh' 2>/dev/null || true; "
            f"pgrep -f '{base_stack}_.*_eval\\.sh' 2>/dev/null || true) | sort -u || echo 'none'"
        )

        try:
            response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [check_cmd]},
            )

            command_id = response["Command"]["CommandId"]

            for attempt in range(SSM_COMMAND_MAX_POLL_ATTEMPTS):
                try:
                    result = self.ssm_client.get_command_invocation(
                        CommandId=command_id, InstanceId=self.instance_id
                    )
                    status = result.get("Status")

                    if status in ["Success", "Failed"]:
                        stdout = result.get("StandardOutputContent", "").strip()

                        train_pids = []
                        eval_pids = []

                        lines = stdout.split("\n")
                        current_type = None
                        for line in lines:
                            line = line.strip()
                            if line == "TRAIN:":
                                current_type = "train"
                            elif line == "EVAL:":
                                current_type = "eval"
                            elif line and line != "none":
                                if current_type == "train":
                                    train_pids.append(line)
                                elif current_type == "eval":
                                    eval_pids.append(line)

                        if train_pids or eval_pids:
                            train_jobs = (
                                [f"PIDs: {', '.join(train_pids)}"] if train_pids else []
                            )
                            eval_jobs = (
                                [f"PIDs: {', '.join(eval_pids)}"] if eval_pids else []
                            )

                            error_msg = build_duplicate_job_error_message(
                                stack_name,
                                train_jobs,
                                eval_jobs,
                                f"on EC2 instance {self.instance_id}",
                            )
                            logger.warning(error_msg)
                            return error_msg

                        return None
                    elif status in ["Pending", "InProgress"]:
                        time.sleep(SSM_COMMAND_POLL_INTERVAL)
                except Exception as e:
                    if "InvocationDoesNotExist" in str(e):
                        time.sleep(SSM_COMMAND_POLL_INTERVAL)
                    else:
                        raise

            return None

        except Exception as e:
            logger.warning(f"Could not check for running jobs: {e}")
            return None

    @staticmethod
    def _extract_instance_id(arn: str) -> str:
        """
        Extract instance ID from ARN or return as-is if already ID
        """
        return validate_ec2_instance_identifier(arn)

    def _get_package_install_cmd(self) -> List[str]:
        return [
            "sudo yum update -y",
            f"sudo yum install -y {BASE_PYTHON_COMMAND} {BASE_PYTHON_COMMAND}-pip git",
        ]

    def _wait_for_ssm_command(self, command_id: str, timeout: int = 300) -> None:
        """
        Wait for SSM command to complete successfully
        """
        logger.info(f"Waiting for SSM command {command_id} to complete...")
        wait_interval = 10
        elapsed = 0

        while elapsed < timeout:
            time.sleep(wait_interval)
            elapsed += wait_interval

            try:
                invocation = self.ssm_client.get_command_invocation(
                    CommandId=command_id, InstanceId=self.instance_id
                )
                status = invocation["Status"]

                if status == "Success":
                    logger.info("Environment setup completed successfully")
                    return
                elif status in ["Failed", "Cancelled", "TimedOut"]:
                    raise RuntimeError(f"SSM command failed with status: {status}")
            except self.ssm_client.exceptions.InvocationDoesNotExist:
                continue

        logger.warning(f"SSM command still running after {timeout}s, proceeding anyway")

    def _execute_training_or_eval(
        self, vf_env_id: str, script_name: str, log_file: str, command_builder
    ):
        """
        Execute training or evaluation command on EC2 instance via SSM

        Checks for duplicate processes before starting.
        Creates folders and tracking files.
        """
        # Check if ANY job is running for this stack (across all sessions)
        error_msg = self._check_for_running_jobs_on_stack(self.stack_name)
        if error_msg:
            raise RuntimeError(error_msg)

        logger.info(
            f"Setting up environment for '{vf_env_id}' on EC2 instance {self.instance_id}"
        )
        install_commands = self._build_setup_commands(vf_env_id)
        setup_response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": install_commands},
        )

        self._wait_for_ssm_command(setup_response["Command"]["CommandId"])

        cmd = command_builder()
        script_content = f"#!/bin/bash\n{cmd}"

        # Create folders and write script
        commands = [
            f"mkdir -p {EC2_LOGS_PATH}",
            f"mkdir -p {EC2_SCRIPTS_PATH}",
            f"cat > {EC2_SCRIPTS_PATH}/{script_name} << 'EOFSCRIPT'\n{script_content}\nEOFSCRIPT",
            f"chmod +x {EC2_SCRIPTS_PATH}/{script_name}",
            f"nohup {EC2_SCRIPTS_PATH}/{script_name} > {EC2_LOGS_PATH}/{log_file} 2>&1 </dev/null & echo $!",
        ]

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
        )

        # Get the PID from the command output
        command_id = response["Command"]["CommandId"]
        self._wait_for_ssm_command(command_id, timeout=30)

        try:
            output = self.ssm_client.get_command_invocation(
                CommandId=command_id, InstanceId=self.instance_id
            )
            pid_str = output["StandardOutputContent"].strip().split("\n")[-1]
            pid = int(pid_str)

            # Determine env_type from script_name
            env_type = EnvType.TRAIN if "train" in script_name else EnvType.EVAL

            # Create tracking files
            self._create_pid_file(self.session_id, env_type, pid)
            self._create_status_file(self.session_id, env_type, JOB_STATUS_RUNNING)
            logger.info(
                f"Created tracking files for session {self.session_id}, PID: {pid}"
            )
        except Exception as e:
            logger.warning(f"Could not create tracking files: {e}")

    def validate_platform(self):
        """
        Validate EC2 instance exists, is ready, and has required IAM role with RFT permissions
        """
        response = self.ec2_client.describe_instances(InstanceIds=[self.instance_id])
        if not response["Reservations"]:
            raise ValueError(f"Instance {self.instance_id} not found")

        instance = response["Reservations"][0]["Instances"][0]
        if instance["State"]["Name"] != "running":
            raise ValueError(f"Instance {self.instance_id} is not running")

        # Check SSM connectivity
        self._validate_ssm_connectivity()

        # Check if instance has an IAM instance profile
        if "IamInstanceProfile" not in instance:
            raise ValueError(
                f"Instance {self.instance_id} does not have an IAM instance profile attached. "
                f"Please attach an instance profile with an IAM role."
            )

        # Get the instance profile and ensure it has RFT permissions
        iam_client = boto3.client("iam", region_name=self.region)
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]

        instance_profile_arn = instance["IamInstanceProfile"]["Arn"]
        instance_profile_name = instance_profile_arn.split("/")[-1]

        try:
            profile = iam_client.get_instance_profile(
                InstanceProfileName=instance_profile_name
            )
            roles = profile["InstanceProfile"]["Roles"]

            if not roles:
                raise ValueError(
                    f"Instance profile {instance_profile_name} has no roles attached. "
                    f"Please attach an IAM role to the instance profile."
                )

            # Get the role and ensure it has RFT permissions
            role_name = roles[0]["RoleName"]
            logger.info(f"EC2 instance using IAM role: {role_name}")

            # Attach RFT policy to the EC2 instance role
            self._ensure_policy_on_role(role_name)

        except Exception as e:
            raise ValueError(
                f"Failed to validate or configure IAM role permissions: {e}"
            )

    def _validate_ssm_connectivity(self):
        """
        Validate that SSM agent is connected and can receive commands
        """
        try:
            response = self.ssm_client.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [self.instance_id]}]
            )

            if not response["InstanceInformationList"]:
                raise ValueError(
                    f"Instance {self.instance_id} is not registered with SSM. "
                    f"Please ensure SSM agent is installed and running."
                )

            instance_info = response["InstanceInformationList"][0]
            ping_status = instance_info.get("PingStatus")

            if ping_status != "Online":
                raise ValueError(
                    f"Instance {self.instance_id} SSM agent status is '{ping_status}'. "
                    f"Expected 'Online'. Please restart the SSM agent or reboot the instance."
                )

            logger.info(f"SSM connectivity validated for instance {self.instance_id}")

        except Exception as e:
            if "is not registered with SSM" in str(e) or "SSM agent status" in str(e):
                raise
            raise ValueError(
                f"Failed to validate SSM connectivity for instance {self.instance_id}: {e}"
            )

    def _ensure_policy_on_role(self, role_name: str):
        """
        Ensure RFT policy is attached to the specified role
        """
        self.attach_rft_policy_to_role(role_name)

    def _ensure_starter_kit_available(self, s3_bucket: Optional[str] = None):
        """
        Ensure starter kit is available for remote deployment.

        For EC2, this just handles S3 upload. Setup commands are run by:
        - deploy_sam_stack() via _build_sam_deploy_commands()
        - start_environment() via _build_setup_commands()

        Args:
            s3_bucket: S3 bucket for upload (optional, uses SageMaker default if not provided)
        """
        # Just use base implementation - no EC2-specific setup needed here
        super()._ensure_starter_kit_available(s3_bucket)

    def deploy_sam_stack(self, s3_bucket: Optional[str] = None):
        """
        Deploy SAM via EC2 instance

        Args:
            s3_bucket: S3 bucket for starter kit upload (if needed)
        """
        # Ensure starter kit is available
        self._ensure_starter_kit_available(s3_bucket)

        # Set log file for SAM deployment
        self.sam_log_file = f"{EC2_BASE_PATH}/{RFT_SAM_LOG}"

        commands = self._build_sam_deploy_commands(
            custom_starter_kit_s3=self._starter_kit_s3_uri
        )

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
        )
        command_id = response["Command"]["CommandId"]
        logger.info(f"SAM deployment initiated on EC2 instance {self.instance_id}")

        # Wait for command to complete
        start_time = time.time()

        while time.time() - start_time < SAM_WAIT_TIME:
            try:
                result = self.ssm_client.get_command_invocation(
                    CommandId=command_id, InstanceId=self.instance_id
                )
                status = result["Status"]
            except self.ssm_client.exceptions.InvocationDoesNotExist:
                continue

            if status == "Success":
                logger.info("SAM deployment completed successfully")
                return
            elif status in ["Failed", "Cancelled", "TimedOut"]:
                error_msg = result.get("StandardErrorContent", "Unknown error")
                raise RuntimeError(f"SAM deployment failed: {error_msg}")

        raise TimeoutError("SAM deployment timed out after 10 minutes")

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
        Start environment on EC2 using unified environment client.

        Args:
            env_type: Environment type (EnvType.TRAIN or EnvType.EVAL) for script and log file naming
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

        Note:
            Script names include session ID to allow multiple sessions/stacks on the same EC2 instance.
            Each session can be managed independently.
        """
        logger.warning(
            f"Starting {env_type.value} environment on EC2. "
            f"Note: Checking for duplicate processes in this session. "
            f"Multiple stacks/sessions can run simultaneously on the same instance."
        )

        def command_builder():
            return self._build_unified_client_command(
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

        # Use new naming format: {session_id}_{env_type}.sh and {session_id}_{env_type}.log
        session_id = getattr(self, "session_id", "default")
        script_name = f"{session_id}_{env_type.value}.sh"
        log_file = f"{session_id}_{env_type.value}.log"

        self._execute_training_or_eval(
            vf_env_id, script_name, log_file, command_builder
        )

        logger.info(f"Environment started on EC2 instance {self.instance_id}")

    def get_logs(
        self,
        env_type: EnvType,
        limit: int,
        start_from_head: bool,
        log_stream_name: Optional[str],
        tail: bool = False,
    ) -> list:
        """
        Get logs from EC2 via SSM for this session
        """
        # Use session-specific log file in new folder structure
        session_id = getattr(self, "session_id", "default")
        log_file = self._get_log_file_path(session_id, env_type)

        if tail:
            logger.info(
                f"Tailing {log_file} from EC2 instance {self.instance_id} (Press Ctrl+C to stop)"
            )
            try:
                last_lines = set()
                while True:
                    response = self.ssm_client.send_command(
                        InstanceIds=[self.instance_id],
                        DocumentName="AWS-RunShellScript",
                        Parameters={
                            "commands": [
                                f"tail -n 50 {log_file} 2>/dev/null || echo 'Log file not found'"
                            ]
                        },
                    )

                    # Wait for command to complete
                    command_id = response["Command"]["CommandId"]
                    self._wait_for_ssm_command(command_id)

                    output = self.ssm_client.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=self.instance_id,
                    )
                    if output["StandardOutputContent"]:
                        lines = output["StandardOutputContent"].strip().split("\n")
                        for line in lines:
                            if line not in last_lines:
                                logger.info(line)
                                last_lines.add(line)
            except KeyboardInterrupt:
                logger.info("Stopped tailing logs")
            return []

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [
                    f"tail -n {limit} {log_file} 2>/dev/null || echo 'Log file not found'"
                ]
            },
        )

        # Wait for command to complete
        command_id = response["Command"]["CommandId"]
        self._wait_for_ssm_command(command_id)

        output = self.ssm_client.get_command_invocation(
            CommandId=command_id, InstanceId=self.instance_id
        )
        return (
            output["StandardOutputContent"].strip().split("\n")
            if output["StandardOutputContent"]
            else []
        )

    def kill_task(
        self,
        env_type: EnvType,
        kill_all_for_stack: bool = False,
        preserve_logs: bool = True,
    ):
        """
        Kill training or evaluation task on EC2.

        Args:
            env_type: Type of environment (TRAIN or EVAL)
            kill_all_for_stack: If True, kills ALL jobs of this type for the stack (cross-session).
                               If False, only kills jobs from current session.
            preserve_logs: If True, keeps log files after killing jobs. If False, deletes them.
        """
        if kill_all_for_stack:
            base_stack = self._extract_base_stack_name(self.stack_name)

            logger.info(
                f"Killing ALL {env_type.value} jobs for stack '{self.stack_name}' on EC2"
            )

            if preserve_logs:
                cmd = f"""
                PIDS=$({{ pgrep -f '{self.stack_name}_.*_{env_type.value}\\.sh' 2>/dev/null || true; pgrep -f '{base_stack}_.*_{env_type.value}\\.sh' 2>/dev/null || true; }} | sort -u)
                
                if [ -n "$PIDS" ]; then
                    echo "Found PIDs: $PIDS"
                    pkill -f '{self.stack_name}_.*_{env_type.value}\\.sh' 2>/dev/null || true
                    pkill -f '{base_stack}_.*_{env_type.value}\\.sh' 2>/dev/null || true
                    
                    cd {EC2_LOGS_PATH} 2>/dev/null || exit 0
                    for file in {self.stack_name}_*_{env_type.value}.* {base_stack}_*_{env_type.value}.*; do
                        if [ -f "$file" ]; then
                            SESSION_ID=$(echo "$file" | sed 's/_{env_type.value}\\..*$//')
                            echo "{JOB_STATUS_KILLED}" > "${{SESSION_ID}}_{env_type.value}.status" 2>/dev/null || true
                        fi
                    done
                    
                    cd {EC2_SCRIPTS_PATH} 2>/dev/null || exit 0
                    rm -f {self.stack_name}_*_{env_type.value}.sh 2>/dev/null || true
                    rm -f {base_stack}_*_{env_type.value}.sh 2>/dev/null || true
                    
                    echo "Killed all {env_type.value} jobs (logs preserved)"
                else
                    echo "No {env_type.value} jobs found"
                fi
                """
            else:
                cmd = f"""
                PIDS=$({{ pgrep -f '{self.stack_name}_.*_{env_type.value}\\.sh' 2>/dev/null || true; pgrep -f '{base_stack}_.*_{env_type.value}\\.sh' 2>/dev/null || true; }} | sort -u)
                
                if [ -n "$PIDS" ]; then
                    echo "Found PIDs: $PIDS"
                    pkill -f '{self.stack_name}_.*_{env_type.value}\\.sh' 2>/dev/null || true
                    pkill -f '{base_stack}_.*_{env_type.value}\\.sh' 2>/dev/null || true
                    
                    cd {EC2_LOGS_PATH} 2>/dev/null || exit 0
                    for file in {self.stack_name}_*_{env_type.value}.* {base_stack}_*_{env_type.value}.*; do
                        if [ -f "$file" ]; then
                            SESSION_ID=$(echo "$file" | sed 's/_{env_type.value}\\..*$//')
                            echo "{JOB_STATUS_KILLED}" > "${{SESSION_ID}}_{env_type.value}.status" 2>/dev/null || true
                            rm -f "${{SESSION_ID}}_{env_type.value}.log" 2>/dev/null || true
                            rm -f "${{SESSION_ID}}_{env_type.value}.pid" 2>/dev/null || true
                            rm -f "${{SESSION_ID}}_{env_type.value}.status" 2>/dev/null || true
                        fi
                    done
                    
                    cd {EC2_SCRIPTS_PATH} 2>/dev/null || exit 0
                    rm -f {self.stack_name}_*_{env_type.value}.sh 2>/dev/null || true
                    rm -f {base_stack}_*_{env_type.value}.sh 2>/dev/null || true
                    
                    echo "Killed all {env_type.value} jobs"
                else
                    echo "No {env_type.value} jobs found"
                fi
                """

            response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [cmd]},
            )

            command_id = response["Command"]["CommandId"]
            self._wait_for_ssm_command(command_id)

            output = self.ssm_client.get_command_invocation(
                CommandId=command_id,
                InstanceId=self.instance_id,
            )

            if output.get("StandardOutputContent"):
                logger.info(f"Kill output: {output['StandardOutputContent'].strip()}")

            logger.info(f"All {env_type.value} jobs killed on EC2")
        else:
            session_id = getattr(self, "session_id", "default")
            script_path = self._get_script_file_path(session_id, env_type)
            log_file = self._get_log_file_path(session_id, env_type)
            pid_file = self._get_pid_file_path(session_id, env_type)
            status_file = self._get_status_file_path(session_id, env_type)

            script_name = f"{session_id}_{env_type.value}.sh"

            if preserve_logs:
                cmd = f"pkill -f {script_name}; echo {JOB_STATUS_KILLED} > {status_file}; rm -f {script_path} {pid_file}"
            else:
                cmd = f"pkill -f {script_name}; echo {JOB_STATUS_KILLED} > {status_file}; rm -f {script_path} {log_file} {pid_file} {status_file}"

            self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [cmd]},
            )
            logger.info(f"{env_type.value} task killed on EC2")

    def cleanup(self, cleanup_environment: bool = False):
        """
        Clean up EC2 resources: processes, logs, and optionally environment

        Args:
            cleanup_environment: If True, delete virtual environment and starter kit directories
        """
        self.kill_task(EnvType.TRAIN)
        self.kill_task(EnvType.EVAL)

        if cleanup_environment:
            # Remove virtual environment and starter kit
            commands = [
                f"rm -rf {STARTER_KIT_PATH_EC2}/{self.python_venv_name}",
                f"rm -rf {STARTER_KIT_PATH_EC2}",
                f"rm -rf {EC2_LOGS_PATH}",
                f"rm -rf {EC2_SCRIPTS_PATH}",
                f"rm -f {EC2_BASE_PATH}/{RFT_SAM_LOG}",
            ]

            # Also remove custom environment if it was used
            if (
                hasattr(self, "custom_env")
                and self.custom_env
                and self.custom_env.s3_uri
            ):
                vf_env_id = self.custom_env.env_id
                parent_path = EC2_BASE_PATH.rsplit("/", 1)[0]
                custom_env_dir = f"{parent_path}/{vf_env_id}"
                custom_env_tarball = f"/tmp/{vf_env_id}.tar.gz"
                commands.extend(
                    [
                        f"rm -rf {custom_env_dir}",
                        f"rm -f {custom_env_tarball}",
                    ]
                )

            self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": commands},
            )
            logger.info(
                f"Deleted environment and logs on EC2 instance {self.instance_id}"
            )

        logger.info(f"EC2 cleanup complete for instance {self.instance_id}")

    def get_state(self) -> Dict:
        """Get EC2 platform state for serialization"""
        return {
            "instance_id": self.instance_id,
            "python_venv_name": self.python_venv_name,
            "base_path": self.base_path,
            "starter_kit_s3": self.starter_kit_s3,
        }

    def restore_state(self, state: Dict):
        """Restore EC2 platform state after deserialization"""
        # Restore custom starter kit S3 URI if it was set
        if "starter_kit_s3" in state:
            self.starter_kit_s3 = state["starter_kit_s3"]
        # EC2 doesn't need special restoration - instance_id is already set in __init__
        # Verify instance is still running
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            if response["Reservations"]:
                status = response["Reservations"][0]["Instances"][0]["State"]["Name"]
                logger.info(f"EC2 instance {self.instance_id} status: {status}")
        except Exception as e:
            logger.warning(f"Could not verify EC2 instance: {e}")
