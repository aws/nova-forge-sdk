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
import time
from typing import Dict, List, Optional

import boto3

from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
    validate_ec2_instance_identifier,
)

from .base_infra import (
    LOG_FILES,
    RFT_EVAL_LOG,
    RFT_SAM_LOG,
    RFT_TRAIN_LOG,
    SAM_WAIT_TIME,
    STARTER_KIT_S3,
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
)
from .common_infra_commands import BASE_PYTHON_COMMAND, CommonInfraCommands

STARTER_KIT_PATH_EC2 = "/home/ec2-user/v1"
EC2_BASE_PATH = "/home/ec2-user"


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
    ):
        super().__init__(region, stack_name, rft_role_name, custom_policy_path)
        self.instance_id = self._extract_instance_id(instance_arn)
        self.python_venv_name = python_venv_name
        self.ec2_client = boto3.client("ec2", region_name=region)
        self.ssm_client = boto3.client("ssm", region_name=region)

        self.base_path = STARTER_KIT_PATH_EC2
        self.starter_kit_s3 = STARTER_KIT_S3
        self.sam_base_dir = EC2_BASE_PATH

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
        """
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

        commands = [
            f"cat > {EC2_BASE_PATH}/{script_name} << 'EOFSCRIPT'\n{script_content}\nEOFSCRIPT",
            f"chmod +x {EC2_BASE_PATH}/{script_name}",
            f"nohup {EC2_BASE_PATH}/{script_name} > {EC2_BASE_PATH}/{log_file} 2>&1 </dev/null & disown",
        ]

        self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
        )

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

    def deploy_sam_stack(self):
        """
        Deploy SAM via EC2 instance
        """
        # Set log file for SAM deployment
        self.sam_log_file = f"{EC2_BASE_PATH}/{RFT_SAM_LOG}"

        commands = self._build_sam_deploy_commands()

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
        )
        command_id = response["Command"]["CommandId"]
        logger.info(f"SAM deployment initiated on EC2 instance {self.instance_id}")

        # Wait for command to complete
        start_time = time.time()
        time.sleep(5)  # Initial delay for command to register

        while time.time() - start_time < SAM_WAIT_TIME:
            try:
                result = self.ssm_client.get_command_invocation(
                    CommandId=command_id, InstanceId=self.instance_id
                )
                status = result["Status"]
            except self.ssm_client.exceptions.InvocationDoesNotExist:
                time.sleep(5)
                continue

            if status == "Success":
                logger.info("SAM deployment completed successfully")
                return
            elif status in ["Failed", "Cancelled", "TimedOut"]:
                error_msg = result.get("StandardErrorContent", "Unknown error")
                raise RuntimeError(f"SAM deployment failed: {error_msg}")

            time.sleep(10)

        raise TimeoutError("SAM deployment timed out after 10 minutes")

    def start_training_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start training environment on EC2
        """

        def command_builder():
            return self._build_command(
                "train",
                vf_env_id,
                vf_env_args,
                stack_outputs.proxy_function_url,
                stack_outputs.rollout_request_queue_url,
                **kwargs,
            )

        time.sleep(10)  # added sleep for environment to be created
        self._execute_training_or_eval(
            vf_env_id, "train_script.sh", RFT_TRAIN_LOG, command_builder
        )
        logger.info(f"Training started on EC2 instance {self.instance_id}")

    def start_evaluation_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start evaluation environment on EC2
        """

        def command_builder():
            return self._build_command(
                "eval",
                vf_env_id,
                vf_env_args,
                stack_outputs.proxy_function_url,
                **kwargs,
            )

        self._execute_training_or_eval(
            vf_env_id, "eval_script.sh", RFT_EVAL_LOG, command_builder
        )
        logger.info(f"Evaluation started on EC2 instance {self.instance_id}")

    def get_logs(
        self,
        env_type: EnvType,
        limit: int,
        start_from_head: bool,
        log_stream_name: Optional[str],
        tail: bool = False,
    ) -> list:
        """
        Get logs from EC2 via SSM
        """

        log_file = LOG_FILES[env_type]

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
                                f"tail -n 50 {EC2_BASE_PATH}/{log_file} 2>/dev/null || echo 'Log file not found'"
                            ]
                        },
                    )
                    time.sleep(2)
                    output = self.ssm_client.get_command_invocation(
                        CommandId=response["Command"]["CommandId"],
                        InstanceId=self.instance_id,
                    )
                    if output["StandardOutputContent"]:
                        lines = output["StandardOutputContent"].strip().split("\n")
                        for line in lines:
                            if line not in last_lines:
                                logger.info(line)
                                last_lines.add(line)
                    time.sleep(3)
            except KeyboardInterrupt:
                logger.info("Stopped tailing logs")
            return []

        response = self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [
                    f"tail -n {limit} {EC2_BASE_PATH}/{log_file} 2>/dev/null || echo 'Log file not found'"
                ]
            },
        )
        time.sleep(2)
        output = self.ssm_client.get_command_invocation(
            CommandId=response["Command"]["CommandId"], InstanceId=self.instance_id
        )
        return (
            output["StandardOutputContent"].strip().split("\n")
            if output["StandardOutputContent"]
            else []
        )

    def kill_task(self, env_type: EnvType):
        """
        Kill training or evaluation task on EC2
        """
        process_names = {EnvType.TRAIN: "train.py", EnvType.EVAL: "evaluate.py"}
        log_files_paths = {
            EnvType.TRAIN: f"{EC2_BASE_PATH}/{RFT_TRAIN_LOG}",
            EnvType.EVAL: f"{EC2_BASE_PATH}/{RFT_EVAL_LOG}",
        }
        process_name = process_names[env_type]
        log_file = log_files_paths[env_type]

        cmd = f"pkill -f {process_name}; rm -f {log_file}"
        self.ssm_client.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
        )
        logger.info(f"{env_type.value} task killed on EC2 instance {self.instance_id}")

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
                f"rm -f {EC2_BASE_PATH}/{RFT_TRAIN_LOG}",
                f"rm -f {EC2_BASE_PATH}/{RFT_EVAL_LOG}",
                f"rm -f {EC2_BASE_PATH}/{RFT_SAM_LOG}",
            ]
            self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": commands},
            )
            logger.info(
                f"Deleted environment and logs on EC2 instance {self.instance_id}"
            )

        logger.info(f"EC2 cleanup complete for instance {self.instance_id}")
