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
LOCAL platform implementation for RFT Multiturn infrastructure.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import boto3

from amzn_nova_customization_sdk.util.logging import logger

from .base_infra import (
    LOG_FILES,
    RFT_EVAL_LOG,
    RFT_SAM_LOG,
    RFT_TRAIN_LOG,
    STARTER_KIT_S3,
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
)
from .common_infra_commands import CommonInfraCommands


class LocalRFTInfrastructure(CommonInfraCommands, BaseRFTInfrastructure):
    """
    LOCAL platform implementation
    """

    def __init__(
        self,
        region: str,
        stack_name: str,
        workspace_dir: str,
        python_venv_name: str,
        rft_role_name: str,
        custom_policy_path: Optional[str] = None,
    ):
        super().__init__(region, stack_name, rft_role_name, custom_policy_path)
        self.workspace_dir = workspace_dir
        self.python_venv_name = python_venv_name
        self.starter_kit_path: str = ""  # Will be set in setup_local
        self.train_process: Optional[subprocess.Popen] = None
        self.eval_process: Optional[subprocess.Popen] = None

        # Set attributes needed by CommonInfraCommands
        self.base_path: str = ""  # Will be set in setup_local
        self.starter_kit_s3 = STARTER_KIT_S3
        self.python_command = sys.executable  # Use current Python for LOCAL

    def _get_package_install_cmd(self) -> List[str]:
        """
        Local doesn't need package installation commands
        """
        return []

    def validate_platform(self):
        """
        Validate local environment
        """
        pass  # No validation needed for LOCAL

    def setup_local(self, workspace_dir: str) -> str:
        """
        Setup LOCAL platform - download starter kit and validate SAM CLI
        """
        logger.info("Validating local environment and installing dependencies")

        self.starter_kit_path = os.path.join(os.path.dirname(workspace_dir), "v1")
        self.base_path = self.starter_kit_path  # Set for CommonInfraCommands
        lambda_proxy_dir = os.path.join(self.starter_kit_path, "lambda_proxy")

        if not os.path.exists(lambda_proxy_dir):
            logger.info(f"Starter kit not found at {self.starter_kit_path}")
            # Will be downloaded by install_local_environment

        self.validate_platform()
        return self.starter_kit_path

    def install_local_environment(self, vf_env_id: str):
        """
        Install amzn-agi-verifiers and environment for LOCAL platform
        """
        if not self.starter_kit_path:
            workspace_dir = os.getcwd()
            self.starter_kit_path = os.path.join(os.path.dirname(workspace_dir), "v1")
            self.base_path = self.starter_kit_path

        venv_path = os.path.join(self.base_path, self.python_venv_name)

        if os.path.exists(venv_path):
            logger.info(
                f"Using existing virtual environment for '{vf_env_id}' at {venv_path}"
            )
        else:
            logger.info(
                f"Creating virtual environment for '{vf_env_id}' at {venv_path}"
            )

        # Use common setup commands (includes downloading starter kit)
        setup_commands = self._build_setup_commands(vf_env_id, self.base_path)

        # Join all commands with && to run in single shell session
        full_command = " && ".join(cmd for cmd in setup_commands if cmd)

        try:
            result = subprocess.run(
                ["/bin/bash", "-c", full_command], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"Setup failed:\n{result.stderr}")
                raise RuntimeError(f"Environment setup failed: {result.stderr}")

            logger.info(f"Environment '{vf_env_id}' setup complete")
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(self.starter_kit_path):
                import shutil

                shutil.rmtree(self.starter_kit_path, ignore_errors=True)
                logger.warning(
                    f"Cleaned up partial installation at {self.starter_kit_path}"
                )
            raise

    def deploy_sam_stack(self):
        """
        Deploy SAM stack from local machine
        """
        # Set attributes for _build_sam_deploy_commands
        self.sam_base_dir = os.path.dirname(self.starter_kit_path)
        self.sam_log_file = os.path.join(self.workspace_dir, RFT_SAM_LOG)

        venv_path = os.path.join(self.sam_base_dir, "v1", self.python_venv_name)
        if os.path.exists(venv_path):
            logger.info(f"Using existing venv: {venv_path}")
        else:
            logger.info(
                f"Creating venv named {self.python_venv_name} at {self.sam_base_dir}/v1/"
            )

        # Get SAM deployment commands from common infra
        commands = self._build_sam_deploy_commands()

        # Join all commands with && to run in single shell session
        full_command = " && ".join(cmd for cmd in commands if cmd)

        with open(self.sam_log_file, "w") as log_f:
            result = subprocess.run(
                ["/bin/bash", "-c", full_command], capture_output=True, text=True
            )
            log_f.write(f"Command: {full_command}\n")
            log_f.write(result.stdout)
            log_f.write(result.stderr)

            if result.returncode != 0:
                raise RuntimeError(f"SAM deployment failed: {result.stderr}")

        logger.info(f"SAM deployment logs saved to: {self.sam_log_file}")

    def start_local_process(
        self, cmd: str, log_file: str, env_vars: Optional[Dict[str, str]] = None
    ) -> subprocess.Popen:
        """
        Start a local subprocess with logging
        """
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                ["/bin/bash", "-c", cmd],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        return process

    def start_training_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start training environment locally.
        """
        # Run setup commands first to ensure dependencies are installed
        setup_cmds = self._build_setup_commands(vf_env_id)
        setup_cmd = " && ".join(setup_cmds)
        logger.info("Running setup commands to ensure dependencies are installed...")
        subprocess.run(["/bin/bash", "-c", setup_cmd], check=True)

        cmd = self._build_command(
            mode="train",
            vf_env_id=vf_env_id,
            vf_env_args=vf_env_args,
            lambda_url=stack_outputs.proxy_function_url,
            queue_url=stack_outputs.rollout_request_queue_url,
            **kwargs,
        )

        log_file = os.path.join(self.workspace_dir, RFT_TRAIN_LOG)
        self.train_process = self.start_local_process(cmd, log_file, {"is_train": "1"})
        logger.info(f"Training started locally, logs: {log_file}")

    def start_evaluation_env(
        self, vf_env_id: str, vf_env_args: Dict, stack_outputs: StackOutputs, **kwargs
    ):
        """
        Start evaluation environment locally.
        """
        # Run setup commands first to ensure dependencies are installed
        setup_cmds = self._build_setup_commands(vf_env_id)
        setup_cmd = " && ".join(setup_cmds)
        logger.info("Running setup commands to ensure dependencies are installed...")
        subprocess.run(["/bin/bash", "-c", setup_cmd], check=True)

        cmd = self._build_command(
            mode="eval",
            vf_env_id=vf_env_id,
            vf_env_args=vf_env_args,
            lambda_url=stack_outputs.proxy_function_url,
            **kwargs,
        )

        log_file = os.path.join(self.workspace_dir, RFT_EVAL_LOG)
        self.eval_process = self.start_local_process(cmd, log_file)
        logger.info(f"Evaluation started locally, logs: {log_file}")

    def get_logs(
        self,
        env_type: EnvType,
        limit: int,
        start_from_head: bool,
        log_stream_name: Optional[str],
        tail: bool = False,
    ) -> list:
        """
        Get logs from local log file
        """

        log_file = os.path.join(self.workspace_dir, LOG_FILES[env_type])

        if not os.path.exists(log_file):
            if not tail:
                return []
            logger.warning(f"Log file not found: {log_file}")
            return []

        if tail:
            try:
                logger.info(f"Tailing {log_file} (Press Ctrl+C to stop)")
                subprocess.run(["tail", "-f", log_file])
            except KeyboardInterrupt:
                logger.info("Stopped tailing logs")
            return []

        with open(log_file, "r") as f:
            lines = f.readlines()

        if start_from_head:
            return lines[:limit]
        else:
            return lines[-limit:]

    def kill_task(self, env_type: EnvType):
        """
        Stop running local process and remove log file
        """
        process = self.train_process if env_type == EnvType.TRAIN else self.eval_process
        log_file = os.path.join(
            self.workspace_dir,
            RFT_TRAIN_LOG if env_type == EnvType.TRAIN else RFT_EVAL_LOG,
        )

        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            logger.info(f"{env_type.value} process terminated")
        else:
            logger.info(f"No running {env_type.value} process found")

        # Remove log file
        if os.path.exists(log_file):
            os.remove(log_file)
            logger.info(f"Removed log file: {log_file}")

    def cleanup(self, cleanup_environment: bool = False):
        """Clean up local resources

        Args:
            cleanup_environment: If True, delete virtual environment and starter kit directories
        """
        if self.train_process and self.train_process.poll() is None:
            self.kill_task(env_type=EnvType.TRAIN)
        if self.eval_process and self.eval_process.poll() is None:
            self.kill_task(env_type=EnvType.EVAL)
        logger.info("Local processes cleaned up")

        if cleanup_environment and self.starter_kit_path:
            import shutil

            # Delete virtual environment
            venv_path = os.path.join(self.base_path, self.python_venv_name)
            if os.path.exists(venv_path):
                shutil.rmtree(venv_path)
                logger.info(f"Deleted virtual environment: {venv_path}")

            # Delete starter kit
            if os.path.exists(self.starter_kit_path):
                shutil.rmtree(self.starter_kit_path)
                logger.info(f"Deleted starter kit: {self.starter_kit_path}")
