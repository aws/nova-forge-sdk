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
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import boto3

from amzn_nova_forge_sdk.util.logging import logger

from .base_infra import (
    BaseRFTInfrastructure,
    EnvType,
    StackOutputs,
)
from .common_infra_commands import CommonInfraCommands
from .constants import (
    JOB_STATUS_KILLED,
    JOB_STATUS_RUNNING,
    RFT_SAM_LOG,
    SDK_RFT_LOGS_DIR,
    STARTER_KIT_S3,
)
from .utils import build_duplicate_job_error_message, validate_starter_kit_path


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
        self.workspace_dir = workspace_dir
        self.python_venv_name = python_venv_name
        self.session_id = session_id or "default"
        self.starter_kit_path_attr: str = ""  # Will be set in setup_local
        self.train_process: Optional[subprocess.Popen] = None
        self.eval_process: Optional[subprocess.Popen] = None

        # Set attributes needed by CommonInfraCommands
        self.base_path: str = ""  # Will be set in setup_local
        self.starter_kit_s3 = STARTER_KIT_S3
        self.python_command = sys.executable  # Use current Python for LOCAL

    def _get_base_logs_dir(self) -> str:
        """Get base logs directory path for Local"""
        return os.path.join(self.workspace_dir, SDK_RFT_LOGS_DIR)

    def _get_logs_dir(self) -> str:
        """Get logs directory path for Local (alias for backward compatibility)"""
        return self._get_base_logs_dir()

    def _create_pid_file(self, session_id: str, env_type: EnvType, pid: int):
        """Create PID file for Local"""
        os.makedirs(self._get_logs_dir(), exist_ok=True)
        pid_file = self._get_pid_file_path(session_id, env_type)
        with open(pid_file, "w") as f:
            f.write(str(pid))

    def _read_pid_file(self, session_id: str, env_type: EnvType) -> Optional[int]:
        """Read PID from PID file for Local"""
        pid_file = self._get_pid_file_path(session_id, env_type)
        if os.path.exists(pid_file):
            try:
                with open(pid_file, "r") as f:
                    return int(f.read().strip())
            except Exception:
                return None
        return None

    def _create_status_file(self, session_id: str, env_type: EnvType, status: str):
        """Create or update status file for Local"""
        os.makedirs(self._get_logs_dir(), exist_ok=True)
        status_file = self._get_status_file_path(session_id, env_type)
        with open(status_file, "w") as f:
            f.write(status)

    def _read_status_file(self, session_id: str, env_type: EnvType) -> Optional[str]:
        """Read status from status file for Local"""
        status_file = self._get_status_file_path(session_id, env_type)
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    return f.read().strip()
            except Exception:
                return None
        return None

    def _check_for_running_jobs_on_stack(self, stack_name: str) -> Optional[str]:
        """
        Check if any jobs are running for this stack locally (across all sessions).

        Args:
            stack_name: Stack name to check

        Returns:
            Error message if running jobs found, None otherwise
        """
        logs_dir = self._get_logs_dir()
        if not os.path.exists(logs_dir):
            return None

        base_stack = self._extract_base_stack_name(stack_name)
        train_jobs = []
        eval_jobs = []

        for filename in os.listdir(logs_dir):
            if filename.endswith(".pid") and base_stack in filename:
                pid_file = os.path.join(logs_dir, filename)
                try:
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())

                    if self._is_process_running(pid):
                        parts = filename.replace(".pid", "").rsplit("_", 1)
                        if len(parts) == 2:
                            session_id, env_type = parts
                            job_info = f"session: {session_id}, PID: {pid}"
                            if env_type == "train":
                                train_jobs.append(job_info)
                            elif env_type == "eval":
                                eval_jobs.append(job_info)
                except Exception:
                    continue

        if train_jobs or eval_jobs:
            error_msg = build_duplicate_job_error_message(
                stack_name, train_jobs, eval_jobs
            )
            logger.warning(error_msg)
            return error_msg

        return None

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

        # Handle custom starter_kit_path if provided
        if self.starter_kit_path:
            # User provided a custom path - validate it exists
            if self.starter_kit_path.startswith("s3://"):
                raise ValueError(
                    "starter_kit_path for LOCAL platform must be a local path, not S3 URI"
                )

            expanded_path = os.path.abspath(os.path.expanduser(self.starter_kit_path))

            # Validate the path exists
            if not os.path.exists(expanded_path):
                raise ValueError(
                    f"Starter kit path does not exist: {expanded_path}\n"
                    f"Please provide a valid path to an existing starter kit directory."
                )

            # Validate it's a valid starter kit
            validate_starter_kit_path(expanded_path)

            self.starter_kit_path_attr = expanded_path
            self.starter_kit_path = expanded_path
            logger.info(f"Using custom starter kit: {self.starter_kit_path_attr}")
        else:
            # No custom path - use default location
            self.starter_kit_path_attr = os.path.join(
                os.path.dirname(workspace_dir), "v1"
            )
            self.starter_kit_path = self.starter_kit_path_attr
            logger.info(f"Using default starter kit path: {self.starter_kit_path_attr}")

        self.base_path = self.starter_kit_path_attr

        if not validate_starter_kit_path(
            self.starter_kit_path_attr, raise_on_invalid=False
        ):
            logger.info(f"Starter kit not found at {self.starter_kit_path_attr}")
            # Will be downloaded by install_local_environment

        self.validate_platform()
        return self.starter_kit_path_attr

    def install_local_environment(self, vf_env_id: str):
        """
        Install amzn-agi-verifiers and environment for LOCAL platform

        """
        # Validate that base_path has been set by setup_local()
        if not self.base_path:
            raise RuntimeError(
                "Local environment not properly initialized. Please ensure setup() was called before start_environment()."
            )

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
            if self.starter_kit_path and os.path.exists(self.starter_kit_path):
                shutil.rmtree(self.starter_kit_path, ignore_errors=True)
                logger.warning(
                    f"Cleaned up partial installation at {self.starter_kit_path}"
                )
            raise

    def deploy_sam_stack(self, s3_bucket: Optional[str] = None):
        """
        Deploy SAM stack from local machine

        Args:
            s3_bucket: Optional S3 bucket (not used for local deployment)
        """
        # Ensure starter_kit_path is set
        if not self.starter_kit_path:
            raise ValueError("starter_kit_path must be set before deploying SAM stack")

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
        Start environment locally using unified environment client.

        Args:
            env_type: Environment type (EnvType.TRAIN or EnvType.EVAL) for log file naming
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
            RuntimeError: If any process (train or eval) is already running for this stack
        """
        # Check if ANY job is running for this stack (across all sessions)
        error_msg = self._check_for_running_jobs_on_stack(self.stack_name)
        if error_msg:
            raise RuntimeError(error_msg)

        # Run setup commands first to ensure dependencies are installed
        setup_cmds = self._build_setup_commands(vf_env_id)
        setup_cmd = " && ".join(setup_cmds)
        logger.info("Running setup commands to ensure dependencies are installed...")
        subprocess.run(["/bin/bash", "-c", setup_cmd], check=True)

        # Build unified client command
        cmd = self._build_unified_client_command(
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

        # Use new folder structure and naming format
        session_id = getattr(self, "session_id", "default")
        log_file = self._get_log_file_path(session_id, env_type)

        # Create logs directory
        os.makedirs(self._get_logs_dir(), exist_ok=True)

        # Set environment variable to differentiate train vs eval processes
        env_vars = {"is_train": "1" if env_type == EnvType.TRAIN else "0"}

        # Store process in the appropriate attribute
        process = self.start_local_process(cmd, log_file, env_vars)
        if env_type == EnvType.TRAIN:
            self.train_process = process
        else:
            self.eval_process = process

        # Create tracking files
        self._create_pid_file(session_id, env_type, process.pid)
        self._create_status_file(session_id, env_type, JOB_STATUS_RUNNING)

        logger.info(f"Environment started locally, logs: {log_file}")

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
        session_id = getattr(self, "session_id", "default")
        log_file = self._get_log_file_path(session_id, env_type)

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

    def kill_task(
        self,
        env_type: EnvType,
        kill_all_for_stack: bool = False,
        preserve_logs: bool = True,
    ):
        """
        Stop running local process, update status, and remove tracking files.

        Args:
            env_type: Type of environment (TRAIN or EVAL)
            kill_all_for_stack: If True, kills ALL jobs of this type for the stack (cross-session).
                               If False, only kills jobs from current session.
            preserve_logs: If True, keeps log files after killing jobs. If False, deletes them.
        """
        if kill_all_for_stack:
            base_stack = self._extract_base_stack_name(self.stack_name)
            logs_dir = os.path.join(self.workspace_dir, SDK_RFT_LOGS_DIR)

            if not os.path.exists(logs_dir):
                logger.info(f"No logs directory found")
                return

            logger.info(
                f"Killing ALL {env_type.value} jobs for stack '{self.stack_name}'"
            )

            killed_count = 0
            for filename in os.listdir(logs_dir):
                if (
                    filename.endswith(".pid")
                    and base_stack in filename
                    and f"_{env_type.value}.pid" in filename
                ):
                    pid_file = os.path.join(logs_dir, filename)
                    try:
                        with open(pid_file, "r") as f:
                            pid = int(f.read().strip())

                        if self._is_process_running(pid):
                            os.kill(pid, signal.SIGTERM)
                            killed_count += 1

                        session_id = filename.replace(f"_{env_type.value}.pid", "")
                        status_file = os.path.join(
                            logs_dir, f"{session_id}_{env_type.value}.status"
                        )
                        with open(status_file, "w") as f:
                            f.write(JOB_STATUS_KILLED)

                        if preserve_logs:
                            if os.path.exists(pid_file):
                                os.remove(pid_file)
                        else:
                            log_file = os.path.join(
                                logs_dir, f"{session_id}_{env_type.value}.log"
                            )
                            for file_path in [log_file, pid_file, status_file]:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Error killing process from {filename}: {e}")

            status_msg = " (logs preserved)" if preserve_logs else ""
            logger.info(f"Killed {killed_count} {env_type.value} job(s){status_msg}")
        else:
            process = (
                self.train_process if env_type == EnvType.TRAIN else self.eval_process
            )
            session_id = getattr(self, "session_id", "default")

            log_file = self._get_log_file_path(session_id, env_type)
            pid_file = self._get_pid_file_path(session_id, env_type)
            status_file = self._get_status_file_path(session_id, env_type)

            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info(f"{env_type.value} process terminated")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"{env_type.value} process did not terminate gracefully, forcing kill"
                    )
                    try:
                        process.kill()
                        process.wait(timeout=5)
                        logger.info(f"{env_type.value} process killed")
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"{env_type.value} process could not be killed within timeout. "
                            f"Process may still be running (PID: {process.pid})"
                        )
            else:
                logger.info(f"No running {env_type.value} process found")

            self._create_status_file(session_id, env_type, JOB_STATUS_KILLED)

            if preserve_logs:
                if os.path.exists(pid_file):
                    os.remove(pid_file)
            else:
                for file_path in [log_file, pid_file, status_file]:
                    if os.path.exists(file_path):
                        os.remove(file_path)

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
            # Delete virtual environment
            venv_path = os.path.join(self.base_path, self.python_venv_name)
            if os.path.exists(venv_path):
                shutil.rmtree(venv_path)
                logger.info(f"Deleted virtual environment: {venv_path}")

            # Delete starter kit
            if os.path.exists(self.starter_kit_path):
                shutil.rmtree(self.starter_kit_path)
                logger.info(f"Deleted starter kit: {self.starter_kit_path}")

            # Delete logs directory
            logs_dir = self._get_logs_dir()
            if os.path.exists(logs_dir):
                shutil.rmtree(logs_dir)
                logger.info(f"Deleted logs directory: {logs_dir}")

    def get_state(self) -> Dict:
        """Get local platform state for serialization"""
        return {
            "python_venv_name": self.python_venv_name,
            "starter_kit_path": self.starter_kit_path,
            "base_path": self.base_path,
            "train_pid": self.train_process.pid
            if self.train_process and self.train_process.poll() is None
            else None,
            "eval_pid": self.eval_process.pid
            if self.eval_process and self.eval_process.poll() is None
            else None,
        }

    def restore_state(self, state: Dict):
        """Restore local platform state after deserialization"""
        # Store PIDs for potential recovery
        # Note: We can't recreate Popen objects, but we can store PIDs for manual kill if needed
        self._train_pid = state.get("train_pid")
        self._eval_pid = state.get("eval_pid")

        if self._train_pid:
            logger.info(f"Found train process PID: {self._train_pid}")
        if self._eval_pid:
            logger.info(f"Found eval process PID: {self._eval_pid}")
