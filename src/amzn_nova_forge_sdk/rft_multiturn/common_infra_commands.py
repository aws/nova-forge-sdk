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
import os
import re
import shlex
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from amzn_nova_forge_sdk.util.logging import logger
from amzn_nova_forge_sdk.validation.rft_multiturn_validator import (
    validate_dict_values,
    validate_env_id,
    validate_path,
    validate_python_command,
    validate_region,
    validate_stack_name,
    validate_url,
)

BASE_PYTHON_COMMAND = "python3.12"


class CommonInfraCommands:
    def _build_unified_client_command(
        self,
        vf_env_id: str,
        vf_env_args: Dict,
        lambda_url: str,
        queue_url: str,
        base_path: Optional[str] = None,
        max_concurrent_rollouts: int = 40,
        max_rollout_timeout: float = 300.0,
        completion_poll_timeout: float = 600.0,
        completion_poll_interval: float = 0.5,
        rollout_poll_interval: float = 1.0,
        log_output_directory: Optional[str] = None,
        config_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> str:
        """
        Build unified environment client command (environment_client.py).

        This replaces both train.py and evaluate.py with a simplified interface.

        Args:
            vf_env_id: Verifier environment identifier
            vf_env_args: Environment arguments dictionary
            lambda_url: Lambda function URL
            queue_url: SQS queue URL
            base_path: Base installation path
            max_concurrent_rollouts: Max concurrent rollouts (default: 40)
            max_rollout_timeout: Per-rollout timeout in seconds (default: 300)
            completion_poll_timeout: Completion polling timeout (default: 600)
            completion_poll_interval: Completion poll interval (default: 0.5)
            rollout_poll_interval: SQS polling interval (default: 1.0)
            log_output_directory: Directory for logs and metrics (optional)
            config_name: If provided, use YAML config mode instead of CLI flags
            config_path: Custom config directory path (optional, use with config_name)

        Returns:
            Shell command string to execute environment_client.py
        """
        # Validate all inputs
        validate_env_id(vf_env_id)
        validate_url(lambda_url, "Lambda URL")
        validate_url(queue_url, "Queue URL", required=True)

        base_path = base_path or getattr(self, "base_path")
        validate_path(base_path)

        region = getattr(self, "region", "us-east-1")
        validate_region(region)

        python_venv_name = getattr(self, "python_venv_name")
        validate_env_id(python_venv_name)

        # Validate numeric parameters
        if not isinstance(max_concurrent_rollouts, int) or max_concurrent_rollouts <= 0:
            raise ValueError(
                f"max_concurrent_rollouts must be a positive integer, got: {max_concurrent_rollouts}"
            )
        if (
            not isinstance(max_rollout_timeout, (int, float))
            or max_rollout_timeout <= 0
        ):
            raise ValueError(
                f"max_rollout_timeout must be a positive number, got: {max_rollout_timeout}"
            )
        if (
            not isinstance(completion_poll_timeout, (int, float))
            or completion_poll_timeout <= 0
        ):
            raise ValueError(
                f"completion_poll_timeout must be a positive number, got: {completion_poll_timeout}"
            )
        if (
            not isinstance(completion_poll_interval, (int, float))
            or completion_poll_interval <= 0
        ):
            raise ValueError(
                f"completion_poll_interval must be a positive number, got: {completion_poll_interval}"
            )
        if (
            not isinstance(rollout_poll_interval, (int, float))
            or rollout_poll_interval <= 0
        ):
            raise ValueError(
                f"rollout_poll_interval must be a positive number, got: {rollout_poll_interval}"
            )

        # Validate vf_env_args dictionary and its values
        validate_dict_values(vf_env_args, "vf_env_args")

        # Validate log_output_directory if provided
        if log_output_directory:
            validate_path(log_output_directory)

        # Quote all dynamic values
        base_path_q = shlex.quote(base_path)
        python_venv_name_q = shlex.quote(python_venv_name)
        vf_env_id_q = shlex.quote(vf_env_id)
        vf_env_id_module = shlex.quote(vf_env_id.replace("-", "_"))
        region_q = shlex.quote(region)

        # Check if environment and dependencies are installed
        env_check = (
            f"cd {base_path_q}\n"
            f". {python_venv_name_q}/bin/activate\n"
            f'python -c "import {vf_env_id_module}" || '
            f"(echo 'Environment {vf_env_id_q} not installed, exiting' && exit 1)\n"
        )

        base = f"cd {base_path_q}\n. {python_venv_name_q}/bin/activate\n"

        # YAML mode - just point to config file
        if config_name:
            config_name_q = shlex.quote(config_name)
            cmd = (
                env_check
                + base
                + f"python {base_path_q}/nova-rl-async-client/src/environment_client.py "
                f"--config-name {config_name_q}"
            )
            # Add custom config path if specified
            if config_path:
                config_path_q = shlex.quote(config_path)
                cmd += f" --config-path {config_path_q}"
            return cmd

        # CLI mode - build full command with all parameters
        # Safely serialize vf_env_args
        try:
            vf_env_args_json = json.dumps(vf_env_args)
        except (TypeError, ValueError) as e:
            raise ValueError(f"vf_env_args contains non-serializable values: {e}")

        lambda_url_q = shlex.quote(lambda_url)
        queue_url_q = shlex.quote(queue_url)
        vf_env_args_json_q = shlex.quote(vf_env_args_json)

        command = (
            env_check
            + base
            + f"python {base_path_q}/nova-rl-async-client/src/environment_client.py "
            f"--environment-id {vf_env_id_q} "
            f"--environment-args {vf_env_args_json_q} "
            f"--rollout-request-sqs-url {queue_url_q} "
            f"--completion-lambda-url {lambda_url_q} "
            f"--completion-poll-interval {completion_poll_interval} "
            f"--completion-poll-timeout {completion_poll_timeout} "
            f"--max-concurrent-rollouts {max_concurrent_rollouts} "
            f"--max-rollout-timeout {max_rollout_timeout} "
            f"--rollout-poll-interval {rollout_poll_interval} "
            f"--aws-region {region_q}"
        )

        # Add log directory if specified
        if log_output_directory:
            log_dir_q = shlex.quote(log_output_directory)
            command += f" --log-output-directory {log_dir_q}"

        return command

    def _build_starter_kit_download_commands(
        self,
        starter_kit_s3: str,
        target_dir: str,
        check_dir: str,
        log_append: str = "",
    ) -> List[str]:
        """
        Build commands to download starter kit (tarball or git repo).

        Args:
            starter_kit_s3: S3 URI of starter kit (tarball or git repo)
            target_dir: Target directory for extraction/clone
            check_dir: Directory to check if starter kit already exists
            log_append: Log redirection string (optional)

        Returns:
            List of shell commands
        """
        starter_kit_s3_q = shlex.quote(starter_kit_s3)
        target_dir_q = shlex.quote(target_dir)
        check_dir_q = shlex.quote(check_dir)

        commands = []

        # Check if starter kit is a tarball or git repo
        if starter_kit_s3.endswith(".tar.gz"):
            # It's a tarball - download and extract
            commands.append(
                f"[ ! -d {check_dir_q} ] && "
                f"aws s3 cp {starter_kit_s3_q} /tmp/starter_kit.tar.gz {log_append} && "
                f"tar -xzf /tmp/starter_kit.tar.gz -C {target_dir_q} {log_append} && "
                f"rm /tmp/starter_kit.tar.gz {log_append} || "
                f"echo 'Starter kit already exists' {log_append}"
            )
        else:
            # It's a git repo - clone it
            parent_dir = target_dir.rsplit("/", 1)[0]
            parent_dir_q = shlex.quote(parent_dir)
            target_base = target_dir.rsplit("/", 1)[-1]
            target_base_q = shlex.quote(target_base)

            commands.append(
                f"cd {parent_dir_q} && [ ! -d {check_dir_q} ] && "
                f"git clone {starter_kit_s3_q} {target_base_q}-tmp {log_append} && "
                f"mv {target_base_q}-tmp/* {target_base_q}-tmp/.git* {target_dir_q}/ 2>/dev/null || true && "
                f"rm -rf {target_base_q}-tmp {log_append} || "
                f"echo 'Starter kit already exists' {log_append}"
            )
            # Only checkout master if it's a git repo
            commands.append(
                f"cd {target_dir_q} && "
                f"(git rev-parse --verify master >/dev/null 2>&1 && git checkout master {log_append} || "
                f"echo 'No master branch found, skipping checkout' {log_append})"
            )

        return commands

    def _build_sam_deploy_commands(
        self, custom_starter_kit_s3: Optional[str] = None
    ) -> List[str]:
        """
        Build SAM deployment commands

        Args:
            custom_starter_kit_s3: Optional custom starter kit S3 URI to use instead of default
        """
        pkg_install = self._get_package_install_cmd()

        base_dir = getattr(self, "sam_base_dir", "/root")
        validate_path(base_dir)

        log_file = getattr(self, "sam_log_file", None)
        if log_file:
            validate_path(log_file)

        python_venv_name = getattr(self, "python_venv_name")
        validate_env_id(python_venv_name)

        # Use custom starter kit if provided, otherwise use default
        if custom_starter_kit_s3:
            starter_kit_s3 = custom_starter_kit_s3
            logger.info(f"Using custom starter kit: {starter_kit_s3}")
        else:
            starter_kit_s3 = getattr(self, "starter_kit_s3", "s3://default-starter-kit")
        validate_url(starter_kit_s3, "Starter kit S3 URL")

        stack_name = getattr(self, "stack_name", "default-stack")
        validate_stack_name(stack_name)

        region = getattr(self, "region", "us-east-1")
        validate_region(region)

        # Quote all dynamic values
        base_dir_q = shlex.quote(base_dir)
        python_venv_name_q = shlex.quote(python_venv_name)
        starter_kit_s3_q = shlex.quote(starter_kit_s3)
        stack_name_q = shlex.quote(stack_name)
        region_q = shlex.quote(region)
        base_python_q = shlex.quote(BASE_PYTHON_COMMAND)

        # Use log redirection only if log_file is specified (EC2), otherwise output to stdout (ECS)
        log_redirect = f"> {shlex.quote(log_file)} 2>&1" if log_file else ""
        log_append = f">> {shlex.quote(log_file)} 2>&1" if log_file else ""

        # Determine if starter kit is a tarball or git repo
        is_tarball = starter_kit_s3.endswith(".tar.gz")

        commands = pkg_install + [
            f"cd {base_dir_q}",
            # Create v1 directory
            f"mkdir -p v1",
            # Only create venv if it doesn't exist (inside v1 directory)
            f"cd {base_dir_q}/v1 && [ ! -d {python_venv_name_q} ] && {base_python_q} -m venv {python_venv_name_q} {log_append} || echo '{python_venv_name_q} already exists' {log_append}",
        ]

        # Install dependencies based on starter kit type
        if is_tarball:
            # For tarball, only need boto3 and aws-sam-cli
            commands.append(
                f"cd {base_dir_q}/v1 && . {python_venv_name_q}/bin/activate && pip install boto3 aws-sam-cli {log_redirect}"
            )
        else:
            # For git repo, need git-remote-s3
            commands.append(
                f"cd {base_dir_q}/v1 && . {python_venv_name_q}/bin/activate && pip install boto3 git-remote-s3 aws-sam-cli {log_redirect}"
            )

        # Download starter kit using common helper
        # Note: For tarball, extract to base_dir (tarball contains v1/ directory)
        # For git repo, clone to base_dir/v1
        commands.extend(
            self._build_starter_kit_download_commands(
                starter_kit_s3=starter_kit_s3,
                target_dir=base_dir if is_tarball else f"{base_dir}/v1",
                check_dir=f"{base_dir}/v1/lambda_proxy",
                log_append=log_append,
            )
        )

        # SAM build and deploy commands (same for both)
        commands.extend(
            [
                f"cd {base_dir_q}/v1/lambda_proxy && . {base_dir_q}/v1/{python_venv_name_q}/bin/activate && sam build {log_append}",
                f"cd {base_dir_q}/v1/lambda_proxy && . {base_dir_q}/v1/{python_venv_name_q}/bin/activate && "
                f"sam deploy --stack-name {stack_name_q} --capabilities CAPABILITY_IAM "
                f"--parameter-overrides ProjectName={stack_name_q} --region {region_q} "
                f"--no-confirm-changeset --no-fail-on-empty-changeset {log_append}",
            ]
        )

        return commands

    def _build_setup_commands(
        self, vf_env_id: str, base_path: Optional[str] = None
    ) -> List[str]:
        """
        Build setup commands for installing starter kit and environment.

        Args:
            vf_env_id: Verifier environment identifier
            base_path: Base installation path (defaults to self.base_path)

        Returns:
            List of shell commands to execute
        """
        # Validate all inputs
        validate_env_id(vf_env_id)

        base_path = base_path or getattr(self, "base_path")
        validate_path(base_path)
        parent_path = base_path.rsplit("/", 1)[0]

        python_venv_name = getattr(self, "python_venv_name")
        validate_env_id(python_venv_name)  # Reuse validation for venv name

        pkg_install = self._get_package_install_cmd()

        # Use instance python_command if available (LOCAL), otherwise use BASE_PYTHON_COMMAND (EC2/ECS)
        python_cmd = getattr(self, "python_command", BASE_PYTHON_COMMAND)
        validate_python_command(python_cmd)

        starter_kit_s3 = getattr(self, "starter_kit_s3", "s3://default-starter-kit")
        validate_url(starter_kit_s3, "Starter kit S3 URL")

        # Quote all dynamic values to prevent injection
        base_path_q = shlex.quote(base_path)
        parent_path_q = shlex.quote(parent_path)
        python_venv_name_q = shlex.quote(python_venv_name)
        python_cmd_q = shlex.quote(python_cmd)
        starter_kit_s3_q = shlex.quote(starter_kit_s3)

        commands = pkg_install + [
            f"cd {parent_path_q}",
            # Create base_path if it doesn't exist
            f"mkdir -p {base_path_q}",
            # Only create venv if it doesn't exist
            f"[ ! -d {base_path_q}/{python_venv_name_q} ] && {python_cmd_q} -m venv {base_path_q}/{python_venv_name_q} || echo '{python_venv_name_q} already exists, skipping creation'",
        ]

        # Install dependencies based on starter kit type
        is_tarball = starter_kit_s3.endswith(".tar.gz")

        if is_tarball:
            # For tarball, only need boto3
            commands.append(
                f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install boto3"
            )
        else:
            # For git repo, need git-remote-s3
            commands.append(
                f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install boto3 git-remote-s3"
            )

        # Download starter kit using common helper
        # Note: For tarball, extract to parent_path (tarball contains base_path directory)
        # For git repo, clone to base_path
        commands.extend(
            self._build_starter_kit_download_commands(
                starter_kit_s3=starter_kit_s3,
                target_dir=parent_path if is_tarball else base_path,
                check_dir=f"{base_path}/nova-rl-async-client",
                log_append="",
            )
        )

        commands.extend(
            [
                f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install -e . --find-links wheelhouse",
                f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install -e nova-rl-async-client --find-links wheelhouse",
            ]
        )

        # Handle custom environment installation
        custom_env = getattr(self, "custom_env", None)
        if custom_env:
            if custom_env.s3_uri:
                # Validate and quote S3 URI
                validate_url(custom_env.s3_uri, "Custom environment S3 URI")
                s3_uri_q = shlex.quote(custom_env.s3_uri)
                vf_env_id_q = shlex.quote(vf_env_id)
                vf_env_id_module = shlex.quote(vf_env_id.replace("-", "_"))

                # Download and install from S3
                commands.extend(
                    [
                        f"cd {parent_path_q}",
                        f"aws s3 cp {s3_uri_q} /tmp/{vf_env_id_q}.tar.gz",
                        f"tar -xzf /tmp/{vf_env_id_q}.tar.gz -C {parent_path_q}",
                        # Install in venv - use full path to venv python to ensure it's in the venv
                        f"cd {parent_path_q} && {base_path_q}/{python_venv_name_q}/bin/{shlex.quote(BASE_PYTHON_COMMAND)} -m pip install -e ./{vf_env_id_q} --find-links {base_path_q}/wheelhouse",
                        # Verify installation
                        f"{base_path_q}/{python_venv_name_q}/bin/{shlex.quote(BASE_PYTHON_COMMAND)} -c \"import {vf_env_id_module}; print('Custom env installed successfully')\" || echo 'Warning: Could not import custom env module'",
                    ]
                )
            elif custom_env.local_path:
                # Validate and quote local path
                validate_path(custom_env.local_path)
                local_path_q = shlex.quote(custom_env.local_path)

                # Install from local path (for LOCAL platform)
                commands.append(
                    f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install -e {local_path_q}"
                )
        elif vf_env_id:
            # Install built-in environment only if vf_env_id is provided
            vf_env_id_q = shlex.quote(vf_env_id)
            commands.append(
                f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && VERIFIERS_PATH=$(pip show amzn-agi-verifiers | grep Location | cut -d' ' -f2)/verifiers && pip install -e $VERIFIERS_PATH/environments/{vf_env_id_q}"
            )

        return commands

    def _get_package_install_cmd(self) -> List[str]:
        """Override in subclass for platform-specific package installation"""
        raise NotImplementedError
