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

from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
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
    """Shared command building logic for ECS and EC2 infrastructures"""

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
            # Install git-remote-s3 in venv BEFORE attempting git clone
            f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install boto3 git-remote-s3",
            # Only clone if starter kit files don't exist (check for nova-rl-async-client directory)
            f"[ ! -d {base_path_q}/nova-rl-async-client ] && git clone {starter_kit_s3_q} {base_path_q}-tmp && mv {base_path_q}-tmp/* {base_path_q}-tmp/.git* {base_path_q}/ 2>/dev/null || true && rm -rf {base_path_q}-tmp || echo 'Starter kit already exists, skipping clone'",
            f"cd {base_path_q} && git checkout master",
            f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install -e . --find-links wheelhouse",
            f"cd {base_path_q} && . {python_venv_name_q}/bin/activate && pip install -e nova-rl-async-client --find-links wheelhouse",
        ]

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

    def _build_command(
        self,
        mode: str,
        vf_env_id: str,
        vf_env_args: Dict,
        lambda_url: str,
        queue_url: Optional[str] = None,
        base_path: Optional[str] = None,
        groups_per_batch: int = 20,
        max_messages_per_poll: int = 10,
        client_timeout: float = 600.0,
        client_poll_interval: float = 0.5,
        rollouts_per_example: int = 1,
        max_concurrent: int = 60,
    ) -> str:
        """
        Build training or evaluation command.

        Args:
            mode: 'train' or 'eval'
            vf_env_id: Verifier environment identifier
            vf_env_args: Environment arguments dictionary
            lambda_url: Lambda function URL
            queue_url: SQS queue URL (required for training)
            base_path: Base installation path
            groups_per_batch: Number of groups per batch (training only)
            max_messages_per_poll: Max messages per poll (training only)
            client_timeout: Client timeout in seconds
            client_poll_interval: Client poll interval in seconds
            rollouts_per_example: Rollouts per example (evaluation only)
            max_concurrent: Max concurrent requests (evaluation only)

        Returns:
            Shell command string to execute
        """
        # Validate all inputs
        if mode not in ("train", "eval"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")

        validate_env_id(vf_env_id)
        validate_url(lambda_url, "Lambda URL")

        if mode == "train":
            validate_url(queue_url, "Queue URL", required=True)

        base_path = base_path or getattr(self, "base_path")
        validate_path(base_path)

        region = getattr(self, "region", "us-east-1")
        validate_region(region)

        python_venv_name = getattr(self, "python_venv_name")
        validate_env_id(python_venv_name)

        # Validate numeric parameters
        if not isinstance(groups_per_batch, int) or groups_per_batch <= 0:
            raise ValueError(
                f"groups_per_batch must be a positive integer, got: {groups_per_batch}"
            )
        if not isinstance(max_messages_per_poll, int) or max_messages_per_poll <= 0:
            raise ValueError(
                f"max_messages_per_poll must be a positive integer, got: {max_messages_per_poll}"
            )
        if not isinstance(client_timeout, (int, float)) or client_timeout <= 0:
            raise ValueError(
                f"client_timeout must be a positive number, got: {client_timeout}"
            )
        if (
            not isinstance(client_poll_interval, (int, float))
            or client_poll_interval <= 0
        ):
            raise ValueError(
                f"client_poll_interval must be a positive number, got: {client_poll_interval}"
            )
        if not isinstance(rollouts_per_example, int) or rollouts_per_example <= 0:
            raise ValueError(
                f"rollouts_per_example must be a positive integer, got: {rollouts_per_example}"
            )
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be a positive integer, got: {max_concurrent}"
            )

        # Validate vf_env_args dictionary and its values
        validate_dict_values(vf_env_args, "vf_env_args")

        # Safely serialize vf_env_args
        try:
            vf_env_args_json = json.dumps(vf_env_args)
        except (TypeError, ValueError) as e:
            raise ValueError(f"vf_env_args contains non-serializable values: {e}")

        # Quote all dynamic values
        base_path_q = shlex.quote(base_path)
        python_venv_name_q = shlex.quote(python_venv_name)
        vf_env_id_q = shlex.quote(vf_env_id)
        vf_env_id_module = shlex.quote(vf_env_id.replace("-", "_"))
        lambda_url_q = shlex.quote(lambda_url)
        region_q = shlex.quote(region)
        vf_env_args_json_q = shlex.quote(vf_env_args_json)

        # Check if environment and dependencies are installed
        env_check = (
            f"cd {base_path_q}\n"
            f". {python_venv_name_q}/bin/activate\n"
            f'python -c "import {vf_env_id_module}" || '
            f"(echo 'Environment {vf_env_id_q} not installed, exiting' && exit 1)\n"
        )

        base = f"cd {base_path_q}\n. {python_venv_name_q}/bin/activate\n"

        if mode == "train":
            queue_url_q = shlex.quote(queue_url) if queue_url else "''"
            return (
                env_check
                + base
                + (
                    f"export is_train=1\n"
                    f"python {base_path_q}/nova-rl-async-client/src/train.py "
                    f"queue_url={queue_url_q} "
                    f"region_name={region_q} "
                    f"groups_per_batch={groups_per_batch} "
                    f"max_messages_per_poll={max_messages_per_poll} "
                    f"client_base_url={lambda_url_q} "
                    f"client_region={region_q} "
                    f"client_timeout={client_timeout} "
                    f"client_poll_interval={client_poll_interval} "
                    f"vf_env_id={vf_env_id_q} "
                    f"vf_env_args={vf_env_args_json_q}"
                )
            )

        num_eval_examples = vf_env_args.get("num_eval_examples", 100)
        if not isinstance(num_eval_examples, int) or num_eval_examples <= 0:
            raise ValueError(
                f"num_eval_examples must be a positive integer, got: {num_eval_examples}"
            )

        return (
            env_check
            + base
            + (
                f"python {base_path_q}/nova-rl-async-client/src/evaluate.py "
                f"num_examples={num_eval_examples} "
                f"rollouts_per_example={rollouts_per_example} "
                f"max_concurrent={max_concurrent} "
                f"client_poll_interval={client_poll_interval} "
                f"client_timeout={client_timeout} "
                f"vf_env_args={vf_env_args_json_q} "
                f"vf_env_id={vf_env_id_q} "
                f"client_base_url={lambda_url_q}"
            )
        )

    def _build_sam_deploy_commands(self) -> List[str]:
        """Build SAM deployment commands"""
        pkg_install = self._get_package_install_cmd()

        base_dir = getattr(self, "sam_base_dir", "/root")
        validate_path(base_dir)

        log_file = getattr(self, "sam_log_file", None)
        if log_file:
            validate_path(log_file)

        python_venv_name = getattr(self, "python_venv_name")
        validate_env_id(python_venv_name)

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

        return pkg_install + [
            f"cd {base_dir_q}",
            # Create v1 directory
            f"mkdir -p v1",
            # Only create venv if it doesn't exist (inside v1 directory)
            f"cd {base_dir_q}/v1 && [ ! -d {python_venv_name_q} ] && {base_python_q} -m venv {python_venv_name_q} {log_append} || echo '{python_venv_name_q} already exists' {log_append}",
            # Install git-remote-s3 in venv before cloning
            f"cd {base_dir_q}/v1 && . {python_venv_name_q}/bin/activate && pip install boto3 git-remote-s3 aws-sam-cli {log_redirect}",
            # Only clone if directory doesn't exist
            f"cd {base_dir_q} && [ ! -d v1/lambda_proxy ] && git clone {starter_kit_s3_q} v1-tmp {log_append} && mv v1-tmp/* v1-tmp/.git* v1/ 2>/dev/null || true && rm -rf v1-tmp {log_append} || echo 'Starter kit already exists' {log_append}",
            f"cd {base_dir_q}/v1 && git checkout master {log_append}",
            f"cd {base_dir_q}/v1/lambda_proxy && . {base_dir_q}/v1/{python_venv_name_q}/bin/activate && sam build {log_append}",
            f"cd {base_dir_q}/v1/lambda_proxy && . {base_dir_q}/v1/{python_venv_name_q}/bin/activate && "
            f"sam deploy --stack-name {stack_name_q} --capabilities CAPABILITY_IAM "
            f"--parameter-overrides ProjectName={stack_name_q} --region {region_q} "
            f"--no-confirm-changeset --no-fail-on-empty-changeset {log_append}",
        ]

    def _get_package_install_cmd(self) -> List[str]:
        """Override in subclass for platform-specific package installation"""
        raise NotImplementedError
