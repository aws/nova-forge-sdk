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
"""Utility functions for RFT Multiturn infrastructure."""

import os
from typing import List


def validate_starter_kit_path(path: str, raise_on_invalid: bool = True) -> bool:
    """
    Validate if a path is a valid starter kit directory by checking for lambda_proxy subdirectory.

    Args:
        path: Path to validate
        raise_on_invalid: If True, raises ValueError on invalid path. If False, returns False.

    Returns:
        True if valid starter kit, False if invalid (only when raise_on_invalid=False)

    Raises:
        ValueError: If path is not a valid starter kit and raise_on_invalid=True
    """
    lambda_proxy_dir = os.path.join(path, "lambda_proxy")

    if not os.path.exists(lambda_proxy_dir):
        if raise_on_invalid:
            raise ValueError(
                f"Invalid starter kit directory (missing lambda_proxy): {path}\n"
                f"Expected structure: {path}/lambda_proxy/"
            )
        return False

    return True


def build_duplicate_job_error_message(
    stack_name: str,
    train_jobs: List[str],
    eval_jobs: List[str],
    platform_info: str = "",
) -> str:
    """
    Build standardized error message for duplicate job detection.

    Args:
        stack_name: Stack name
        train_jobs: List of train job identifiers
        eval_jobs: List of eval job identifiers
        platform_info: Optional platform-specific info (e.g., "on EC2 instance i-123")

    Returns:
        Formatted error message
    """
    job_details = []
    kill_instructions = []

    if train_jobs:
        job_details.append(f"{len(train_jobs)} TRAIN job(s) ({', '.join(train_jobs)})")
        kill_instructions.append(
            "rft_infra.kill_task(env_type=EnvType.TRAIN, kill_all_for_stack=True)"
        )
    if eval_jobs:
        job_details.append(f"{len(eval_jobs)} EVAL job(s) ({', '.join(eval_jobs)})")
        kill_instructions.append(
            "rft_infra.kill_task(env_type=EnvType.EVAL, kill_all_for_stack=True)"
        )

    platform_suffix = f" {platform_info}" if platform_info else ""

    return (
        f"Found running job(s) for stack '{stack_name}'{platform_suffix}:\n"
        f"  - {' and '.join(job_details)}\n"
        f"To stop them, run:\n"
        f"  {chr(10).join('  ' + instr for instr in kill_instructions)}"
    )
