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
Utility functions for platform detection and validation
"""

from typing import Optional

from amzn_nova_customization_sdk.model.model_enums import Platform


def detect_platform_from_path(s3_path: str) -> Optional[Platform]:
    """
    Detect platform from S3 checkpoint path pattern.

    Platform identifiers appear in the escrow bucket name:
    - SMTJ: s3://customer-escrow-{account}-smtj-{id}/...
    - SMHP: s3://customer-escrow-{account}-hp-{id}/...

    Args:
        s3_path: S3 checkpoint path to analyze

    Returns:
        Platform.SMTJ, Platform.SMHP, or None if cannot determine

    Examples:
        >>> detect_platform_from_path("s3://customer-escrow-123-smtj-abc/model/checkpoint")
        Platform.SMTJ
        >>> detect_platform_from_path("s3://customer-escrow-123-hp-abc/model/checkpoint")
        Platform.SMHP
        >>> detect_platform_from_path("s3://my-bucket/output")
        None
    """
    if "-smtj-" in s3_path:
        return Platform.SMTJ
    elif "-hp-" in s3_path:
        return Platform.SMHP
    return None


def validate_platform_compatibility(
    checkpoint_platform: Optional[Platform],
    execution_platform: Platform,
    checkpoint_source: str = "checkpoint",
) -> None:
    """
    Validate that checkpoint platform matches execution platform.

    SMHP-trained checkpoints can only be used on SMHP.
    SMTJ-trained checkpoints can only be used on SMTJ.

    Args:
        checkpoint_platform: Platform where checkpoint was trained (None if unknown)
        execution_platform: Platform where job will execute
        checkpoint_source: Description of checkpoint source for error messages

    Raises:
        ValueError: If platforms are incompatible

    Note:
        If checkpoint_platform is None, logs a warning but allows execution.
    """
    if checkpoint_platform is None:
        from amzn_nova_customization_sdk.util.logging import logger

        logger.warning(
            f"Cannot determine platform for {checkpoint_source}. "
            f"If this checkpoint was trained on a different platform than {execution_platform.value}, "
            f"the job may fail with a validation error."
        )
        return

    if checkpoint_platform != execution_platform:
        raise ValueError(
            f"Platform mismatch: {checkpoint_source} was trained on {checkpoint_platform.value} "
            f"but you are trying to run on {execution_platform.value}. "
            f"Checkpoints must be evaluated/used on the same platform they were trained on. "
            f"Please use {checkpoint_platform.value} for this checkpoint."
        )
