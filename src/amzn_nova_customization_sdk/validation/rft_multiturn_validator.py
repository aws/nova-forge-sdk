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
"""Input validation utilities for RFT multiturn infrastructure commands.

This module provides validation functions to prevent command injection and ensure
safe handling of user inputs in shell commands.
"""

import re
from typing import Dict, Optional


def validate_env_id(vf_env_id: str) -> None:
    """
    Validate environment ID to prevent shell injection.

    Args:
        vf_env_id: Environment identifier to validate

    Raises:
        ValueError: If environment ID contains invalid characters
    """
    if not re.match(r"^[a-zA-Z0-9_-]+$", vf_env_id):
        raise ValueError(
            f"Invalid environment ID: {vf_env_id}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )


def validate_path(path: str) -> None:
    """
    Validate file path to prevent path traversal and injection.

    Args:
        path: File path to validate

    Raises:
        ValueError: If path contains invalid characters or patterns
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Check for path traversal attempts
    if ".." in path:
        raise ValueError(f"Path traversal detected in: {path}")

    # Allow alphanumeric, forward slash, hyphen, underscore, and dot
    if not re.match(r"^[a-zA-Z0-9/_.-]+$", path):
        raise ValueError(
            f"Invalid path: {path}. "
            "Only alphanumeric characters, forward slashes, hyphens, underscores, and dots are allowed."
        )


def validate_url(
    url: Optional[str], url_type: str = "URL", required: bool = True
) -> None:
    """
    Validate URL format to prevent injection.

    Args:
        url: URL to validate
        url_type: Type of URL for error messages
        required: If True, raises error when url is None. If False, allows None.

    Raises:
        ValueError: If URL format is invalid or required but None
    """
    if url is None:
        if required:
            raise ValueError(f"{url_type} is required")
        return

    if not isinstance(url, str) or not url:
        raise ValueError(f"{url_type} must be a non-empty string")

    # Basic URL validation - must start with http://, https://, or s3://
    if not re.match(r"^(https?|s3)://[a-zA-Z0-9._/-]+$", url):
        raise ValueError(
            f"Invalid {url_type}: {url}. Must be a valid HTTP, HTTPS, or S3 URL."
        )


def validate_stack_name(stack_name: str) -> None:
    """
    Validate CloudFormation stack name.

    Args:
        stack_name: Stack name to validate

    Raises:
        ValueError: If stack name is invalid
    """
    if not stack_name or not isinstance(stack_name, str):
        raise ValueError("Stack name must be a non-empty string")

    # CloudFormation stack names: alphanumeric and hyphens only
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9-]*$", stack_name):
        raise ValueError(
            f"Invalid stack name: {stack_name}. "
            "Must start with a letter and contain only alphanumeric characters and hyphens."
        )


def validate_region(region: str) -> None:
    """
    Validate AWS region name.

    Args:
        region: AWS region to validate

    Raises:
        ValueError: If region format is invalid
    """
    if not region or not isinstance(region, str):
        raise ValueError("Region must be a non-empty string")

    # AWS region format: us-east-1, eu-west-2, etc.
    if not re.match(r"^[a-z]{2}-[a-z]+-\d+$", region):
        raise ValueError(
            f"Invalid AWS region: {region}. "
            "Must follow AWS region naming convention (e.g., us-east-1)."
        )


def validate_dict_values(data: Dict, dict_name: str = "dictionary") -> None:
    """
    Validate dictionary values to ensure they're safe for JSON serialization
    and don't contain injection attempts.

    Args:
        data: Dictionary to validate
        dict_name: Name of dictionary for error messages

    Raises:
        ValueError: If dictionary contains invalid values
    """
    if not isinstance(data, dict):
        raise ValueError(f"{dict_name} must be a dictionary, got: {type(data)}")

    def validate_value(value, path=""):
        """Recursively validate dictionary values."""
        # Allow basic JSON-serializable types
        if value is None or isinstance(value, (bool, int, float)):
            return

        if isinstance(value, str):
            # Check for null bytes (security risk)
            if "\x00" in value:
                raise ValueError(f"String contains null bytes at {path}")
            return

        if isinstance(value, list):
            # Recursively validate list items
            for i, item in enumerate(value):
                validate_value(item, f"{path}[{i}]")
            return

        if isinstance(value, dict):
            # Recursively validate dict values
            for key, val in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Dictionary key must be string at {path}, got: {type(key)}"
                    )
                validate_value(val, f"{path}.{key}")
            return

        # Reject unsupported types
        raise ValueError(
            f"Unsupported type at {path}: {type(value)}. "
            "Only None, bool, int, float, str, list, and dict are allowed."
        )

    # Validate all values in the dictionary
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Dictionary key must be string in {dict_name}, got: {type(key)}"
            )
        validate_value(value, f"{dict_name}.{key}")


def validate_python_command(python_cmd: str) -> None:
    """
    Validate Python command to prevent command injection.

    Args:
        python_cmd: Python command to validate (e.g., "python3.12" or "/usr/bin/python3")

    Raises:
        ValueError: If command contains invalid characters or patterns
    """
    if not python_cmd or not isinstance(python_cmd, str):
        raise ValueError("Python command must be a non-empty string")

    # Allow simple command names (python3.12) or absolute paths (/usr/bin/python3)
    if not re.match(r"^([a-zA-Z0-9._-]+|/[a-zA-Z0-9._/-]+)$", python_cmd):
        raise ValueError(
            f"Invalid python command: {python_cmd}. "
            "Must be a simple command name or absolute path without special characters."
        )


def validate_platform(platform: str) -> None:
    """
    Validate platform value.

    Args:
        platform: Platform identifier to validate

    Raises:
        ValueError: If platform is not one of the supported values
    """
    if not platform or not isinstance(platform, str):
        raise ValueError("Platform must be a non-empty string")

    valid_platforms = ["local", "ec2", "ecs"]
    if platform not in valid_platforms:
        raise ValueError(
            f"Invalid platform: {platform}. "
            f"Must be one of: {', '.join(valid_platforms)}"
        )


def validate_ec2_instance_identifier(identifier: str) -> str:
    """
    Validate and extract EC2 instance ID from ARN or instance ID.

    Args:
        identifier: EC2 instance ID (i-xxxxx) or full ARN

    Returns:
        Instance ID (i-xxxxx format)

    Raises:
        ValueError: If identifier format is invalid
    """
    if not identifier or not isinstance(identifier, str):
        raise ValueError("EC2 instance identifier must be a non-empty string")

    # Validate format: either "i-xxxxx" or full EC2 ARN
    instance_id_pattern = r"^i-[a-f0-9]{8,17}$"
    arn_pattern = r"^arn:aws:ec2:[a-z0-9-]+:\d{12}:instance/i-[a-f0-9]{8,17}$"

    if re.match(instance_id_pattern, identifier):
        return identifier
    elif re.match(arn_pattern, identifier):
        return identifier.split("/")[-1]
    else:
        raise ValueError(
            f"Invalid EC2 instance identifier: {identifier}. "
            f"Expected format: 'i-xxxxx' or 'arn:aws:ec2:region:account:instance/i-xxxxx'"
        )


def validate_ecs_cluster_arn(cluster_arn: str) -> None:
    """
    Validate ECS cluster ARN format.

    Args:
        cluster_arn: ECS cluster ARN to validate

    Raises:
        ValueError: If ARN format is invalid
    """
    if not cluster_arn or not isinstance(cluster_arn, str):
        raise ValueError("ECS cluster ARN must be a non-empty string")

    # ECS cluster ARN format: arn:aws:ecs:region:account:cluster/cluster-name
    arn_pattern = r"^arn:aws:ecs:[a-z0-9-]+:\d{12}:cluster/[a-zA-Z0-9_-]+$"

    if not re.match(arn_pattern, cluster_arn):
        raise ValueError(
            f"Invalid ECS cluster ARN: {cluster_arn}. "
            f"Expected format: 'arn:aws:ecs:region:account:cluster/cluster-name'"
        )
