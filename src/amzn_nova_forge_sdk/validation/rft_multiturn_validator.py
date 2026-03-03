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
            "Invalid environment ID: {vf_env_id}. "
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

    # Validate AGIModelLens Lambda ARN compatibility
    # Lambda function name will be: {stack_name}-SageMaker-rollout
    # AGIModelLens requires: [a-zA-Z0-9._-]{0,32}[Ss]age[Mm]aker[a-zA-Z0-9._-]{0,32}
    # This means the part before "-SageMaker" must be <= 32 characters
    if len(stack_name) > 32:
        raise ValueError(
            f"Stack name '{stack_name}' is too long ({len(stack_name)} characters). "
            f"Maximum length is 32 characters to ensure compatibility with AGIModelLens Lambda ARN validation. "
            f"The Lambda function name will be '{stack_name}-SageMaker-rollout'."
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


def validate_amazon_linux_ami(ec2_client, instance_id: str) -> None:
    """
    Validate that an EC2 instance is using an Amazon Linux AMI.

    This validation ensures compatibility with RFT workloads which require
    Amazon Linux 2 (AL2) or Amazon Linux 2023 (AL2023).

    Args:
        ec2_client: Boto3 EC2 client
        instance_id: EC2 instance ID to validate

    Raises:
        ValueError: If the instance is not using an Amazon Linux AMI

    Note:
        This function logs warnings for non-critical errors (e.g., API issues,
        permissions) but raises ValueError for actual AMI validation failures.
    """
    from amzn_nova_forge_sdk.util.logging import logger

    try:
        # Get instance details
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        if not response["Reservations"]:
            logger.warning(f"Instance {instance_id} not found during AMI validation")
            return

        instance = response["Reservations"][0]["Instances"][0]
        image_id = instance.get("ImageId")
        if not image_id:
            logger.warning("Could not determine AMI ID for instance")
            return

        # Get AMI details
        images = ec2_client.describe_images(ImageIds=[image_id])
        if not images["Images"]:
            logger.warning(f"Could not find AMI details for {image_id}")
            return

        image = images["Images"][0]
        image_name = image.get("Name", "").lower()
        image_description = image.get("Description", "").lower()

        # Check if it's Amazon Linux (AL2, AL2023, or older AL1)
        is_amazon_linux = (
            "amzn" in image_name
            or "amazon linux" in image_name
            or "amazon linux" in image_description
            or "al2023" in image_name
            or "al2" in image_name
        )

        if not is_amazon_linux:
            raise ValueError(
                f"Instance {instance_id} is not using an Amazon Linux AMI. "
                f"Found AMI: {image_id} ({image.get('Name', 'Unknown')}). "
                f"Please use Amazon Linux 2 (AL2) or Amazon Linux 2023 (AL2023) for RFT workloads."
            )

        logger.info(f"Validated Amazon Linux AMI: {image.get('Name', image_id)}")

    except ValueError:
        # Re-raise ValueError from AMI validation - this is a critical error
        raise
    except ec2_client.exceptions.ClientError as e:
        # If we can't describe the image (e.g., permissions issue), log warning but don't fail
        logger.warning(
            f"Could not validate AMI type: {e}. "
            f"Proceeding with assumption of Amazon Linux."
        )
    except Exception as e:
        # Only catch non-critical errors (e.g., API issues, permissions)
        logger.warning(
            f"Could not validate Amazon Linux AMI during initialization: {e}"
        )
