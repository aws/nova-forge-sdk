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
import logging
import re
from typing import Dict, List, Optional, Tuple

from amzn_nova_forge_sdk.model.model_config import SUPPORTED_SMI_CONFIGS
from amzn_nova_forge_sdk.model.model_enums import Model
from amzn_nova_forge_sdk.util.logging import logger

S3_URI_PREFIX_REGEX = re.compile(r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9_.-]+)*/$")

BEDROCK_DEPLOYMENT_ARN_REGEX = re.compile(
    r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:custom-model-deployment/[A-Za-z0-9-_]+$"
)

SAGEMAKER_ENDPOINT_ARN_REGEX = re.compile(
    r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:endpoint/[A-Za-z0-9-_]+$"
)


def validate_s3_uri_prefix(s3_uri: str) -> None:
    """
    Validation method that checks string is an S3 URI that is a prefix

    Args:
        s3_uri: User provided s3 URI

    Raises:
        ValueError: If validation fails
    """
    if not S3_URI_PREFIX_REGEX.match(s3_uri):
        raise ValueError(f"S3 URI must fit pattern {S3_URI_PREFIX_REGEX.pattern}")


def validate_endpoint_arn(endpoint_arn: str) -> None:
    """
    Validation method that checks endpoint arn is either a bedrock or sagemaker endpoint

    Args:
        endpoint_arn: User provided endpoint arn

    Raises:
        ValueError: If validation fails
    """
    if not (
        SAGEMAKER_ENDPOINT_ARN_REGEX.match(endpoint_arn)
        or BEDROCK_DEPLOYMENT_ARN_REGEX.match(endpoint_arn)
    ):
        raise ValueError(
            f"Endpoint must fit either SageMaker Endpoint pattern {SAGEMAKER_ENDPOINT_ARN_REGEX.pattern} "
            f"or Bedrock Deployment pattern {BEDROCK_DEPLOYMENT_ARN_REGEX.pattern}"
        )


def validate_sagemaker_environment_variables(
    env_vars: Dict[str, str],
    model: Optional[Model] = None,
    instance_type: Optional[str] = None,
) -> None:
    """
    Validation method that checks string is a SageMaker endpoint that is a prefix

    Args:
        env_vars: User provided environment variables
        model: Nova model being deployed (for SMI config validation)
        instance_type: SageMaker instance type (for SMI config validation)

    Raises:
        ValueError: If validation fails
    """
    # Define the list of accepted keys

    required_keys = {"CONTEXT_LENGTH", "MAX_CONCURRENCY"}

    optional_keys = {
        "DEFAULT_TEMPERATURE",
        "DEFAULT_TOP_P",
        "DEFAULT_TOP_K",
        "DEFAULT_MAX_NEW_TOKENS",
        "DEFAULT_LOGPROBS",
    }

    accepted_keys = required_keys.union(optional_keys)

    # Check if all provided keys are in the accepted keys
    for key in env_vars.keys():
        if key not in accepted_keys:
            raise ValueError(f"Invalid environment variable: {key}")

    # Check that required keys are present
    missing_keys = required_keys - set(env_vars.keys())
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")

    for key, value in env_vars.items():
        try:
            # Convert value to float for validation
            float_value = float(value)

            if key in [
                "CONTEXT_LENGTH",
                "MAX_CONCURRENCY",
                "DEFAULT_MAX_NEW_TOKENS",
            ]:
                if float_value <= 0 or not float_value.is_integer():
                    raise ValueError(f"{key} must be a positive integer")

            elif key == "DEFAULT_TOP_K":
                if float_value <= 0 or not float_value.is_integer():
                    raise ValueError(f"{key} must be an integer greater or equal to 1")

            elif key == "DEFAULT_TEMPERATURE":
                if float_value < 0 or float_value > 2.0:
                    raise ValueError(f"{key} must be between 0 and 2.0")

            elif key == "DEFAULT_TOP_P":
                if float_value < 1e-10 or float_value > 1.0:
                    raise ValueError(f"{key} must be between 1e-10 and 1.0")

            elif key == "DEFAULT_LOGPROBS":
                if float_value < 0:
                    raise ValueError(f"{key} must be a non-negative number")

        except ValueError as e:
            raise ValueError(f"Invalid value for {key}: {e}")

    # Validate CONTEXT_LENGTH and MAX_CONCURRENCY against supported SMI configs
    if model is not None and instance_type is not None:
        _validate_smi_config_bounds(env_vars, model, instance_type)


def _validate_smi_config_bounds(
    env_vars: Dict[str, str], model: Model, instance_type: str
) -> None:
    """Validate CONTEXT_LENGTH and MAX_CONCURRENCY against the supported SMI config table."""
    config_key = (model, instance_type)
    tiers = SUPPORTED_SMI_CONFIGS.get(config_key)

    if tiers is None:
        logger.warning(
            f"No SMI configuration found for ({model.name}, {instance_type}). "
            f"Skipping CONTEXT_LENGTH/MAX_CONCURRENCY bounds validation."
        )
        return

    sorted_tiers = sorted(tiers, key=lambda t: t[0])
    context_length = float(env_vars["CONTEXT_LENGTH"])
    max_concurrency = float(env_vars["MAX_CONCURRENCY"])

    # Find the applicable tier: smallest max_context_length >= user's context_length
    max_supported_context = sorted_tiers[-1][0]
    if context_length > max_supported_context:
        raise ValueError(
            f"CONTEXT_LENGTH={env_vars['CONTEXT_LENGTH']} exceeds maximum supported value "
            f"of {max_supported_context} for {model.name} on {instance_type}."
        )

    for tier_context, tier_concurrency in sorted_tiers:
        if context_length <= tier_context:
            if max_concurrency > tier_concurrency:
                raise ValueError(
                    f"MAX_CONCURRENCY={env_vars['MAX_CONCURRENCY']} exceeds maximum supported value "
                    f"of {tier_concurrency} for {model.name} on {instance_type} "
                    f"at CONTEXT_LENGTH={env_vars['CONTEXT_LENGTH']} (tier <={tier_context})."
                )
            return


def validate_unit_count(unit_count: Optional[int]) -> None:
    """
    Validation method that checks unit count is not None and is larger than 1

    Args:
        unit_count: User provided unit_count

    Raises:
        ValueError: If validation fails
    """
    if unit_count is None or unit_count < 1:
        raise ValueError("unit_count must be a positive integer value")
