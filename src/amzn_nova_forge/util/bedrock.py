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
Helper functions for Bedrock model deployment, management, and recipe parsing.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.client import BaseClient

from amzn_nova_forge.model.model_enums import (
    DeployPlatform,
    Model,
    TrainingMethod,
)
from amzn_nova_forge.model.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge.util.logging import logger

DEPLOYMENT_ARN_NAME = {
    DeployPlatform.BEDROCK_OD: "customModelDeploymentArn",
    DeployPlatform.BEDROCK_PT: "provisionedModelArn",
}

BEDROCK_EXECUTION_ROLE_NAME = "BedrockDeployModelExecutionRole"

# Mapping from SDK TrainingMethod enum to Bedrock CustomizationType
# Only SFT_LORA and RFT_LORA are supported on Bedrock
METHOD_TO_CUSTOMIZATION_TYPE = {
    "sft_lora": "FINE_TUNING",
    "rft_lora": "REINFORCEMENT_FINE_TUNING",
}


# Mapping from SDK Model enum to Bedrock base model identifiers
# Note: Foundation model ARNs must match the region where the job is created
def get_bedrock_model_identifier(model_type: str, region: str = "us-east-1") -> str:
    """Get Bedrock foundation model ARN for a given model type and region.

    Args:
        model_type: Model type string (e.g., "amazon.nova-micro-v1:0:128k")
        region: AWS region (default: "us-east-1")

    Returns:
        Bedrock foundation model ARN
    """
    return f"arn:aws:bedrock:{region}::foundation-model/{model_type}"


# TODO: Move this functionality to extend BaseJobResult in the src/amzn_nova_forge/model folder
def monitor_model_create(client, model: dict, endpoint_name: str) -> str:
    """
    Monitors the status of a custom model creation in Bedrock.

    Args:
        client: The boto3 bedrock client used in the script
        model: Response dictionary from create_custom_model
        endpoint_name: The name of the model endpoint.

    Returns:
        str: Final status of the model ('ACTIVE' or raises exception)
    """
    start_time = datetime.now(timezone.utc)

    while True:
        try:
            curr_model = client.get_custom_model(modelIdentifier=model["modelArn"])
            current_status = curr_model["modelStatus"]
            elapsed_time = datetime.now(timezone.utc) - start_time

            logger.info(f"Status: {current_status} | Elapsed: {elapsed_time}")

            if current_status.upper() == "ACTIVE":
                logger.info(
                    f"\n\nSUCCESS! Model creation is complete! '{endpoint_name}' is now ACTIVE!"
                )
                logger.info(f"Total time elapsed: {elapsed_time}")
                logger.info(f"Model ARN: {model['modelArn']}\n\n")
                return current_status
            elif current_status.upper() in ["FAILED", "STOPPED"]:
                error_msg = (
                    f"\n\nERROR! Model '{endpoint_name}' status is: {current_status}\n"
                )
                logger.error(
                    f"{error_msg}\nPlease check the AWS console for more details.\n"
                )
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}\n")
            raise
        time.sleep(60)  # Sleep for a minute.


def check_deployment_status(
    deployment_arn: str, platform: DeployPlatform
) -> Optional[str]:
    """
    Checks the current status of a Bedrock deployment.

    Args:
        deployment_arn: The ARN of the deployment to check
        platform: The deployment platform (BEDROCK_OD or BEDROCK_PT)

    Raises:
        Exception: If unable to check deployment status
    """
    status = None

    bedrock_client = boto3.client("bedrock")
    if platform == DeployPlatform.BEDROCK_OD:
        try:
            status = bedrock_client.get_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )["status"]
            logger.info(
                "\nDEPLOYMENT STATUS UPDATE:\n"
                f"The current status of the on-demand deployment is: '{status}'\n"
                f"- Deployment ARN: {deployment_arn}"
            )
        except Exception as e:
            raise Exception(f"Failed to check deployment status: {e}.")

    elif platform == DeployPlatform.BEDROCK_PT:
        try:
            status = bedrock_client.get_provisioned_model_throughput(
                provisionedModelId=deployment_arn
            )["status"]
            logger.info(
                "\nDEPLOYMENT STATUS UPDATE:\n"
                f"The current status of the provisioned throughput deployment is: '{status}'\n"
                f"- Deployment ARN: {deployment_arn}"
            )
        except Exception as e:
            raise Exception(f"Failed to check deployment status: {e}.")

    return status


def get_required_bedrock_deletion_permissions(
    platform: DeployPlatform, deployment_arn: str
) -> List[Tuple[str, str]]:
    """
    Get required permissions for deleting a deployment.

    Args:
        platform: The deployment platform (BEDROCK_OD or BEDROCK_PT)
        deployment_arn: The ARN of the deployment to delete

    Returns:
        List of (action, resource) tuples for required permissions
    """
    if platform == DeployPlatform.BEDROCK_OD:
        return [("bedrock:DeleteCustomModelDeployment", deployment_arn)]
    elif platform == DeployPlatform.BEDROCK_PT:
        return [("bedrock:DeleteProvisionedModelThroughput", deployment_arn)]
    return []


def get_required_bedrock_update_permissions(
    platform: DeployPlatform, deployment_arn: str
) -> List[Tuple[str, str]]:
    """
    Get required permissions for updating a deployment.

    Note that updating deployments is currently only available
    for BEDROCK_PT, so for BEDROCK_OD this is a no-op.

    Args:
        platform: The deployment platform (BEDROCK_OD or BEDROCK_PT)
        deployment_arn: The ARN of the deployment to update

    Returns:
        List of (action, resource) tuples for required permissions
    """
    if platform == DeployPlatform.BEDROCK_PT:
        return [("bedrock:UpdateProvisionedModelThroughput", deployment_arn)]
    return []


def check_existing_deployment(
    endpoint_name: str, platform: DeployPlatform
) -> Optional[str]:
    """
    Check if a deployment with the given name exists.

    Args:
        endpoint_name: The name of the endpoint to check
        platform: The deployment platform (BEDROCK_OD or BEDROCK_PT)

    Returns:
        Optional[str]: The ARN of the existing deployment if found, None otherwise
    """
    bedrock_client = boto3.client("bedrock")

    try:
        if platform == DeployPlatform.BEDROCK_OD:
            response = bedrock_client.list_custom_model_deployments(
                nameContains=endpoint_name
            )
            for deployment in response.get("modelDeploymentSummaries", []):
                if deployment["customModelDeploymentName"] == endpoint_name:
                    return deployment["customModelDeploymentArn"]

        elif platform == DeployPlatform.BEDROCK_PT:
            response = bedrock_client.list_provisioned_model_throughputs(
                nameContains=endpoint_name
            )
            for deployment in response.get("provisionedModelSummaries", []):
                if deployment["provisionedModelName"] == endpoint_name:
                    return deployment["provisionedModelArn"]

    except Exception as e:
        logger.warning(
            f"Failed to check for existing deployment '{endpoint_name}': {e}"
        )
        return None

    return None


def delete_existing_deployment(
    deployment_arn: str, platform: DeployPlatform, endpoint_name: str
) -> None:
    """
    Delete an existing deployment and wait for completion.

    Args:
        deployment_arn: The ARN of the deployment to delete
        platform: The deployment platform (BEDROCK_OD or BEDROCK_PT)
        endpoint_name: The name of the endpoint (for logging)

    Raises:
        Exception: If deletion fails or times out
    """
    bedrock_client = boto3.client("bedrock")

    try:
        logger.info(f"Deleting existing deployment '{endpoint_name}'...")

        if platform == DeployPlatform.BEDROCK_OD:
            bedrock_client.delete_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )
        elif platform == DeployPlatform.BEDROCK_PT:
            bedrock_client.delete_provisioned_model_throughput(
                provisionedModelId=deployment_arn
            )

        # Wait for deletion to complete
        start_time = datetime.now(timezone.utc)
        max_wait_time = 600  # 10 minutes

        while True:
            elapsed = datetime.now(timezone.utc) - start_time
            if elapsed.total_seconds() > max_wait_time:
                raise Exception(f"Deletion timeout after {max_wait_time}s")

            try:
                if platform == DeployPlatform.BEDROCK_OD:
                    status = bedrock_client.get_custom_model_deployment(
                        customModelDeploymentIdentifier=deployment_arn
                    )["status"]
                elif platform == DeployPlatform.BEDROCK_PT:
                    status = bedrock_client.get_provisioned_model_throughput(
                        provisionedModelId=deployment_arn
                    )["status"]

                logger.info(f"Deletion status: {status} | Elapsed: {elapsed}")

                if status in ["DELETING"]:
                    time.sleep(30)
                    continue
                elif status in ["DELETED"]:
                    break
                else:
                    raise Exception(f"Unexpected status during deletion: {status}")

            except bedrock_client.exceptions.ResourceNotFoundException:
                # Deployment no longer exists - deletion complete
                break
            except Exception as e:
                if "ResourceNotFound" in str(e):
                    break
                raise

        logger.info(f"Successfully deleted deployment '{endpoint_name}'")

    except Exception as e:
        error_str = str(e).lower()

        # Check for commitment term errors
        if "commitment term" in error_str or "cannot be deleted" in error_str:
            raise Exception(
                f"Cannot delete Provisioned Throughput deployment '{endpoint_name}': "
                f"Deployment is still within commitment term. {e}"
            )

        # Generic error
        raise Exception(f"Failed to delete deployment '{endpoint_name}': {e}")


def update_provisioned_throughput_model(
    deployment_arn: str, new_model_arn: str, endpoint_name: str
) -> None:
    """
    Update a Provisioned Throughput deployment to use a new custom model.

    Args:
        deployment_arn: The ARN of the PT deployment to update
        new_model_arn: The ARN of the new custom model to associate
        endpoint_name: The name of the endpoint (for logging)

    Raises:
        Exception: If update fails
    """
    bedrock_client = boto3.client("bedrock")

    try:
        logger.info(f"Updating PT deployment '{endpoint_name}' to new model...")
        bedrock_client.update_provisioned_model_throughput(
            provisionedModelId=deployment_arn, desiredModelId=new_model_arn
        )
        logger.info(
            f"Successfully initiated PT deployment update for '{endpoint_name}'"
        )

    except Exception as e:
        raise Exception(f"Failed to update PT deployment '{endpoint_name}': {e}")


def invoke_model(
    model_id: str, request_body: Dict[str, str], bedrock_runtime: BaseClient
) -> SingleInferenceResult:
    current_time = datetime.now(timezone.utc)

    # TODO: Add support for invoke_model_with_response_stream
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id, body=json.dumps(request_body)
        )
        body = response["body"].read().decode("utf-8")

        return SingleInferenceResult(
            job_id=response["ResponseMetadata"]["RequestId"],
            inference_output_path="",
            started_time=current_time,
            streaming_response=None,
            nonstreaming_response=body,
        )

    except Exception as e:
        raise Exception(f"Failed invoke Bedrock model '{model_id}': {e}")


# ============================================================================
# Bedrock-specific recipe parsing utilities
# ============================================================================


def get_customization_type(training_method: TrainingMethod) -> str:
    """Get Bedrock CustomizationType for a given TrainingMethod.

    Maps SDK TrainingMethod enum values to Bedrock API CustomizationType strings.
    Only SFT and single-turn RFT methods are supported on Bedrock.

    Args:
        training_method: The training method from the recipe configuration

    Returns:
        Bedrock CustomizationType string ("FINE_TUNING" or "REINFORCEMENT_FINE_TUNING")

    Raises:
        ValueError: If the training method is not supported on Bedrock
    """
    # Use enum value as key to avoid circular import issues
    customization_type = METHOD_TO_CUSTOMIZATION_TYPE.get(training_method.value)

    if customization_type is None:
        supported_methods = ", ".join(METHOD_TO_CUSTOMIZATION_TYPE.keys())
        raise ValueError(
            f"Training method '{training_method.value}' is not supported on Bedrock. "
            f"Supported methods: {supported_methods}"
        )

    return customization_type


def resolve_base_model_identifier(
    recipe_path: str,
    base_model_identifier: Optional[str] = None,
    region: str = "us-east-1",
) -> str:
    """Resolve the Bedrock base model identifier.

    If base_model_identifier is provided, returns that value.
    Otherwise, extracts the model_type from the recipe configuration and maps it
    to the corresponding Bedrock base model ARN for the specified region.

    Args:
        recipe_path: Path to the local recipe YAML file
        base_model_identifier: Optional explicit base model identifier to use
        region: AWS region for the foundation model ARN (default: "us-east-1")

    Returns:
        Bedrock base model identifier (ARN)

    Raises:
        ValueError: If model cannot be determined or is not supported on Bedrock
        FileNotFoundError: If recipe file cannot be found
        Exception: If recipe parsing fails
    """
    import yaml

    # If base_model_identifier was explicitly provided, use it
    if base_model_identifier is not None:
        logger.info(
            f"Using explicitly provided base_model_identifier: {base_model_identifier}"
        )
        return base_model_identifier

    # Otherwise, extract model from recipe config
    try:
        with open(recipe_path, "r") as f:
            recipe_config = yaml.safe_load(f)

        # Extract model_type from recipe config
        if "run" not in recipe_config or "model_type" not in recipe_config["run"]:
            raise ValueError(
                f"Recipe configuration at '{recipe_path}' is missing required 'run.model_type' field"
            )

        model_type = recipe_config["run"]["model_type"]

        # Validate model_type is supported (check against Model enum)
        supported_models = [model.model_type for model in Model]
        if model_type not in supported_models:
            raise ValueError(
                f"Model type '{model_type}' is not supported on Bedrock. "
                f"Supported models: {', '.join(supported_models)}"
            )

        # Build region-specific Bedrock identifier
        bedrock_identifier = get_bedrock_model_identifier(model_type, region)
        logger.info(f"Resolved base model identifier: {bedrock_identifier}")
        return bedrock_identifier

    except FileNotFoundError:
        logger.error(f"Recipe file not found: {recipe_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse recipe configuration from '{recipe_path}': {e}")
        raise


def parse_bedrock_recipe_config(
    recipe_path: str, method: TrainingMethod
) -> Dict[str, Any]:
    """Extract hyperparameters from a built recipe YAML for Bedrock API.

    This function reads a recipe that has already been built and validated by RecipeBuilder,
    and extracts the hyperparameters needed for the Bedrock API call.

    For SFT jobs:
        - Extracts top-level hyperparameters from training_config (excluding nested dicts)
        - Returns them in 'hyperparameters' key for top-level API parameter

    For RFT jobs:
        - Extracts hyperparameters from training_config.rft (excluding graderConfig)
        - Returns them in 'rft_hyperparameters' key for customizationConfig.rftConfig.hyperParameters

    Args:
        recipe_path: Path to the built recipe YAML file (output from RecipeBuilder)
        method: Training method (SFT_LORA, RFT_LORA, etc.) to determine extraction strategy

    Returns:
        Dict with keys:
            - 'hyperparameters': Dict of SFT hyperparameters (empty for RFT)
            - 'rft_hyperparameters': Dict of RFT hyperparameters (empty for SFT)
            - 'recipe_config': Full parsed recipe configuration

    Example:
        >>> # SFT job
        >>> data = parse_bedrock_recipe_config("sft_recipe.yaml", TrainingMethod.SFT_LORA)
        >>> data['hyperparameters']
        {'epochCount': '1', 'batchSize': '128', 'learningRate': '0.00001'}
        >>> data['rft_hyperparameters']
        {}

        >>> # RFT job
        >>> data = parse_bedrock_recipe_config("rft_recipe.yaml", TrainingMethod.RFT_LORA)
        >>> data['hyperparameters']
        {}
        >>> data['rft_hyperparameters']
        {'epochCount': '2', 'batchSize': '64', 'learningRate': '0.0001'}
    """
    import yaml

    # Load the built recipe
    with open(recipe_path, "r") as f:
        recipe_config = yaml.safe_load(f)

    hyperparameters: Dict[str, str] = {}
    rft_hyperparameters: Dict[str, Any] = {}  # RFT hyperparameters use native types

    if not recipe_config or "training_config" not in recipe_config:
        logger.warning(
            "No training_config found in recipe - using empty hyperparameters"
        )
        return {
            "hyperparameters": hyperparameters,
            "rft_hyperparameters": rft_hyperparameters,
            "recipe_config": recipe_config,
        }

    training_config = recipe_config["training_config"]

    # Determine if this is an RFT job
    is_rft = method == TrainingMethod.RFT_LORA

    if is_rft and "rft" in training_config and isinstance(training_config["rft"], dict):
        # For RFT: Extract hyperparameters from training_config.rft
        # Exclude graderConfig as it's provided via rft_lambda_arn
        rft_config = training_config["rft"]
        for key, value in rft_config.items():
            # Skip graderConfig and nested dicts
            if key != "graderConfig" and not isinstance(value, dict):
                # RFT hyperparameters must be native types (int, float, str)
                # Do NOT convert to string - Bedrock expects native types for RFT
                rft_hyperparameters[key] = value
    else:
        # For SFT: Extract top-level hyperparameters from training_config
        # Exclude nested dicts like 'trainer', 'rft', 'rollout', etc.
        for key, value in training_config.items():
            # Skip method field and nested dicts
            if key != "method" and not isinstance(value, dict):
                # Bedrock API requires all hyperparameters to be strings
                hyperparameters[key] = str(value)

    return {
        "hyperparameters": hyperparameters,
        "rft_hyperparameters": rft_hyperparameters,
        "recipe_config": recipe_config,
    }


def get_bedrock_job_details(bedrock_client, job_id: str) -> Dict[str, Any]:
    """
    Get detailed Bedrock job information including status, phases, and metrics.
    This is a shared utility used by both BedrockStatusManager and BedrockStrategy.

    Args:
        bedrock_client: Boto3 Bedrock client
        job_id: Job identifier (ARN or name)

    Returns:
        Dict containing job details from get_model_customization_job API

    Raises:
        Exception: If unable to retrieve job details
    """
    return bedrock_client.get_model_customization_job(jobIdentifier=job_id)


def log_bedrock_job_status(response: Dict[str, Any]) -> None:
    """
    Log detailed Bedrock job status information.
    Shared utility for displaying job status, phases, metrics, and failure messages.

    Args:
        response: Response from get_model_customization_job API
    """
    status = response.get("status", "Unknown")
    job_name = response.get("jobName", "N/A")

    logger.info(f"\nCurrent Job Status:")
    logger.info(f"  Job Name: {job_name}")
    logger.info(f"  Status: {status}")

    # Show phase details if available
    if "statusDetails" in response:
        details = response["statusDetails"]
        logger.info(f"  Phase Details:")

        if "validationDetails" in details:
            val_status = details["validationDetails"].get("status", "N/A")
            logger.info(f"    Validation: {val_status}")

        if "dataProcessingDetails" in details:
            dp_status = details["dataProcessingDetails"].get("status", "N/A")
            logger.info(f"    Data Processing: {dp_status}")

        if "trainingDetails" in details:
            train_status = details["trainingDetails"].get("status", "N/A")
            logger.info(f"    Training: {train_status}")

    # Show metrics if available
    if "trainingMetrics" in response:
        train_loss = response["trainingMetrics"].get("trainingLoss")
        if train_loss is not None:
            logger.info(f"  Training Loss: {train_loss:.6f}")

    if "validationMetrics" in response and response["validationMetrics"]:
        val_loss = response["validationMetrics"][0].get("validationLoss")
        if val_loss is not None:
            logger.info(f"  Validation Loss: {val_loss:.6f}")

    # Show failure message if job failed
    if status == "Failed":
        failure_msg = response.get("failureMessage", "No failure message available")
        logger.error(f"  Failure Message: {failure_msg}")
