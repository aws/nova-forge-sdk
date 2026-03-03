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
Helper functions for Bedrock model deployment and management.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.client import BaseClient

from amzn_nova_forge_sdk.model.model_enums import DeployPlatform
from amzn_nova_forge_sdk.model.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge_sdk.util.logging import logger

DEPLOYMENT_ARN_NAME = {
    DeployPlatform.BEDROCK_OD: "customModelDeploymentArn",
    DeployPlatform.BEDROCK_PT: "provisionedModelArn",
}

BEDROCK_EXECUTION_ROLE_NAME = "BedrockDeployModelExecutionRole"


# TODO: Move this functionality to extend BaseJobResult in the src/amzn_nova_forge_sdk/model folder
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
