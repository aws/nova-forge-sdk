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
from importlib import resources
from typing import Dict, List, Optional, Tuple

import boto3

from amzn_nova_customization_sdk.model.model_enums import DeployPlatform
from amzn_nova_customization_sdk.util.logging import logger

DEPLOYMENT_ARN_NAME = {
    DeployPlatform.BEDROCK_OD: "customModelDeploymentArn",
    DeployPlatform.BEDROCK_PT: "provisionedModelArn",
}

BEDROCK_EXECUTION_ROLE_NAME = "BedrockDeployModelExecutionRole"


# TODO: Move this functionality to extend BaseJobResult in the src/amzn_nova_customization_sdk/model folder
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


def create_bedrock_execution_role(
    iam_client, role_name: str, bedrock_resource: str = "*", s3_resource: str = "*"
) -> Dict:
    """
    Creates a new IAM Role that allows for Bedrock model creation and deployment.

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        bedrock_resource: Optional name of the bedrock resources that IAM role should have restricted create and get access to
        s3_resource: Optional name of additional s3 resources that IAM role should have restricted read access to such as the training output bucket

    Returns:
        Dict: The IAM role response containing role details

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts")
    with (
        resources.files("amzn_nova_customization_sdk.model")
        .joinpath("bedrock_policies.json")
        .open() as f
    ):
        policies = json.load(f)

    # Create a new execution role for creating and deploying the models.
    try:
        # Checks if the role exists already.
        bedrock_execution_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        bedrock_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for models to be created and deployed.",
        )
    except Exception as e:
        raise Exception(
            f"Failed to create the Bedrock execution role {role_name}: {str(e)}"
        )

    if bedrock_resource != "*":
        policies["bedrock_policy"]["Statement"][0]["Resource"] = (
            f"arn:aws:bedrock:*:*:custom-model/{bedrock_resource}*"
        )

    else:
        policies["bedrock_policy"]["Statement"][0]["Resource"] = "*"

    # S3 resources needed are the escrow bucket and the training output bucket
    if s3_resource != "*":
        account_id = sts_client.get_caller_identity()["Account"]

        policies["s3_read_policy"]["Statement"][0]["Resource"] = [
            f"arn:aws:s3:::{s3_resource}*",
            f"arn:aws:s3:::{s3_resource}*/*",
            f"arn:aws:s3:::customer-escrow-{account_id}*",
            f"arn:aws:s3:::customer-escrow-{account_id}*/*",
        ]
    else:
        policies["s3_read_policy"]["Statement"][0]["Resource"] = "*"

    # Create and attach policies
    for policy_name in ["bedrock_policy", "s3_read_policy"]:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

            logger.info(
                f"Creating {policy_name} with the following permissions {json.dumps(policies[policy_name])}."
            )

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except iam_client.exceptions.EntityAlreadyExistsException:
            # If the policy already exists, get its ARN and attach it to the role.
            logger.info(
                f"The {policy_name} already exists in your account, attaching it to the role now."
            )
            policy_arn = iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{sts_client.get_caller_identity()['Account']}:policy/{role_name}{policy_name.title()}"
            )["Policy"]["Arn"]

            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except Exception as e:
            raise Exception(
                f"Failed to create or attach policy {policy_name}: {str(e)}"
            )
    return bedrock_execution_role


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
