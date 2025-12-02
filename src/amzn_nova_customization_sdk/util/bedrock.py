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
from typing import Dict, Optional

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

            logger.info(f"📊 Status: {current_status} | Elapsed: {elapsed_time}")

            if current_status.upper() == "ACTIVE":
                logger.info(
                    f"\n\n✅ SUCCESS! Model creation is complete! '{endpoint_name}' is now ACTIVE!"
                )
                logger.info(f"🎉 Total time elapsed: {elapsed_time}")
                logger.info(f"🔗 Model ARN: {model['modelArn']}\n\n")
                return current_status
            elif current_status.upper() in ["FAILED", "STOPPED"]:
                error_msg = f"\n\n❌ ERROR! Model '{endpoint_name}' status is: {current_status}\n"
                logger.error(
                    f"{error_msg}\nPlease check the AWS console for more details.\n"
                )
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"⚠️  Error checking status: {str(e)}\n")
            raise
        time.sleep(60)  # Sleep for a minute.


def create_bedrock_execution_role(iam_client, role_name: str) -> Dict:
    """
    Creates a new IAM Role that allows for Bedrock model creation and deployment.

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.

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

    # Create and attach policies
    for policy_name in ["bedrock_policy", "passrole_policy", "s3_read_policy"]:
        try:
            policy_arn = iam_client.create_policy(
                PolicyName=f"{role_name}{policy_name.title()}",
                PolicyDocument=json.dumps(policies[policy_name]),
            )["Policy"]["Arn"]

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
