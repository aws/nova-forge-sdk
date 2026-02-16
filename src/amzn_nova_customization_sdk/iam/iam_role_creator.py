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
from importlib import resources
from typing import Any, Dict, Optional

import boto3

from amzn_nova_customization_sdk.util.logging import logger


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
        resources.files("amzn_nova_customization_sdk.iam")
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


def create_sagemaker_execution_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    kms_resource: str = "*",
    ec2_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_metric_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_logstream_resource: str = "*",
    cloudwatch_loggroup_resource: str = "*",
) -> Dict:
    """
    Creates a new IAM Role that allows for SageMaker model creation and deployment.

    Args:
        iam_client: The boto3 client to use when creating the role.
        role_name: The name of the role to create.
        s3_resource: Optional name of additional s3 resources that IAM role should have restricted read access to such as the training output bucket
        kms_resource: Optional name of KMS resource that IAM role should have restricted access to
        ec2_condition: Optional condition for the EC2 resource to limit access
    Returns:
        Dict: The IAM role response containing role details

    Raises:
        Exception: If it fails at creating the new role.
    """
    sts_client = boto3.client("sts")
    with (
        resources.files("amzn_nova_customization_sdk.iam")
        .joinpath("sagemaker_policies.json")
        .open() as f
    ):
        policies = json.load(f)

    # Create a new execution role for creating and deploying the models.
    try:
        # Checks if the role exists already.
        sagemaker_execution_role = iam_client.get_role(RoleName=role_name)
    except iam_client.exceptions.NoSuchEntityException:
        logger.info(f"The {role_name} role doesn't exist. Creating it now...")
        sagemaker_execution_role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(policies["trust_policy"]),
            Description="This role allows for models to be created and deployed to SageMaker.",
        )
    except Exception as e:
        raise Exception(
            f"Failed to create the SageMaker execution role {role_name}: {str(e)}"
        )

    # Allow KMS access
    policies["kms_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:kms:*:*:key/{kms_resource}*" if kms_resource != "*" else "*"
    )

    # Add condition to EC2 policy
    if ec2_condition is not None:
        policies["ec2_policy"]["Statement"][0]["Condition"] = ec2_condition

    if cloudwatch_metric_condition is not None:
        policies["cloudwatch_metric_policy"]["Statement"][0]["Condition"] = (
            cloudwatch_metric_condition
        )

    policies["cloudwatch_logstream_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:logs:*:*:log-group:{cloudwatch_loggroup_resource}:log-stream:{cloudwatch_logstream_resource}"
        if cloudwatch_logstream_resource != "*"
        else "*"
    )

    policies["cloudwatch_loggroup_policy"]["Statement"][0]["Resource"] = (
        f"arn:aws:logs:*:*:log-group:{cloudwatch_loggroup_resource}*"
        if cloudwatch_loggroup_resource != "*"
        else "*"
    )

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
    for policy_name in [
        "cloudwatch_ec2_ec2_policy",
        "cloudwatch_metric_policy",
        "cloudwatch_logstream_policy",
        "ecr_read_policy",
        "s3_read_policy",
        "kms_policy",
        "ec2_policy",
    ]:
        try:
            logger.info(f"{json.dumps(policies[policy_name])}.")

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
    return sagemaker_execution_role
