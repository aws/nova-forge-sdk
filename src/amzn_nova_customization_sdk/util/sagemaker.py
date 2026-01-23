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
Helper functions for Sagemaker management.
"""

import json
from typing import Any, Dict, List, Optional

import boto3

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import ModelArtifacts


def get_model_artifacts(
    job_name: str, infra: RuntimeManager, output_s3_path: str
) -> ModelArtifacts:
    """
    Retrieve model artifacts for a job

    Args:
        job_name: Name of the job
        infra: Infrastructure of the job
        output_s3_path: Output S3 path of the job (only necessary for HyperPod)

    Returns:
        ModelArtifacts: Model artifact S3 paths

    Raises:
        Exception: If unable to obtain job artifact information
    """
    sagemaker_client = boto3.client("sagemaker")

    if isinstance(infra, SMTJRuntimeManager):
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)

        return ModelArtifacts(
            checkpoint_s3_path=response["CheckpointConfig"]["S3Uri"],
            output_s3_path=response["OutputDataConfig"]["S3OutputPath"],
        )
    # TODO: Figure out a reliable way to determine the RIG of a given job
    elif isinstance(infra, SMHPRuntimeManager):
        response = sagemaker_client.describe_cluster(ClusterName=infra.cluster_name)
        rigs = response.get("RestrictedInstanceGroups", [])

        # If there's only one RIG in the cluster, we know that the job had to be submitted to that RIG
        checkpoint_s3_path = None
        if len(rigs) == 1:
            checkpoint_s3_path = (
                rigs[0].get("EnvironmentConfig", {}).get("S3OutputPath")
            )

        return ModelArtifacts(
            checkpoint_s3_path=checkpoint_s3_path,
            output_s3_path=output_s3_path,
        )
    else:
        raise ValueError(f"Unsupported platform")


def get_cluster_instance_info(
    cluster_name: str, region: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get instance types and counts from a HyperPod cluster.

    Args:
        cluster_name: Name of the HyperPod cluster
        region: AWS region (optional, uses default session region if not provided)

    Returns:
        Dict with 'normal_instance_groups' and 'restricted_instance_groups' keys

    Raises:
        Exception: If unable to describe the cluster
    """
    if region is None:
        sagemaker_client = boto3.client("sagemaker")
    else:
        sagemaker_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker_client.describe_cluster(ClusterName=cluster_name)

        normal_instance_groups = []
        restricted_instance_groups = []

        # Process normal instance groups
        for group in response.get("InstanceGroups", []):
            group_info = {
                "instance_group_name": group["InstanceGroupName"],
                "instance_type": group["InstanceType"],
                "current_count": group["CurrentCount"],
                "target_count": group["TargetCount"],
                "status": group["Status"],
            }
            normal_instance_groups.append(group_info)

        # Process restricted instance groups
        for group in response.get("RestrictedInstanceGroups", []):
            group_info = {
                "instance_group_name": group["InstanceGroupName"],
                "instance_type": group["InstanceType"],
                "current_count": group["CurrentCount"],
                "target_count": group["TargetCount"],
                "status": group["Status"],
            }
            restricted_instance_groups.append(group_info)

        return {
            "normal_instance_groups": normal_instance_groups,
            "restricted_instance_groups": restricted_instance_groups,
        }

    except Exception as e:
        raise RuntimeError(
            f"Failed to get cluster instance info for {cluster_name}: {str(e)}"
        )


def _get_hub_content(
    hub_name: str,
    hub_content_name: str,
    hub_content_type: str,
    region: str,
) -> Dict[str, Any]:
    """
     Get hub content from SageMaker via the DescribeHubContent API

    Args:
        hub_name: Name of the SageMaker Hub
        hub_content_name: Name of the hub content
        hub_content_type: Type of hub content
        region: AWS region

    Returns:
        Dict containing hub content
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker_client.describe_hub_content(
            HubName=hub_name,
            HubContentType=hub_content_type,
            HubContentName=hub_content_name,
        )

        # Parse HubContentDocument if it's a JSON string
        if "HubContentDocument" in response:
            hub_content_document = response["HubContentDocument"]
            if isinstance(hub_content_document, str):
                try:
                    response["HubContentDocument"] = json.loads(hub_content_document)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, leave the string as is
                    pass

    except Exception as e:
        raise RuntimeError(
            f"Failed to get SageMaker hub content for '{hub_content_name}': {str(e)}"
        )

    return response
