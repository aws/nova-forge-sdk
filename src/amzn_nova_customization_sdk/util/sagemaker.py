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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import (
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    ModelArtifacts,
)
from amzn_nova_customization_sdk.model.model_enums import DeploymentMode, Model
from amzn_nova_customization_sdk.model.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_customization_sdk.validation.endpoint_validator import (
    validate_s3_uri_prefix,
)

from .logging import logger

# Model Parameters
DEFAULT_CONTEXT_LENGTH = "12000"
DEFAULT_MAX_CONCURRENCY = "16"

SAGEMAKER_EXECUTION_ROLE_NAME = "SageMakerDeployModelExecutionRole"


def _get_sagemaker_inference_image(region: str) -> str:
    if region not in REGION_TO_ESCROW_ACCOUNT_MAPPING:
        raise ValueError(
            f"Unsupported region: {region}. Supported regions are: {list(REGION_TO_ESCROW_ACCOUNT_MAPPING.keys())}"
        )

    return f"{REGION_TO_ESCROW_ACCOUNT_MAPPING[region]}.dkr.ecr.{region}.amazonaws.com/nova-inference-repo:SM-Inference-latest"


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


# TODO: Update environment variables when variables finalized
def setup_environment_variables(
    context_length: str = DEFAULT_CONTEXT_LENGTH,
    max_concurrency: str = DEFAULT_MAX_CONCURRENCY,
    temperature: Optional[str] = None,
    top_p: Optional[str] = None,
    top_k: Optional[str] = None,
    max_new_tokens: Optional[str] = None,
    logprobs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Set up environment variables for model configuration.

    Args:
        context_length (str, optional): Context length. Defaults to DEFAULT_CONTEXT_LENGTH.
        max_concurrency (str, optional): Maximum number of concurrency. Defaults to DEFAULT_MAX_CONCURRENCY.
        temperature (str, optional): Sampling temperature for text generation. Defaults to None.
        top_p (str, optional): Nucleus sampling probability threshold. Defaults to None.
        top_k (str, optional): Top-k sampling parameter. Defaults to None.
        max_new_tokens (str, optional): Maximum number of new tokens to generate. Defaults to None.
        logprobs (str, optional): Number of log probabilities to return. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary of environment variables for model configuration.
    """
    environment = {"CONTEXT_LENGTH": context_length, "MAX_CONCURRENCY": max_concurrency}

    # Add optional parameters if provided
    if temperature is not None:
        environment["DEFAULT_TEMPERATURE"] = temperature
    if top_p is not None:
        environment["DEFAULT_TOP_P"] = top_p
    if top_k is not None:
        environment["DEFAULT_TOP_K"] = top_k
    if max_new_tokens is not None:
        environment["DEFAULT_MAX_NEW_TOKENS"] = max_new_tokens
    if logprobs is not None:
        environment["DEFAULT_LOGPROBS"] = logprobs

    return environment


def _monitor_endpoint_creation(sagemaker_client: BaseClient, endpoint_name: str) -> str:
    """
    Monitors the status of a custom endpoint creation in SageMaker.

    Args:
        sagemaker_client: The boto3 sagemaker client used in the script
        endpoint_name: The name of the model endpoint.

    Returns:
        str: Final status of the model ('INSERVICE' or raises exception)
    """
    start_time = datetime.now(timezone.utc)

    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            elapsed_time = datetime.now(timezone.utc) - start_time

            logger.info(f"Status: {status} | Elapsed: {elapsed_time}")

            if status.upper() == "INSERVICE":
                logger.info(
                    f"\n\nSUCCESS! Endpoint creation is complete! '{endpoint_name}' is now INSERVICE!"
                )
                logger.info(f"Total time elapsed: {elapsed_time}")
                return status
            elif status.upper() in ["FAILED"]:
                error_msg = (
                    f"\n\nERROR! Endpoint '{endpoint_name}' status is: {status}\n"
                )
                logger.error(
                    f"{error_msg}\nPlease check the AWS console for more details.\n"
                )
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}\n")
            raise
        time.sleep(60)  # Sleep for a minute.


# TODO: This will be replaced by Hub content as source of truth
def _validate_sagemaker_instance_type_for_model_deployment(
    instance_type: str, model: Model
) -> None:
    """
    Validation method that checks the instance_type and if it is compatible with the desired model

    Args:
        instance_type: instance type
        model: Model (enum)

    Raises:
        ValueError: If validation fails

    """

    accepted_configs = {
        Model.NOVA_MICRO: [
            "ml.g5.12xlarge",
            "ml.g6.12xlarge",
            "ml.g5.48xlarge",
            "ml.g6.48xlarge",
            "ml.p5.48xlarge",
        ],
        Model.NOVA_LITE: [
            "ml.g5.12xlarge",
            "ml.g6.12xlarge",
            "ml.g5.48xlarge",
            "ml.g6.48xlarge",
            "ml.p5.48xlarge",
        ],
        Model.NOVA_LITE_2: ["ml.p5.48xlarge"],
        Model.NOVA_PRO: ["ml.g6.48xlarge", "ml.p5.48xlarge"],
    }

    if instance_type not in accepted_configs[model]:
        raise ValueError(
            f"{instance_type} is not in the supported instances list for {model}: {accepted_configs[model]}"
        )


def create_model_and_endpoint_config(
    region: str,
    model_name: str,
    model_s3_location: str,
    sagemaker_execution_role_arn: str,
    endpoint_config_name: str,
    endpoint_name: str,
    sagemaker_client: BaseClient,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    instance_type: Optional[str] = "ml.g5.4xlarge",
    environment: Dict[str, Any] = {},
    initial_instance_count: Optional[int] = 1,
    network_isolation: bool = True,
) -> str:
    """
    Create a SageMaker model, endpoint configuration, and endpoint.
    If DeploymentMode is FAIL_IF_EXISTS, deployment will fail if model, endpoint configuration or endpoint already exist.
    DeploymentMode UPDATE_IF_EXISTS is not supported as model and endpoint configuration do not support updates.

    Args:
        region (str): AWS region
        model_name (str): Name of the SageMaker model.
        model_s3_location (str): S3 URI where the model artifacts are stored.
        sagemaker_execution_role_arn (str): IAM role ARN for SageMaker execution.
        endpoint_config_name (str): Name for the endpoint configuration.
        endpoint_name (str): Name for the SageMaker endpoint.
        instance_type (str): EC2 instance type for the endpoint.
        environment (Dict[str, Any]): Environment variables for the model.
        sagemaker_client (BaseClient): SageMaker client
        deployment_mode (DeploymentMode): How to handle existing model, endpoint configs and endpoints
        initial_instance_count (int, optional): Number of instances for the endpoint. Defaults to 1.
        network_isolation (bool, optional): Enable network isolation. Defaults to True.
    Returns:
        str: endpoint ARN

    Raises:
        Exception: If there's an error creating the model, endpoint config, or endpoint.
    """

    validate_s3_uri_prefix(s3_uri=model_s3_location)

    try:
        # Check for existing resources based on deployment mode
        existing_model = None
        existing_endpoint_config = None
        existing_endpoint = None

        try:
            existing_model = sagemaker_client.describe_model(ModelName=model_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pass
            else:
                raise

        try:
            existing_endpoint_config = sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pass
            else:
                raise

        try:
            existing_endpoint = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                pass
            else:
                raise

        # Handle existing resources based on deployment mode
        if deployment_mode in [
            DeploymentMode.FAIL_IF_EXISTS,
            DeploymentMode.UPDATE_IF_EXISTS,
        ]:
            if existing_model:
                raise Exception(f"Model '{model_name}' already exists.")
            if existing_endpoint_config:
                raise Exception(
                    f"Endpoint configuration '{endpoint_config_name}' already exists."
                )
            if existing_endpoint:
                raise Exception(f"Endpoint '{endpoint_name}' already exists.")

        logger.info(f"Creating model: {model_name}...")
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": _get_sagemaker_inference_image(region),
                "ModelDataSource": {
                    "S3DataSource": {
                        "S3Uri": model_s3_location,
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                    }
                },
                "Environment": environment,
            },
            ExecutionRoleArn=sagemaker_execution_role_arn,
            EnableNetworkIsolation=network_isolation,
        )
        logger.info(f"Model created successfully: {model_response['ModelArn']}")

        production_variant = {
            "VariantName": "primary",
            "ModelName": model_name,
            "InitialInstanceCount": initial_instance_count,
            "InstanceType": instance_type,
        }

        logger.info(f"Creating endpoint configuration: {endpoint_config_name}...")
        # Create endpoint configuration
        config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[production_variant],
        )
        logger.info(
            f"Endpoint configuration created successfully: {config_response['EndpointConfigArn']}"
        )

        logger.info(f"Creating endpoint: {endpoint_name}...")
        endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )

        # Poll for endpoint status
        logger.info(
            "Waiting for endpoint creation to complete. This can take ~10 minutes..."
        )
        try:
            _monitor_endpoint_creation(sagemaker_client, endpoint_name)
        except Exception as e:
            raise Exception(f"Failed to create deployment {endpoint_name}: {e}")

        return endpoint_response["EndpointArn"]

    except Exception as e:
        logger.info(f"Error creating model and endpoint: {e}")
        raise


def invoke_sagemaker_inference(
    request_body: Dict[str, Any],
    endpoint_name: str,
    sagemaker_client: BaseClient,
) -> SingleInferenceResult:
    """
     Invoke Sagemaker inference and return result

    Args:
        request_body (Dict[str, Any]): The payload to send to the inference endpoint.
        endpoint_name (str): Name of the SageMaker inference endpoint.
        sagemaker_client (BaseClient): Sagemaker client

    Returns:
        - Generator[str, None, None] for streaming responses
        - str for non-streaming responses
    """
    current_time = datetime.now(timezone.utc)

    body = json.dumps(request_body)
    is_streaming = request_body.get("stream", False)

    try:
        logger.info(
            f"Invoking endpoint ({'streaming' if is_streaming else 'non-streaming'})..."
        )

        if is_streaming:
            response = sagemaker_client.invoke_endpoint_with_response_stream(
                EndpointName=endpoint_name, ContentType="application/json", Body=body
            )

            event_stream = response["Body"]

            def stream_generator():
                for event in event_stream:
                    if "PayloadPart" in event:
                        chunk = event["PayloadPart"]
                        if "Bytes" in chunk:
                            data = chunk["Bytes"].decode()
                            yield data

            return SingleInferenceResult(
                job_id=response["ResponseMetadata"]["RequestId"],
                inference_output_path="",
                started_time=current_time,
                streaming_response=stream_generator(),
                nonstreaming_response=None,
            )
        else:
            response = sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )

            body_content = json.loads(response["Body"].read().decode("utf-8"))

            return SingleInferenceResult(
                job_id=body_content["id"],
                inference_output_path="",
                started_time=datetime.fromtimestamp(body_content["created"]),
                streaming_response=None,
                nonstreaming_response=body_content["choices"],
            )

    except Exception as e:
        raise Exception(f"Error invoking endpoint {endpoint_name}: {str(e)}")
