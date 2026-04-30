# Copyright Amazon.com, Inc. or its affiliates

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
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from amzn_nova_forge.core.constants import (
    ESCROW_URI_TAG_KEY,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    SUPPORTED_SMI_CONFIGS,
    _escrow_tag_value,
)
from amzn_nova_forge.core.enums import DeploymentMode, Model, Platform
from amzn_nova_forge.core.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge.core.runtime import RuntimeManager
from amzn_nova_forge.core.types import ModelArtifacts
from amzn_nova_forge.validation.endpoint_validator import (
    validate_s3_uri_prefix,
)

from .logging import logger

SAGEMAKER_EXECUTION_ROLE_NAME = "SageMakerDeployModelExecutionRole"


def register_lambda_as_hub_content(
    lambda_arn: str,
    hub_name: str,
    sagemaker_client: Any,
    evaluator_name: Optional[str] = None,
) -> str:
    """Register a Lambda ARN as a JsonDoc hub-content and return the hub-content ARN.

    The serverless API's EvaluatorArn field only accepts hub-content ARNs, not Lambda ARNs
    directly. This wraps the Lambda ARN in a JsonDoc document inside a private hub,
    creating the hub if it doesn't exist.

    The hub-content is upserted — if a document with the same name already exists at the
    same version, the existing ARN is returned so repeated train() calls are idempotent.

    Args:
        lambda_arn: A valid Lambda function ARN.
        hub_name: Name of the private hub to register into.
        sagemaker_client: Boto3 SageMaker client.
        evaluator_name: Optional human-readable name for the hub-content entry.
            Defaults to the Lambda function name derived from the ARN.

    Returns:
        The hub-content ARN that can be passed as EvaluatorArn.
    """
    # Use provided name or derive from Lambda function name
    if evaluator_name:
        content_name = re.sub(r"[^a-zA-Z0-9-]", "-", evaluator_name)[:63]
    else:
        content_name = re.sub(r"[^a-zA-Z0-9-]", "-", lambda_arn.split(":")[-1])[:63]
    content_version = "0.0.1"
    document = json.dumps(
        {
            "SubType": "AWS/Evaluator",
            "JsonContent": json.dumps(
                {
                    "EvaluatorType": "RewardFunction",
                    "Reference": lambda_arn,
                }
            ),
        }
    )

    # Ensure the hub exists
    try:
        sagemaker_client.describe_hub(HubName=hub_name)
    except sagemaker_client.exceptions.ResourceNotFound:
        logger.info(f"Creating private hub '{hub_name}' for reward function registration.")
        try:
            sagemaker_client.create_hub(
                HubName=hub_name,
                HubDescription="Private hub for Nova Forge serverless reward functions",
            )
        except sagemaker_client.exceptions.ResourceInUse:
            logger.info(f"Hub '{hub_name}' was created concurrently; proceeding.")

    # Upsert the JsonDoc hub-content
    try:
        resp = sagemaker_client.import_hub_content(
            HubName=hub_name,
            HubContentName=content_name,
            HubContentType="JsonDoc",
            HubContentVersion=content_version,
            DocumentSchemaVersion="2.0.0",
            HubContentDocument=document,
        )
        hub_content_arn = resp["HubContentArn"]
        logger.info(f"Registered Lambda as hub-content: {hub_content_arn}")
    except sagemaker_client.exceptions.ResourceInUse:
        # Version already exists — check if it still points to the same Lambda ARN.
        # If the user updated their Lambda to a different ARN, register a new version.
        # Retry up to 10 times, bumping the patch version on each ResourceInUse.
        major, minor, patch = content_version.split(".")
        hub_content_arn = None
        for attempt in range(10):
            bump_version = f"{major}.{minor}.{int(patch) + attempt}"
            existing = sagemaker_client.describe_hub_content(
                HubName=hub_name,
                HubContentName=content_name,
                HubContentType="JsonDoc",
                HubContentVersion=bump_version,
            )
            existing_doc = json.loads(existing["HubContentDocument"])
            existing_ref = json.loads(existing_doc.get("JsonContent", "{}")).get("Reference")

            if existing_ref == lambda_arn:
                hub_content_arn = existing["HubContentArn"]
                logger.info(f"Reusing existing hub-content: {hub_content_arn}")
                break

            # Lambda ARN changed — try the next version
            next_version = f"{major}.{minor}.{int(patch) + attempt + 1}"
            logger.info(f"Lambda ARN changed (was {existing_ref}), trying version {next_version}.")
            try:
                resp = sagemaker_client.import_hub_content(
                    HubName=hub_name,
                    HubContentName=content_name,
                    HubContentType="JsonDoc",
                    HubContentVersion=next_version,
                    DocumentSchemaVersion="2.0.0",
                    HubContentDocument=document,
                )
                hub_content_arn = resp["HubContentArn"]
                logger.info(f"Registered updated Lambda as hub-content: {hub_content_arn}")
                break
            except sagemaker_client.exceptions.ResourceInUse:
                # Another version already exists — keep bumping
                continue

        if hub_content_arn is None:
            raise RuntimeError(
                f"Could not register Lambda ARN as hub-content after 10 retries "
                f"(all versions 0.0.1–0.0.{int(patch) + 10} are in use)."
            )

    return hub_content_arn


def extract_lambda_arn_from_hub_content(
    hub_content_arn: str,
    sagemaker_client: Any,
) -> Optional[str]:
    """Extract the Lambda ARN stored inside a JsonDoc hub-content evaluator.

    Args:
        hub_content_arn: A SageMaker hub-content ARN.
        sagemaker_client: Boto3 SageMaker client.

    Returns:
        The Lambda ARN if found, or None if extraction fails.
    """
    try:
        # ARN: arn:aws:sagemaker:region:account:hub-content/hub/type/name/version
        resource = hub_content_arn.split(":", 5)[5]  # hub-content/hub/type/name/version
        _, hub_name, _, content_name, content_version = resource.split("/")
        resp = sagemaker_client.describe_hub_content(
            HubName=hub_name,
            HubContentName=content_name,
            HubContentType="JsonDoc",
            HubContentVersion=content_version,
        )
        doc = json.loads(resp["HubContentDocument"])
        inner = json.loads(doc.get("JsonContent", "{}"))
        return inner.get("Reference")
    except Exception as e:
        logger.warning(f"Could not extract Lambda ARN from hub-content '{hub_content_arn}': {e}")
        return None


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

    if infra.platform in (Platform.SMTJ, Platform.SMTJServerless):
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        # Serverless jobs populate OutputModelPackageArn; use it to get the checkpoint S3 URI
        # SMTJ jobs use CheckpointConfig.S3Uri
        checkpoint_s3_path = None
        model_package_arn = response.get("OutputModelPackageArn")
        if model_package_arn:
            # For serverless, get the S3 checkpoint URI from the model package directly
            try:
                pkg = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
                checkpoint_s3_path = (
                    pkg.get("InferenceSpecification", {})
                    .get("Containers", [{}])[0]
                    .get("ModelDataSource", {})
                    .get("S3DataSource", {})
                    .get("S3Uri")
                )
            except Exception as e:
                logger.warning(
                    "Failed to extract checkpoint path for serverless job '%s': %s",
                    job_name,
                    e,
                )
        if (
            not checkpoint_s3_path
            and "CheckpointConfig" in response
            and response["CheckpointConfig"]
        ):
            checkpoint_s3_path = response["CheckpointConfig"]["S3Uri"]
        return ModelArtifacts(
            checkpoint_s3_path=checkpoint_s3_path,
            output_s3_path=response["OutputDataConfig"]["S3OutputPath"],
            output_model_arn=model_package_arn,
        )
    elif infra.platform == Platform.SMHP:
        try:
            cluster_name = infra.cluster_name  # type: ignore[attr-defined]
        except AttributeError:
            raise ValueError("SMHPRuntimeManager requires cluster_name for get_model_artifacts")
        response = sagemaker_client.describe_cluster(ClusterName=cluster_name)
        rigs = response.get("RestrictedInstanceGroups", [])

        # If there's only one RIG in the cluster, we know that the job had to be submitted to that RIG
        checkpoint_s3_path = None
        if len(rigs) == 1:
            checkpoint_s3_path = rigs[0].get("EnvironmentConfig", {}).get("S3OutputPath")

        return ModelArtifacts(
            checkpoint_s3_path=checkpoint_s3_path,
            output_s3_path=output_s3_path,
        )
    else:
        raise ValueError(f"Unsupported platform: {infra.platform}")


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
        raise RuntimeError(f"Failed to get cluster instance info for {cluster_name}: {str(e)}")


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
                error_msg = f"\n\nERROR! Endpoint '{endpoint_name}' status is: {status}\n"
                logger.error(f"{error_msg}\nPlease check the AWS console for more details.\n")
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}\n")
            raise
        time.sleep(60)  # Sleep for a minute.


def _validate_sagemaker_instance_type_for_model_deployment(
    instance_type: str,
    model: Model,
    context_length: Optional[str] = None,
    max_concurrency: Optional[str] = None,
) -> None:
    """
    Validation method that checks the instance_type and if it is compatible with the desired model.
    Validates CONTEXT_LENGTH and MAX_CONCURRENCY against supported SMI configurations when both values are provided.

    Args:
        instance_type: instance type
        model: Model (enum)
        context_length: Optional CONTEXT_LENGTH value to validate
        max_concurrency: Optional MAX_CONCURRENCY value to validate

    Raises:
        ValueError: If validation fails

    """
    # Check if the model and instance combination is supported
    config_key = (model, instance_type)
    if config_key not in SUPPORTED_SMI_CONFIGS:
        # Collect all supported instance types for this model for error message
        supported_instances = [inst for (m, inst) in SUPPORTED_SMI_CONFIGS.keys() if m == model]
        if not supported_instances:
            raise ValueError(
                f"No supported instance types found for {model}. "
                f"Please check SUPPORTED_SMI_CONFIGS in constants.py"
            )
        raise ValueError(
            f"{instance_type} is not in the supported instances list for {model}: "
            f"{sorted(supported_instances)}"
        )

    # If context_length and max_concurrency are provided, validate SMI config bounds
    if context_length is not None and max_concurrency is not None:
        tiers = SUPPORTED_SMI_CONFIGS[config_key]
        context_length_val = int(context_length)
        max_concurrency_val = int(max_concurrency)

        for tier_context, tier_concurrency in tiers:
            if context_length_val <= tier_context and max_concurrency_val <= tier_concurrency:
                return

        # If no tier matches, raise an error with available options
        raise ValueError(
            f"CONTEXT_LENGTH={context_length} and MAX_CONCURRENCY={max_concurrency} "
            f"is not a supported configuration for {model.name} on {instance_type}. "
            f"Available tiers (max_context_length, max_concurrency): {tiers}"
        )


def create_sagemaker_model(
    region: str,
    model_name: str,
    model_s3_location: str,
    sagemaker_execution_role_arn: str,
    sagemaker_client: BaseClient,
    environment: Dict[str, Any] = {},
    network_isolation: bool = True,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    tags: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Create a SageMaker model resource.

    Args:
        region: AWS region
        model_name: Name of the SageMaker model
        model_s3_location: S3 URI where model artifacts are stored
        sagemaker_execution_role_arn: IAM role ARN for SageMaker execution
        sagemaker_client: SageMaker client
        environment: Environment variables for the model
        network_isolation: Enable network isolation
        deployment_mode: How to handle existing model

    Returns:
        str: Model ARN

    Raises:
        Exception: If model already exists (FAIL_IF_EXISTS) or creation fails
    """
    validate_s3_uri_prefix(s3_uri=model_s3_location)

    if deployment_mode in [
        DeploymentMode.FAIL_IF_EXISTS,
        DeploymentMode.UPDATE_IF_EXISTS,
    ]:
        try:
            sagemaker_client.describe_model(ModelName=model_name)
            raise Exception(f"Model '{model_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

    logger.info(f"Creating model: {model_name}...")
    create_kwargs = {
        "ModelName": model_name,
        "PrimaryContainer": {
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
        "ExecutionRoleArn": sagemaker_execution_role_arn,
        "EnableNetworkIsolation": network_isolation,
    }
    if tags:
        create_kwargs["Tags"] = tags
    model_response = sagemaker_client.create_model(**create_kwargs)
    logger.info(f"Model created successfully: {model_response['ModelArn']}")
    return model_response["ModelArn"]


def find_sagemaker_model_by_tag(escrow_uri: str, sagemaker_client: BaseClient) -> Optional[str]:
    """Find an existing SageMaker model tagged with the given escrow URI.

    Uses ResourceGroupsTaggingAPI for efficient tag-based lookup (single API call).
    Returns model ARN or None. Catches permission errors gracefully.
    """
    tag_value = _escrow_tag_value(escrow_uri)
    try:
        tagging_client = boto3.client(
            "resourcegroupstaggingapi",
            region_name=sagemaker_client.meta.region_name,
        )
        response = tagging_client.get_resources(
            TagFilters=[{"Key": ESCROW_URI_TAG_KEY, "Values": [tag_value]}],
            ResourceTypeFilters=["sagemaker:model"],
        )
        results = response.get("ResourceTagMappingList", [])
        if results:
            return results[0]["ResourceARN"]
    except ClientError as e:
        logger.warning(
            f"Could not search SageMaker models by tag (may lack tag:GetResources permission): {e}"
        )
    except Exception as e:
        logger.warning(f"Unexpected error searching SageMaker models: {e}")
    return None


def create_sagemaker_endpoint(
    model_name: str,
    endpoint_config_name: str,
    endpoint_name: str,
    instance_type: str,
    sagemaker_client: BaseClient,
    initial_instance_count: int = 1,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
) -> str:
    """Create a SageMaker endpoint config and endpoint.

    Args:
        model_name: Name of the existing SageMaker model
        endpoint_config_name: Name for the endpoint configuration
        endpoint_name: Name for the endpoint
        instance_type: EC2 instance type
        sagemaker_client: SageMaker client
        initial_instance_count: Number of instances
        deployment_mode: How to handle existing resources

    Returns:
        str: Endpoint ARN

    Raises:
        Exception: If resources exist (FAIL_IF_EXISTS) or creation fails
    """
    if deployment_mode in [
        DeploymentMode.FAIL_IF_EXISTS,
        DeploymentMode.UPDATE_IF_EXISTS,
    ]:
        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            raise Exception(f"Endpoint configuration '{endpoint_config_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            raise Exception(f"Endpoint '{endpoint_name}' already exists.")
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise

    logger.info(f"Creating endpoint configuration: {endpoint_config_name}...")
    config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
            }
        ],
    )
    logger.info(
        f"Endpoint configuration created successfully: {config_response['EndpointConfigArn']}"
    )

    logger.info(f"Creating endpoint: {endpoint_name}...")
    endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    logger.info("Waiting for endpoint creation to complete. This can take ~10 minutes...")
    try:
        _monitor_endpoint_creation(sagemaker_client, endpoint_name)
    except Exception as e:
        raise Exception(f"Failed to create deployment {endpoint_name}: {e}")

    return endpoint_response["EndpointArn"]


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
        logger.info(f"Invoking endpoint ({'streaming' if is_streaming else 'non-streaming'})...")

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
