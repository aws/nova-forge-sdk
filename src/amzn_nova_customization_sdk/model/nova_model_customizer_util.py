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
Utility functions used by nova_model_customizer
"""

from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.model.result import TrainingResult
from amzn_nova_customization_sdk.recipe.recipe_config import BYOD_AVAILABLE_EVAL_TASKS
from amzn_nova_customization_sdk.util.checkpoint_util import (
    extract_checkpoint_path_from_job_output,
)
from amzn_nova_customization_sdk.util.logging import logger


def set_output_s3_path(
    region: str, output_s3_path: Optional[str] = None, kms_key_id: Optional[str] = None
) -> str:
    """
    Constructs the output S3 path.

    Raises:
        ValueError: If unable to construct the output S3 path
    """
    s3_client = boto3.client("s3")
    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    # If no output S3 path is provided, use a default S3 bucket
    if output_s3_path is None:
        output_bucket = f"sagemaker-nova-{account_id}-{region}"
        output_s3_path = f"s3://{output_bucket}/output"
        try:
            s3_client.head_bucket(Bucket=output_bucket)
        except Exception:
            kms_arn = (
                f"arn:aws:kms:{region}:{account_id}:key/{kms_key_id}"
                if kms_key_id
                else None
            )
            create_s3_bucket(s3_client, output_bucket, kms_arn)
        logger.info(
            f"No output S3 bucket was provided. Using default output S3 bucket '{output_bucket}'."
        )
        return output_s3_path
    # If output S3 path is provided, check if the bucket exists. If it doesn't, try and create it.
    else:
        output_bucket = urlparse(output_s3_path).netloc
        try:
            s3_client.head_bucket(Bucket=output_bucket, ExpectedBucketOwner=account_id)
            return output_s3_path
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                kms_arn = (
                    f"arn:aws:kms:{region}:{account_id}:key/{kms_key_id}"
                    if kms_key_id
                    else None
                )
                create_s3_bucket(s3_client, output_bucket, kms_arn)
                return output_s3_path
            elif error_code in ("403", "Forbidden", "AccessDenied"):
                raise Exception(
                    f"Bucket '{output_bucket}' already exists, but is not owned by you. Please provide a different value for output_s3_path, or omit that parameter."
                )
            else:
                raise


def create_s3_bucket(
    s3_client: BaseClient, output_bucket: str, kms_key_arn: Optional[str] = None
) -> None:
    """
    Creates an S3 bucket

    Raises:
        Exception: If unable to create the S3 bucket
    """
    try:
        s3_client.create_bucket(Bucket=output_bucket)

        if kms_key_arn:
            s3_client.put_bucket_encryption(
                Bucket=output_bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "aws:kms",
                                "KMSMasterKeyID": kms_key_arn,
                            }
                        }
                    ]
                },
            )

            logger.info(
                f"Created '{output_bucket}' with SSE-S3 encryption using KMS key {kms_key_arn}."
            )
        else:
            logger.info(f"Created '{output_bucket}' with SSE-S3 encryption.")
    except Exception as e:
        raise Exception(f"Failed to create output bucket {output_bucket}: {str(e)}")


def resolve_model_checkpoint_path(
    model_path: Optional[str],
    job_result: Optional[TrainingResult],
    customizer_job_id: Optional[str],
    customizer_output_s3_path: Optional[str],
    customizer_model_path: Optional[str],
    fail_on_error: bool = False,
) -> Optional[str]:
    """
    Resolves the model checkpoint path using a fallback chain.

    Priority order:
    1. Explicit model_path parameter (if provided)
    2. Extract from job_result (if provided)
    3. Customizer's model_path (if set)
    4. Extract from customizer's most recent job (if job_id exists)

    Args:
        model_path: Explicitly provided model path
        job_result: Optional TrainingResult to extract checkpoint from
        customizer_job_id: Job ID from the customizer instance
        customizer_output_s3_path: Output S3 path from the customizer instance
        customizer_model_path: Model path from the customizer instance
        fail_on_error: If True, raises exception when path cannot be resolved. If False, logs warning and returns None.

    Returns:
        Optional[str]: Resolved model checkpoint path, or None if fail_on_error=False and no path found

    Raises:
        Exception: If fail_on_error=True and no path can be resolved or extraction fails
    """
    try:
        # 1. Use explicit model_path if provided
        if model_path is not None:
            return model_path

        # 2. Try to extract from job_result if provided
        if job_result is not None:
            return extract_checkpoint_path_from_job_output(
                output_s3_path=job_result.model_artifacts.output_s3_path,
                job_result=job_result,
            )

        # 3. Use customizer's model_path if set
        if customizer_model_path is not None:
            return customizer_model_path

        # 4. Try to extract from customizer's most recent job
        if customizer_job_id and customizer_output_s3_path:
            return extract_checkpoint_path_from_job_output(
                output_s3_path=customizer_output_s3_path, job_id=customizer_job_id
            )

        # No path could be resolved
        raise Exception(
            "No model path provided and no recent training job found. "
            "Please provide model_path or job_result parameter."
        )
    except Exception as e:
        if fail_on_error:
            raise
        logger.warning(f"Could not resolve model checkpoint path: {e}")
        return None


def requires_custom_eval_data(eval_task) -> bool:
    """
    Determines if an evaluation task requires custom (BYOD) data.

    Args:
        eval_task: The evaluation task to check (EvaluationTask enum)

    Returns:
        True if the task requires custom data, False otherwise
    """
    return eval_task.value in BYOD_AVAILABLE_EVAL_TASKS
