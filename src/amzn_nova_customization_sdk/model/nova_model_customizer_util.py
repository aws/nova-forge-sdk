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
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import (
    IMAGE_REPO_REGISTRY,
    METHOD_IMAGE_REGISTRY,
    REGION_TO_ESCROW_ACCOUNT_MAPPING,
    RUNTIME_PREFIX_REGISTRY,
)
from amzn_nova_customization_sdk.model.model_enums import (
    Platform,
    TrainingMethod,
    Version,
)
from amzn_nova_customization_sdk.util.logging import logger


def set_output_s3_path(region: str, output_s3_path: Optional[str] = None) -> str:
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
            try:
                s3_client.create_bucket(Bucket=output_bucket)
            except Exception as e:
                raise Exception(
                    f"Failed to create output bucket {output_bucket}: {str(e)}"
                )
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
            if error_code == "NoSuchBucket":
                try:
                    s3_client.create_bucket(Bucket=output_bucket)
                    return output_s3_path
                except Exception as ce:
                    raise Exception(
                        f"Failed to create output bucket {output_bucket}: {str(ce)}"
                    )
            elif error_code in ("403", "Forbidden", "AccessDenied"):
                raise Exception(
                    f"Bucket '{output_bucket}' already exists, but is not owned by you. Please provide a different value for output_s3_path, or omit that parameter."
                )
            else:
                raise


def set_image_uri(
    region: str, method: TrainingMethod, version: Version, infra: RuntimeManager
) -> str:
    """
    Constructs the image URI.

    Raises:
        ValueError: If infrastructure manager type is invalid
    """
    prefix = (
        f"{REGION_TO_ESCROW_ACCOUNT_MAPPING[region]}"
        f".dkr.ecr.{region}.amazonaws.com/{IMAGE_REPO_REGISTRY[method]}:"
    )

    if isinstance(infra, SMTJRuntimeManager):
        platform_infix = RUNTIME_PREFIX_REGISTRY["SMTJRuntimeManager"]
        platform = Platform.SMTJ
    elif isinstance(infra, SMHPRuntimeManager):
        platform_infix = RUNTIME_PREFIX_REGISTRY["SMHPRuntimeManager"]
        platform = Platform.SMHP
    else:
        raise ValueError("Invalid infrastructure manager")

    method_suffix = (
        METHOD_IMAGE_REGISTRY.get(platform, {}).get(version, {}).get(method, "")
    )
    return prefix + platform_infix + method_suffix
