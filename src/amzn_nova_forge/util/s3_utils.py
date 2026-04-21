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
"""Centralised S3 bucket utilities for the Nova Forge SDK.

The SDK uses a default data-preparation bucket:

- ``sagemaker-forge-dataprep-{account_id}-{region}`` — intermediate
  data-preparation artifacts (Glue scripts/wheels, Bedrock batch
  staging files, etc.).

All operations that auto-create a bucket should use the helpers here
so the naming stays consistent.
"""

from typing import Optional

import boto3
from botocore.exceptions import ClientError

from amzn_nova_forge.util.logging import logger

DATAPREP_BUCKET_PREFIX = "sagemaker-forge-dataprep"

# Key prefixes inside the data-prep bucket. Each runtime keeps its
# artifacts (entry scripts, bundled wheels, etc.) under its own prefix so
# Glue, SMTJ, and any future runtimes do not collide.
GLUE_ARTIFACT_PREFIX = "nova-forge/glue-artifacts"
SMTJ_DATAPREP_ARTIFACT_PREFIX = "nova-forge/smtj-dataprep-artifacts"


def get_dataprep_bucket_name(
    account_id: Optional[str] = None,
    region: Optional[str] = None,
) -> str:
    """Return the data-prep bucket: ``sagemaker-forge-dataprep-{account_id}-{region}``."""
    if region is None:
        region = boto3.session.Session().region_name or "us-east-1"
    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"{DATAPREP_BUCKET_PREFIX}-{account_id}-{region}"


def ensure_bucket_exists(
    bucket: str,
    region: Optional[str] = None,
    kms_key_arn: Optional[str] = None,
) -> None:
    """Create *bucket* if it does not already exist.

    Raises:
        PermissionError: If the bucket exists but is not owned by the caller,
            or if the caller lacks permission to create the bucket.
        Exception: If bucket creation fails for any other reason.
    """
    if region is None:
        region = boto3.session.Session().region_name or "us-east-1"

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            _create_bucket(s3, bucket, region, kms_key_arn)
        elif code in ("403", "Forbidden", "AccessDenied"):
            raise PermissionError(
                f"Bucket '{bucket}' exists but is not owned by the current account. "
                "Provide a different bucket name or ensure you have access."
            ) from exc
        else:
            raise


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _create_bucket(
    s3_client,
    bucket: str,
    region: str,
    kms_key_arn: Optional[str] = None,
) -> None:
    """Create an S3 bucket with optional KMS encryption."""
    try:
        create_params: dict = {"Bucket": bucket}
        if region != "us-east-1":
            create_params["CreateBucketConfiguration"] = {
                "LocationConstraint": region,
            }
        s3_client.create_bucket(**create_params)

        if kms_key_arn:
            s3_client.put_bucket_encryption(
                Bucket=bucket,
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
                "Auto-created S3 bucket '%s' in your account with KMS encryption (%s).",
                bucket,
                kms_key_arn,
            )
        else:
            logger.info(
                "Auto-created S3 bucket '%s' in your account for data preparation artifacts.",
                bucket,
            )
    except ClientError as exc:
        error_code = exc.response["Error"].get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            raise PermissionError(
                f"Permission denied when creating S3 bucket '{bucket}'. "
                "Your IAM identity needs the s3:CreateBucket permission, or you can "
                "create the bucket manually and pass it via the s3_artifact_bucket parameter."
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to create bucket '{bucket}': {exc}") from exc
