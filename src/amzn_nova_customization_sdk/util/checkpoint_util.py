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
Utility functions related to checkpoint objects.
"""

import json
import tarfile
import tempfile
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.model.model_enums import Platform
from amzn_nova_customization_sdk.model.result import TrainingResult
from amzn_nova_customization_sdk.util.logging import logger


def extract_checkpoint_path_from_job_output(
    output_s3_path: str,
    job_id: Optional[str] = None,
    job_result: Optional[TrainingResult] = None,
) -> str:
    """
    Extracts the model checkpoint path from a training job's output.

    This method:
    1. Checks if the output_s3_path exists
    2. Checks if the job's output directory exists
    3. Retrieves the tar.gz output from the job's output directory
    4. Extracts the manifest from the tar.gz
    5. Extracts the checkpoint path from the manifest JSON

    Args:
        output_s3_path: The S3 output path where job artifacts are stored
        job_id: Optional training job ID. Retrieved from job_result if absent.
        job_result: Optional TrainingResult object to check job status.
            Also used to retrieve job_id if it's not provided

    Returns:
        str: The S3 path to the model checkpoint

    Raises:
        Exception: If unable to extract checkpoint path or job is not completed
    """
    from amzn_nova_customization_sdk.model.result.job_result import JobStatus

    s3_client = boto3.client("s3")

    # If job_result is provided, check if job is completed
    if job_result is not None:
        status, _ = job_result.get_job_status()
        if status != JobStatus.COMPLETED:
            raise Exception(
                f"Training job {job_id} is not completed. Current status: {status.value}"
                "Please try again once the job has completed."
            )

        # If job_id is not provided but job_result is, acquire the job id from the result object
        if job_id is None:
            job_id = job_result.job_id
    # Throw a ValueError if we weren't given job_id or job_result
    elif job_id is None:
        raise ValueError(
            "Either job_id or job_result must be non-None in extract_checkpoint_path_from_job_output."
        )

    # Parse S3 path
    parsed_url = urlparse(output_s3_path)
    bucket = parsed_url.netloc
    base_key = parsed_url.path.lstrip("/")

    # Construct the path to the output tar.gz file
    output_key = f"{base_key}/{job_id}/output/output.tar.gz"

    # Save the full s3 URL to use in error messages
    full_output_s3_url = f"s3://{bucket}/{output_key}"

    # Section 1: Detect platform if not already known from job_result
    platform = None
    if job_result is not None and hasattr(job_result, "model_artifacts"):
        if job_result.model_artifacts.checkpoint_s3_path:
            from amzn_nova_customization_sdk.util.platform_util import (
                detect_platform_from_path,
            )

            platform = detect_platform_from_path(
                job_result.model_artifacts.checkpoint_s3_path
            )

    # Section 2: Extract manifest based on platform (try SMHP first if unknown)
    manifest = None
    try:
        if platform == Platform.SMHP or platform is None:
            # Try SMHP format: manifest.json directly in S3
            try:
                manifest_key = f"{base_key}/{job_id}/manifest.json"
                s3_client.head_object(Bucket=bucket, Key=manifest_key)
                manifest_obj = s3_client.get_object(Bucket=bucket, Key=manifest_key)
                manifest_content = manifest_obj["Body"].read()
                manifest = json.loads(manifest_content)
                platform = Platform.SMHP
                logger.info("Extracted manifest for SMHP training job")
            except (ClientError, json.JSONDecodeError) as e:
                if platform == Platform.SMHP:
                    # Platform was explicitly detected as SMHP but manifest not found
                    raise KeyError(
                        f"manifest.json not found at s3://{bucket}/{manifest_key}"
                    ) from e
                # Platform unknown, try SMTJ format
                logger.debug(f"SMHP format not found, trying SMTJ: {e}")

        if manifest is None:
            # Try SMTJ format: manifest.json inside output.tar.gz
            s3_client.head_object(Bucket=bucket, Key=output_key)
            with tempfile.NamedTemporaryFile() as tmp_file:
                s3_client.download_file(bucket, output_key, tmp_file.name)

                with tarfile.open(tmp_file.name, "r:gz") as tar:
                    manifest_file = tar.extractfile("manifest.json")
                    if manifest_file is None:
                        raise KeyError(
                            f"manifest.json not found in {full_output_s3_url}"
                        )
                    manifest_content = manifest_file.read()
                    manifest = json.loads(manifest_content)
                    platform = Platform.SMTJ
                    logger.info("Extracted manifest for SMTJ training job")

    except (KeyError, json.JSONDecodeError, tarfile.TarError, ClientError) as e:
        error_args = [f"Failed to extract manifest from {full_output_s3_url}: {e}"]
        if (
            isinstance(e, ClientError)
            and e.response.get("Error", {}).get("Code") == "404"
        ):
            error_args.append("Job may not be completed or output path is incorrect.")
        raise Exception(*error_args) from e

    # Section 3: Extract checkpoint path from manifest (shared logic)
    try:
        if "checkpoint_s3_bucket" not in manifest:
            raise KeyError(
                f"checkpoint_s3_bucket not found in manifest.json from {full_output_s3_url}"
            )

        checkpoint_path = manifest["checkpoint_s3_bucket"]

        if not checkpoint_path or not checkpoint_path.strip():
            raise ValueError(
                f"checkpoint_s3_bucket is empty in manifest.json from {full_output_s3_url}"
            )

        logger.info(
            f"Successfully extracted checkpoint path from {platform.value if platform else platform} job: {checkpoint_path}"
        )
        return checkpoint_path

    except (KeyError, ValueError) as e:
        raise Exception(
            f"Failed to extract checkpoint path from manifest in {full_output_s3_url}: {e}"
        ) from e


def validate_checkpoint_uri(checkpoint_uri: str, region: str):
    """
    Validates a user's S3 checkpoint URI

    Args:
        checkpoint_uri: S3 checkpoint URI
        region: AWS region

    Raises:
        ValueError: If the bucket or key don't exist, or if the S3 URI is invalid format
    """
    try:
        s3_client = boto3.client("s3", region_name=region)
        # Parse S3 URI
        if not checkpoint_uri.startswith("s3://"):
            raise ValueError(f"Model path must be an S3 URI, got: {checkpoint_uri}")

        # Remove s3:// prefix and split bucket/key
        s3_path = checkpoint_uri[5:]  # Remove "s3://"
        bucket, key = s3_path.split("/", 1)

        # Check if object exists
        s3_client.head_object(Bucket=bucket, Key=key)

    except Exception as e:
        if "NoSuchBucket" in str(e):
            raise ValueError(
                f"S3 bucket {bucket} does not exist when validating model checkpoint {checkpoint_uri}: {str(e)}"
            )
        elif "NoSuchKey" in str(e):
            raise ValueError(
                f"Model checkpoint does not exist at {checkpoint_uri}: {str(e)}"
            )
        elif "Model path must be an S3 URI" in str(e):
            raise ValueError(f"Model path must be an S3 URI, got: {checkpoint_uri}")
        else:
            logger.debug("Unable to validate checkpoint URI.")
