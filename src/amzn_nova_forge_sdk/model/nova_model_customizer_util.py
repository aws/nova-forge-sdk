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

from amzn_nova_forge_sdk.model.result import TrainingResult
from amzn_nova_forge_sdk.recipe.recipe_config import BYOD_AVAILABLE_EVAL_TASKS
from amzn_nova_forge_sdk.util.checkpoint_util import (
    extract_checkpoint_path_from_job_output,
)
from amzn_nova_forge_sdk.util.logging import logger


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


# ==================== Job Caching Utilities ====================

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from amzn_nova_forge_sdk.model.result import BaseJobResult
from amzn_nova_forge_sdk.model.result.job_result import JobStatus


def get_recipe_directory(generated_recipe_dir: Optional[str]) -> Optional[str]:
    """
    Get the directory path for recipe storage, handling both directory and file paths.
    """
    if not generated_recipe_dir:
        return None
    if generated_recipe_dir.endswith((".yaml", ".yml")):
        return os.path.dirname(generated_recipe_dir) or "."
    return generated_recipe_dir


def generate_job_hash(customizer, job_name: str, job_type: str, **job_params) -> str:
    """
    Generate segmented hash where each parameter gets its own labeled hash segment.
    This allows flexible job cache matching by matching only relevant segments.
    """
    segments = {}

    segments["model"] = hashlib.sha256(
        str(customizer.model.value).encode()
    ).hexdigest()[:8]
    segments["method"] = hashlib.sha256(
        str(customizer.method.value).encode()
    ).hexdigest()[:8]
    segments["data_s3_path"] = hashlib.sha256(
        (customizer.data_s3_path or "").encode()
    ).hexdigest()[:8]
    segments["job_type"] = hashlib.sha256(job_type.encode()).hexdigest()[:8]
    segments["model_path"] = hashlib.sha256(
        str(customizer.model_path).encode()
    ).hexdigest()[:8]

    if "recipe_path" in job_params:
        segments["recipe_path"] = hashlib.sha256(
            str(job_params["recipe_path"]).encode()
        ).hexdigest()[:8]

    overrides = job_params.get("overrides", {})
    for param, value in overrides.items():
        segments[f"override_{param}"] = hashlib.sha256(str(value).encode()).hexdigest()[
            :8
        ]

    if hasattr(customizer.infra, "instance_type"):
        segments["instance_type"] = hashlib.sha256(
            str(customizer.infra.instance_type).encode()
        ).hexdigest()[:8]
    if hasattr(customizer.infra, "instance_count"):
        segments["instance_count"] = hashlib.sha256(
            str(customizer.infra.instance_count).encode()
        ).hexdigest()[:8]

    for key, value in job_params.items():
        if key not in ["recipe_path", "overrides"]:
            segments[key] = hashlib.sha256(str(value).encode()).hexdigest()[:8]

    segment_pairs = [f"{k}:{v}" for k, v in sorted(segments.items())]
    return ",".join(segment_pairs)


def should_persist_results(customizer) -> bool:
    """
    Check if results should be persisted based on configuration.
    """
    if not customizer.enable_job_caching:
        return False
    if not customizer.job_cache_dir:
        logger.warning("Job caching enabled but job_cache_dir is not set")
        return False
    cache_path = Path(customizer.job_cache_dir)
    if not cache_path.exists():
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"Failed to create job cache directory '{customizer.job_cache_dir}': {e}"
            )
            return False
    return True


def matches_job_cache_criteria(
    job_caching_config: dict, stored_hash: str, current_hash: str
) -> bool:
    """
    Check if stored segmented hash matches current hash based on job caching config.
    """

    def parse_segments(hash_str: str) -> Dict[str, str]:
        segments = {}
        for pair in hash_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                segments[key] = value
        return segments

    stored_segments = parse_segments(stored_hash)
    current_segments = parse_segments(current_hash)

    config = job_caching_config

    exclude_params = config.get("exclude_params", [])
    if isinstance(exclude_params, list):
        for param in exclude_params:
            stored_segments.pop(param, None)
            current_segments.pop(param, None)

    include_params = config.get("include_params", [])
    if isinstance(include_params, list):
        for param in include_params:
            if stored_segments.get(param) != current_segments.get(param):
                return False

    exclude_params = config.get("exclude_params", [])
    if isinstance(exclude_params, list) and "*" in exclude_params:
        return True

    if config.get("include_core", True):
        core_fields = ["model", "method", "data_s3_path", "job_type", "model_path"]
        for field in core_fields:
            if stored_segments.get(field) != current_segments.get(field):
                return False

    if config.get("include_recipe", True):
        if stored_segments.get("recipe_path") != current_segments.get("recipe_path"):
            return False
        all_override_keys: set[str] = set()
        for segments in [stored_segments, current_segments]:
            all_override_keys.update(
                k for k in segments.keys() if k.startswith("override_")
            )
        for override_key in all_override_keys:
            if stored_segments.get(override_key) != current_segments.get(override_key):
                return False

    if config.get("include_infra", False):
        infra_fields = ["instance_type", "instance_count"]
        for field in infra_fields:
            if stored_segments.get(field) != current_segments.get(field):
                return False

    return True


def get_result_file_path(
    customizer, job_name: str, job_type: str, **job_params
) -> Path:
    """
    Get path for persisted result file (job caching only).
    """
    if not should_persist_results(customizer):
        raise ValueError("Cannot get result file path when persistence is disabled")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
    filename = f"{job_name}_{job_type}_{timestamp}.json"
    return Path(customizer.job_cache_dir) / filename


def load_existing_result(
    customizer, job_name: str, job_type: str, **job_params
) -> Optional[BaseJobResult]:
    """
    Load existing result if available and matches job cache criteria.
    """
    if not should_persist_results(customizer):
        return None

    try:
        allowed_statuses = customizer._job_caching_config.get("allowed_statuses", None)
        assert isinstance(allowed_statuses, list)
    except (AssertionError, TypeError):
        logger.error(
            f"Invalid allowed_statuses configuration: expected list, got {type(allowed_statuses).__name__} with value {allowed_statuses}. Skipping job cache lookup."
        )
        return None

    try:
        current_hash = generate_job_hash(customizer, job_name, job_type, **job_params)
        results_dir = Path(customizer.job_cache_dir)

        if not results_dir.exists():
            return None

        pattern = f"{job_name}_{job_type}_*.json"
        for result_file in results_dir.glob(pattern):
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)

                stored_hash = data.get("_job_cache_hash")
                if stored_hash and matches_job_cache_criteria(
                    customizer._job_caching_config, stored_hash, current_hash
                ):
                    result = BaseJobResult.load(str(result_file))
                    if hasattr(result, "_job_cache_hash"):
                        result._job_cache_hash = stored_hash

                    job_status, raw_status = result.get_job_status()
                    if job_status in allowed_statuses:
                        logger.info(
                            f"Reusing existing {job_type} result for {job_name} with status {job_status} from {result_file.absolute()}"
                        )
                        return result
                    else:
                        logger.info(
                            f"Found matching {job_type} result for {job_name} but job status {job_status} not in allowed statuses {[s.value for s in allowed_statuses]}"
                        )
            except Exception as e:
                logger.debug(f"Skipping corrupted result file {result_file}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Failed to search for existing results: {e}")

    return None


def collect_all_parameters(
    customizer, job_name: str, job_type: str, **job_params
) -> dict:
    """
    Collect all relevant parameters from customizer, infra manager, and job params.
    """
    all_params = {}

    if hasattr(customizer.infra, "__dict__"):
        infra_params = {
            f"infra_{k}": v
            for k, v in customizer.infra.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        all_params.update(infra_params)

    customizer_params = {
        "model": customizer.model.value
        if hasattr(customizer.model, "value")
        else str(customizer.model),
        "method": customizer.method.value
        if hasattr(customizer.method, "value")
        else str(customizer.method),
        "data_s3_path": customizer.data_s3_path,
        "output_s3_path": customizer.output_s3_path,
        "model_path": customizer.model_path,
        "deployment_mode": customizer.deployment_mode.value
        if hasattr(customizer.deployment_mode, "value")
        else str(customizer.deployment_mode),
    }
    all_params.update(customizer_params)
    all_params.update(job_params)

    return all_params


def persist_result(
    customizer, result: BaseJobResult, job_name: str, job_type: str, **job_params
) -> None:
    """
    Persist job result to file if persistence is enabled.
    """
    if not should_persist_results(customizer):
        return

    try:
        result_file = get_result_file_path(customizer, job_name, job_type, **job_params)
        result_file.parent.mkdir(parents=True, exist_ok=True)

        data = result._to_dict()
        data["__class_name__"] = result.__class__.__name__

        if customizer.enable_job_caching:
            all_params = collect_all_parameters(
                customizer, job_name, job_type, **job_params
            )
            segmented_hash = generate_job_hash(
                customizer, job_name, job_type, **all_params
            )
            data["_job_cache_hash"] = segmented_hash
            data["_all_parameters"] = all_params

        with open(result_file, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Job result saved to {result_file}")
    except Exception as e:
        logger.warning(f"Failed to persist {job_type} result for {job_name}: {e}")
