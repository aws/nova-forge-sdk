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
"""Centralised S3 utilities for the Nova Forge SDK.

This module provides:

1. **Bucket management** — naming, creation, and existence checks for the
   SDK's data-preparation bucket.
2. **File operations** — download with caching, URI parsing, line reading,
   and prefix listing for working with S3-hosted files.

The SDK uses a default data-preparation bucket:

- ``sagemaker-forge-dataprep-{account_id}-{region}`` — intermediate
  data-preparation artifacts (Glue scripts/wheels, Bedrock batch
  staging files, etc.).

All operations that auto-create a bucket should use the helpers here
so the naming stays consistent.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from collections.abc import Iterator
from typing import Any, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from tqdm import tqdm

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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into (bucket, key).

    Raises:
        ValueError: If the URI is not a valid ``s3://bucket/key`` format.
    """
    if not uri.startswith("s3://") or len(uri) <= 5:
        msg = f"Invalid S3 URI: {uri}"
        raise ValueError(msg)
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    if not bucket:
        msg = f"Invalid S3 URI (empty bucket name): {uri}"
        raise ValueError(msg)
    return bucket, parts[1] if len(parts) > 1 else ""


def is_s3(path: str) -> bool:
    """Return True if path is an S3 URI."""
    return path.startswith("s3://")


def s3_cache_path(s3_uri: str, cache_dir: str) -> str:
    """Compute the local cache path for an S3 URI."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(s3_uri.encode()).hexdigest()[:12]
    base = os.path.basename(s3_uri.rstrip("/"))
    return os.path.join(cache_dir, f"{base}.{h}")


def cached_s3_file(s3_uri: str, cache_dir: str) -> str | None:
    """Return local cache path if the file is already cached, else None."""
    path = s3_cache_path(s3_uri, cache_dir)
    if os.path.exists(path):
        return path
    return None


def download_s3_file(s3_client: Any, bucket: str, key: str, local_path: str) -> None:
    """Download a single S3 object to a local path (atomic via rename).

    Uses multipart transfer and tqdm progress if available. Boto3 exceptions
    propagate to the caller.
    """

    tmp_path = local_path + ".tmp"
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024,
    )
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        total_bytes = head.get("ContentLength", 0)
        logger.info("Downloading %s/%s ...", bucket, key)
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Downloading",
            ncols=80,
        ) as pbar:
            s3_client.download_file(
                bucket,
                key,
                tmp_path,
                Config=config,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
        os.replace(tmp_path, local_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def read_lines(
    path: str, s3_client: Any | None = None, cache_dir: str | None = None
) -> Iterator[str]:
    """Read lines from a local or S3 file, yielding one line at a time.

    Returns an iterator that streams lines without loading the entire file
    into memory at once.

    For S3 files, downloads to cache first if *cache_dir* is provided.
    When *cache_dir* is ``None``, the S3 object is streamed directly into
    memory without caching. Only use for small files (e.g., log files).
    For large data files, use :func:`ensure_local` and stream.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If an S3 URI is malformed.
        RuntimeError: If an S3 client is required but not provided.
    """
    if is_s3(path):
        if cache_dir is not None:
            cached = cached_s3_file(path, cache_dir)
            if cached:
                with open(cached) as f:
                    yield from f
                return
        if s3_client is None:
            msg = f"S3 client required to read {path}"
            raise RuntimeError(msg)
        bucket, key = parse_s3_uri(path)
        if cache_dir is not None:
            local_path = s3_cache_path(path, cache_dir)
            download_s3_file(s3_client, bucket, key, local_path)
            with open(local_path) as f:
                yield from f
            return
        # No cache — stream directly into memory
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        yield from io.StringIO(content)
    else:
        if not os.path.exists(path):
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        with open(path) as f:
            yield from f


def ensure_local(path: str, s3_client: Any | None = None, cache_dir: str | None = None) -> str:
    """Ensure a file is available locally (downloading from S3 if needed).

    When *cache_dir* is provided, files are cached for reuse across calls.
    When *cache_dir* is ``None``, the file is downloaded to a temporary
    location (caller is responsible for cleanup if needed).

    Returns the local file path.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If an S3 URI is malformed.
        RuntimeError: If an S3 client is required but not provided.
    """
    if not is_s3(path):
        if not os.path.exists(path):
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        return path

    if cache_dir is not None:
        cached = cached_s3_file(path, cache_dir)
        if cached:
            return cached

    if s3_client is None:
        msg = f"S3 client required to download {path}"
        raise RuntimeError(msg)
    bucket, key = parse_s3_uri(path)
    if cache_dir is not None:
        local_path = s3_cache_path(path, cache_dir)
    else:
        suffix = os.path.basename(path.rstrip("/"))
        fd, local_path = tempfile.mkstemp(suffix=f"_{suffix}")
        os.close(fd)
    try:
        download_s3_file(s3_client, bucket, key, local_path)
    except BaseException:
        if cache_dir is None:
            with contextlib.suppress(OSError):
                os.remove(local_path)
        raise
    return local_path


def list_s3_prefix(uri: str, s3_client: Any, suffix: str = "") -> list[str]:
    """List S3 objects under a prefix, optionally filtered by suffix.

    Returns a list of full ``s3://`` URIs.

    Raises:
        ValueError: If the URI is malformed.
    """
    bucket, prefix = parse_s3_uri(uri)
    if not prefix.endswith("/"):
        prefix += "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    uris: list[str] = [
        f"s3://{bucket}/{obj['Key']}"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if not suffix or obj["Key"].endswith(suffix)
    ]
    return uris
