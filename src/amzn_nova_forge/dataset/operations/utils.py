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
"""Shared utilities for dataset filter operations."""

from __future__ import annotations

import io
import os
import uuid

import boto3
import pyarrow as pa
import pyarrow.parquet as pq


def validate_paths_for_remote_execution(input_path: str, output_path: str) -> None:
    """Raise ValueError if local paths are used with a remote runtime manager.

    Remote runtimes (Glue, SMTJ) run on AWS and can only read/write S3.
    """
    local_paths = []
    if not input_path.startswith("s3://"):
        local_paths.append(f"input_path={input_path!r}")
    if not output_path.startswith("s3://"):
        local_paths.append(f"output_path={output_path!r}")
    if local_paths:
        raise ValueError(
            f"Remote runtime managers require S3 paths, got local path(s): "
            f"{', '.join(local_paths)}. "
            f"Provide S3 URIs instead of local paths."
        )


def validate_default_bucket_access(
    input_path: str,
    output_path: str,
    default_role_name: str,
) -> None:
    """Raise ValueError if S3 paths use non-default buckets with the default IAM role.

    The SDK's auto-created IAM roles are scoped to the default data-prep
    bucket (``sagemaker-forge-dataprep-<account>-<region>``). If the user's
    input or output points to a different bucket, the job will fail with
    AccessDenied at runtime. This check catches that early with a clear
    actionable message.
    """
    from amzn_nova_forge.util.s3_utils import DATAPREP_BUCKET_PREFIX

    default_prefix = DATAPREP_BUCKET_PREFIX + "-"
    for label, path in [("input_path", input_path), ("output_path", output_path)]:
        if path.startswith("s3://"):
            bucket_name = path[len("s3://") :].split("/", 1)[0]
            if not bucket_name.startswith(default_prefix):
                raise ValueError(
                    f"The {label} bucket '{bucket_name}' is not the SDK default "
                    f"data-prep bucket ('{DATAPREP_BUCKET_PREFIX}-<account>-<region>'). "
                    f"The auto-created IAM role '{default_role_name}' only has "
                    f"S3 access to the default bucket. Either:\n"
                    f"  1. Use the default bucket by omitting {label}, or\n"
                    f"  2. Provide a custom runtime_manager / execution_role_name "
                    f"with an IAM role that has access to '{bucket_name}'."
                )


def convert_to_s3_parquet(dataset_callable, s3_base_path: str, batch_size: int = 10000) -> str:
    """Convert a dataset generator to Parquet files on S3.

    Consumes the generator in batches, writes each batch as a separate
    Parquet file to a temp S3 directory derived from s3_base_path.
    Memory stays bounded by one batch at a time.

    Args:
        dataset_callable: A callable that returns an Iterator[Dict].
        s3_base_path: An S3 URI used to derive the temp directory.
        batch_size: Number of records per Parquet part file.

    Returns:
        The S3 directory URI containing the converted Parquet files.
    """

    if not s3_base_path.startswith("s3://"):
        raise ValueError(
            "Cannot convert data to S3 without an S3 base path. Provide output_path as an S3 URI."
        )

    base = s3_base_path.rstrip("/")
    parent = base.rsplit("/", 1)[0] if "/" in base else base
    run_id = uuid.uuid4().hex[:12]
    temp_dir = f"{parent}/_forge_converted_input/{run_id}/"

    temp_no_scheme = temp_dir[len("s3://") :]
    bucket, key_prefix = temp_no_scheme.split("/", 1)

    s3_client = boto3.client("s3")
    part_idx = 0
    batch: list[dict] = []

    for record in dataset_callable():
        batch.append(record)
        if len(batch) >= batch_size:
            _write_parquet_part(s3_client, bucket, key_prefix, part_idx, batch)
            part_idx += 1
            batch = []

    if batch:
        _write_parquet_part(s3_client, bucket, key_prefix, part_idx, batch)

    return temp_dir


def _write_parquet_part(
    s3_client, bucket: str, key_prefix: str, part_idx: int, records: list[dict]
) -> None:
    """Write a batch of records as a Parquet file to S3."""

    table = pa.Table.from_pylist(records)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    key = f"{key_prefix}part_{part_idx:05d}.parquet"
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())


def upload_local_file_to_s3(local_path: str, s3_base_path: str) -> str:
    """Upload a local file to a temp S3 location, preserving its format.

    Args:
        local_path: The resolved local file path.
        s3_base_path: An S3 URI used to derive the upload location.

    Returns:
        The S3 URI of the uploaded file.
    """

    if not s3_base_path.startswith("s3://"):
        raise ValueError(
            "Cannot upload local files without an S3 base path. Provide output_path as an S3 URI."
        )

    base = s3_base_path.rstrip("/")
    parent = base.rsplit("/", 1)[0] if "/" in base else base
    run_id = uuid.uuid4().hex[:12]
    filename = os.path.basename(local_path)
    s3_key_dir = f"{parent}/_forge_uploaded_input/{run_id}/"

    temp_no_scheme = s3_key_dir[len("s3://") :]
    bucket, key_prefix = temp_no_scheme.split("/", 1)

    s3_client = boto3.client("s3")
    key = f"{key_prefix}{filename}"
    with open(local_path, "rb") as f:
        s3_client.put_object(Bucket=bucket, Key=key, Body=f)

    return f"s3://{bucket}/{key}"
