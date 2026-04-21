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
"""File path resolution and directory scanning utilities for dataset loaders."""

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..util.logging import logger
from .operations.base import DataPrepError


def resolve_path(path: str) -> str:
    """Normalize a file path to an absolute path.

    S3 URIs pass through unchanged. Local paths get tilde expansion
    and are resolved relative to the current working directory.
    """
    if path.startswith("s3://"):
        return path
    return str(Path(path).expanduser().resolve())


def is_directory(path: str) -> bool:
    """Return True if the path refers to a directory (local or S3)."""
    if path.startswith("s3://"):
        return path.endswith("/")
    return os.path.isdir(path)


def check_path_exists(path: str) -> None:
    """Verify that a single file path exists. Raises DataPrepError for local
    files that don't exist. For S3, performs a best-effort HeadObject check —
    logs a warning on permission errors rather than raising.

    Args:
        path: Resolved absolute local path or S3 URI.

    Raises:
        DataPrepError: If a local file does not exist.
    """
    if path.startswith("s3://"):
        s3_path = path[len("s3://") :]
        bucket, _, key = s3_path.partition("/")
        try:
            client = boto3.client("s3")
            client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                raise DataPrepError(f"File not found: {path}") from e
            if error_code in ("403", "AccessDenied"):
                # User may not have read permissions — don't block
                logger.warning(
                    f"Unable to confirm S3 object exists (access denied): {path}. "
                    f"Proceeding anyway."
                )
                return
            # Other errors — log warning, don't block
            logger.warning(
                f"Unable to confirm S3 object exists: {path}. Proceeding anyway. Error: {e}"
            )
        except Exception as e:
            logger.warning(
                f"Unable to confirm S3 object exists: {path}. Proceeding anyway. Error: {e}"
            )
    else:
        if not os.path.isfile(path):
            raise DataPrepError(f"File not found: {path}")


def check_extension(path: str, expected_extensions: set[str]) -> None:
    """Verify that a file path has one of the expected extensions.

    Args:
        path: File path or S3 URI.
        expected_extensions: Set of allowed extensions (e.g. {".jsonl"}).

    Raises:
        DataPrepError: If the file extension doesn't match any expected extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in expected_extensions:
        raise DataPrepError(
            f"File '{path}' has extension '{ext}', expected one of {sorted(expected_extensions)}."
        )


def scan_directory(dir_path: str, expected_extensions: set[str]) -> list[str]:
    """Dispatch to local or S3 directory scanner."""
    if dir_path.startswith("s3://"):
        return scan_s3_directory(dir_path, expected_extensions)
    return scan_local_directory(dir_path, expected_extensions)


def scan_local_directory(dir_path: str, expected_extensions: set[str]) -> list[str]:
    """Scan a local directory for files matching expected extensions.

    Lists immediate children (non-recursive), skipping subdirectories
    and hidden/metadata files (names starting with '.' or '_').
    Raises DataPrepError if any non-matching data files are found or if
    no matching files exist. Returns sorted file paths.
    """
    matching = []
    unexpected: list[str] = []

    try:
        entries = os.listdir(dir_path)
    except OSError as e:
        raise DataPrepError(f"Cannot read directory '{dir_path}': {e}") from e

    for name in entries:
        if name.startswith(".") or name.startswith("_"):
            continue
        full_path = os.path.join(dir_path, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in expected_extensions:
            matching.append(full_path)
        else:
            unexpected.append(name)

    if unexpected:
        raise DataPrepError(
            f"Directory '{dir_path}' contains unexpected files: "
            f"{sorted(unexpected)}. Expected only files with extensions "
            f"{sorted(expected_extensions)}."
        )

    if not matching:
        raise DataPrepError(
            f"No files with extensions {sorted(expected_extensions)} found in '{dir_path}'."
        )

    matching.sort()
    return matching


def scan_s3_directory(dir_path: str, expected_extensions: set[str]) -> list[str]:
    """Scan an S3 prefix for files matching expected extensions.

    Uses boto3 list_objects_v2 with Delimiter='/' for non-recursive listing.
    Skips hidden/metadata files (names starting with '.' or '_').
    Raises DataPrepError if any non-matching files are found or if no
    matching files exist. Returns sorted S3 URIs.
    """
    prefix = dir_path[len("s3://") :]
    bucket, _, key_prefix = prefix.partition("/")
    if not key_prefix.endswith("/"):
        key_prefix += "/"

    client = boto3.client("s3")
    matching = []
    unexpected: list[str] = []

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            if not name or name.startswith(".") or name.startswith("_"):
                continue
            ext = os.path.splitext(name)[1].lower()
            s3_uri = f"s3://{bucket}/{key}"
            if ext in expected_extensions:
                matching.append(s3_uri)
            else:
                unexpected.append(name)

    if unexpected:
        raise DataPrepError(
            f"S3 directory '{dir_path}' contains unexpected files: "
            f"{sorted(unexpected)}. Expected only files with extensions "
            f"{sorted(expected_extensions)}."
        )

    if not matching:
        raise DataPrepError(
            f"No files with extensions {sorted(expected_extensions)} found in '{dir_path}'."
        )

    matching.sort()
    return matching
