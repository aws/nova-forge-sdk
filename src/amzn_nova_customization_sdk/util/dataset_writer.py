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
Utility module for streaming dataset writes to local files and S3.

This module provides efficient streaming write operations that avoid
loading entire datasets into memory.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator

import boto3

from .logging import logger


class DatasetWriteError(Exception):
    """Custom exception for dataset write errors."""

    pass


class DatasetWriter:
    """
    Handles streaming writes of dataset iterators to local files or S3.

    This class provides memory-efficient methods for writing large datasets
    without materializing them entirely in memory.
    """

    @staticmethod
    def save_to_local(
        save_path: str, dataset_iter: Iterator[Dict], is_jsonl: bool
    ) -> None:
        """
        Stream dataset to local file without loading all data into memory.

        Args:
            save_path: Local file path
            dataset_iter: Iterator of dataset records
            is_jsonl: True for JSONL format, False for JSON single-object format

        Raises:
            DatasetWriteError: If write operation fails or JSON format has multiple records
        """
        try:
            local_path = Path(save_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with local_path.open("w", encoding="utf-8") as f:
                count = 0
                for item in dataset_iter:
                    count += 1

                    # JSON format: check for multiple records before writing
                    if not is_jsonl and count > 1:
                        raise DatasetWriteError(
                            f"JSON format expects exactly one dictionary, but found multiple records. "
                            f"Use JSONL format (.jsonl extension) for datasets with multiple records."
                        )

                    # Write the record
                    indent = None if is_jsonl else 2
                    f.write(json.dumps(item, indent=indent, ensure_ascii=False))
                    if is_jsonl:
                        f.write("\n")

                # Handle empty dataset
                if count == 0:
                    if is_jsonl:
                        # JSONL: empty file is fine, nothing to write
                        pass
                    else:
                        # JSON: write empty object
                        f.write("{}")

        except DatasetWriteError:
            raise
        except Exception as e:
            raise DatasetWriteError(f"Failed to write to local file {save_path}: {e}")

    @staticmethod
    def save_to_s3(
        save_path: str, dataset_iter: Iterator[Dict], is_jsonl: bool
    ) -> None:
        """
        Stream dataset to S3 without loading all data into memory.
        Uses a temporary file and boto3's upload_file for efficient multipart upload.

        Args:
            save_path: S3 path (s3://bucket/key)
            dataset_iter: Iterator of dataset records
            is_jsonl: True for JSONL format, False for JSON single-object format

        Raises:
            DatasetWriteError: If S3 upload fails or JSON format has multiple records
        """
        # Validate S3 path format
        if not save_path.startswith("s3://"):
            raise DatasetWriteError(
                f"Invalid S3 path '{save_path}'. Expected format: s3://bucket/key"
            )

        s3_path = save_path[5:]  # Remove 's3://'

        # Validate S3 path contains both bucket and key
        parts = s3_path.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise DatasetWriteError(
                f"Invalid S3 path '{save_path}'. Expected format: s3://bucket/key"
            )

        bucket, key = parts

        tmp_path = None
        try:
            # Create a temporary file and write data to it
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".json"
            ) as tmp_file:
                tmp_path = tmp_file.name

                # Use save_to_local to write to the temp file
                DatasetWriter.save_to_local(tmp_path, dataset_iter, is_jsonl)

            # Upload the temporary file to S3
            s3_client = boto3.client("s3")
            s3_client.upload_file(
                tmp_path, bucket, key, ExtraArgs={"ContentType": "application/json"}
            )

        except DatasetWriteError:
            raise
        except Exception as e:
            raise DatasetWriteError(f"Failed to upload to S3: {e}")
        finally:
            # Clean up temporary file
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    logger.warning(
                        f"Failed to delete tmp file {tmp_path} after S3 upload."
                    )
                    pass  # Best effort cleanup
