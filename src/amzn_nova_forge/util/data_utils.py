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
Data utility functions for Nova customization.

This module provides utility functions for inspecting and processing datasets.
"""

import json

from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.recipe import load_file_content


def _has_multimodal_content(record: dict) -> bool:
    """Check if a single record contains multimodal content."""
    if "messages" not in record:
        return False

    for msg in record["messages"] or []:
        if not isinstance(msg, dict):
            continue
        # Use `or []` to handle both missing key and explicit null value
        for content_item in msg.get("content") or []:
            if isinstance(content_item, dict) and any(
                key in content_item for key in ["image", "video", "document"]
            ):
                return True
    return False


def _check_records(records) -> bool:
    """Return True if any record in the iterable contains multimodal content."""
    for record in records:
        if _has_multimodal_content(record):
            return True
    return False


def is_multimodal_data(data_s3_path: str) -> bool:
    """
    Check if dataset contains multimodal data by scanning records.

    Supports .jsonl (line-delimited JSON, streamed) and .json (full JSON array/object,
    loaded into memory). Returns True as soon as a multimodal record is found.

    Uses the same file loading approach as the dataset loader (load_file_content),
    including UTF-8 BOM handling via utf-8-sig encoding for JSONL files.

    Args:
        data_s3_path: S3 path to the dataset (.jsonl or .json)

    Returns:
        True if multimodal fields detected, False otherwise
    """
    try:
        if data_s3_path.endswith(".json"):
            # .json files are a single JSON object or array — read all lines then parse.
            lines = list(load_file_content(data_s3_path, extension=".json", encoding="utf-8"))
            content = "\n".join(lines)
            data = json.loads(content)
            records = data if isinstance(data, list) else [data]
            return _check_records(records)
        else:
            # .jsonl (default): stream line by line for memory efficiency.
            # utf-8-sig strips the UTF-8 BOM that some tools add, matching dataset loader behavior.
            for line in load_file_content(data_s3_path, extension=".jsonl", encoding="utf-8-sig"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if _has_multimodal_content(record):
                        return True
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    continue
            return False
    except Exception as e:
        logger.warning(
            f"Failed to check multimodal data from {data_s3_path}: {e}. Defaulting to text-only."
        )
        return False
