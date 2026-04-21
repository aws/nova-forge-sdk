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
"""Describes the current state of data flowing through the operation pipeline."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Iterator, Optional


class DataLocation(Enum):
    """Where the data physically resides."""

    S3 = "s3"
    LOCAL = "local"


@dataclass
class DataState:
    """Operation-agnostic description of a dataset's current state.

    Operations inspect this to decide whether they need to convert,
    upload, or otherwise prepare the data before execution.

    Attributes:
        path: Where the data lives (local path or S3 URI).
        format: Data format string (e.g. "jsonl", "parquet", "arrow").
        location: Where the data resides (S3 or local).
        generator: A callable that returns an Iterator[Dict] over the records.
    """

    path: str
    format: str
    location: DataLocation
    generator: Optional[Callable[[], Iterator[Dict]]] = None

    @staticmethod
    def from_loader(loader) -> DataState:
        """Build a DataState from a DatasetLoader's current state.

        Raises:
            ValueError: If the loader has no data source (load() not called).
        """
        path = loader._load_path
        if not path:
            raise ValueError(
                "No data source provided. "
                "Call load() before filter()/transform()/execute(). "
                "Example: loader.load('s3://bucket/data.jsonl').filter(...).execute()"
            )
        location = DataLocation.S3 if path.startswith("s3://") else DataLocation.LOCAL

        return DataState(
            path=path,
            format=loader._get_format(),
            location=location,
            generator=loader.dataset,
        )


class OutputPathResolver:
    """Generates output paths for chained operations within a single execute() session.

    Produces paths of the form::

        <parent>/<input_stem>/<session_id>/<method_name>/

    Where:
    - ``parent`` is the parent directory of the original load path.
    - ``input_stem`` identifies the source file/folder (e.g. ``train``
      from ``train.jsonl``).
    - ``session_id`` is a UTC timestamp (``YYYY-MM-DD_HH-MM-SS``)
      shared across all operations in one ``execute()`` call.
    - ``method_name`` is the filter/transform method enum value.

    All auto-generated outputs for a pipeline session are grouped under
    the same ``<parent>/<stem>/<session>/`` tree regardless of mid-chain
    overrides. This keeps outputs predictable and discoverable.
    """

    def __init__(self, load_path: str) -> None:
        base = load_path.rstrip("/")
        self._parent = base.rsplit("/", 1)[0] if "/" in base else base
        self._input_stem = self._extract_stem(load_path)
        self._session_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    def resolve(self, method) -> str:
        """Return the auto-generated output path for *method*."""
        method_name = method.value if hasattr(method, "value") else str(method)
        return f"{self._parent}/{self._input_stem}/{self._session_id}/{method_name}/"

    @staticmethod
    def _extract_stem(path: str) -> str:
        """Extract a human-readable stem from a path.

        - File: strip extension (``train.jsonl`` → ``train``).
        - Directory (trailing slash): last component (``s3://bucket/raw/`` → ``raw``).
        """
        cleaned = path.rstrip("/")
        basename = posixpath.basename(cleaned)
        stem, ext = posixpath.splitext(basename)
        return stem if ext else basename
