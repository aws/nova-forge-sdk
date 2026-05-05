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
"""JSONL dataset loader."""

import json
from typing import Callable, Dict, Iterator

from ..util.logging import logger
from ..util.recipe import load_file_content
from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError


class JSONLDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".jsonl"}
    _FORMAT = "jsonl"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single JSONL file."""

        def jsonl_generator():
            """Generator that yields records from JSONL file line by line."""
            try:
                for line in load_file_content(
                    file_path=path, extension=".jsonl", encoding="utf-8-sig"
                ):
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            preview = line[:120] + ("..." if len(line) > 120 else "")
                            logger.warning("Skipping malformed JSON line in %s: %s", path, preview)
            except Exception as e:
                raise DataPrepError(f"Error loading JSONL file {path}: {e}") from e

        return jsonl_generator
