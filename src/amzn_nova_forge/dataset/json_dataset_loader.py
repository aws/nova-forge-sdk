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
"""JSON dataset loader."""

import json
from typing import Callable, Dict, Iterator

from ..util.recipe import load_file_content
from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError


class JSONDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".json"}
    _FORMAT = "json"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single JSON file."""

        def json_generator():
            """Generator that yields records from JSON file."""
            try:
                lines = list(load_file_content(file_path=path, extension=".json", encoding="utf-8"))
                content = "\n".join(lines)
                data = json.loads(content)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data
            except Exception as e:
                raise DataPrepError(f"Error loading JSON file {path}: {e}") from e

        return json_generator
