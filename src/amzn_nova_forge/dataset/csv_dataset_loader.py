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
"""CSV dataset loader."""

import csv
from typing import Callable, Dict, Iterator

from ..util.recipe import load_file_content
from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError


class CSVDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".csv"}
    _FORMAT = "csv"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single CSV file."""

        def csv_generator():
            """Generator that yields records from CSV file row by row."""
            try:
                lines = load_file_content(file_path=path, extension=".csv", encoding="utf-8-sig")
                reader = csv.DictReader(lines)
                yield from reader
            except UnicodeError:
                lines = load_file_content(file_path=path, extension=".csv", encoding="utf-8")
                reader = csv.DictReader(lines)
                yield from reader
            except Exception as e:
                raise DataPrepError(f"Error loading CSV file {path}: {e}") from e

        return csv_generator
