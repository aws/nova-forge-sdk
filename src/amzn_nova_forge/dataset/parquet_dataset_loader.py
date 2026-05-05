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
"""Parquet dataset loader."""

from typing import Callable, Dict, Iterator

import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError


class ParquetDatasetLoader(DatasetLoader):
    """Load Parquet files lazily using PyArrow batch iteration."""

    _EXTENSIONS = {".parquet", ".pq"}
    _FORMAT = "parquet"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single Parquet file."""

        def parquet_generator():
            try:
                if path.startswith("s3://"):
                    fs = pafs.S3FileSystem()
                    s3_path = path[len("s3://") :]
                    pf = pq.ParquetFile(fs.open_input_file(s3_path))
                else:
                    pf = pq.ParquetFile(path)

                for batch in pf.iter_batches():
                    yield from batch.to_pylist()
            except Exception as e:
                raise DataPrepError(f"Error loading Parquet file {path}: {e}") from e

        return parquet_generator
