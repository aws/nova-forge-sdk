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
"""Arrow IPC/Feather dataset loader."""

from typing import Callable, Dict, Iterator

import pyarrow as pa
import pyarrow.fs as pafs

from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError


class ArrowDatasetLoader(DatasetLoader):
    """Load Arrow IPC/Feather files lazily using batch iteration."""

    _EXTENSIONS = {".arrow", ".feather", ".ipc"}
    _FORMAT = "arrow"

    @staticmethod
    def _iter_arrow_batches(source) -> Iterator:
        """Yield records from an Arrow IPC source, trying Stream then File format.

        Arrow IPC has two layouts: Stream (sequential, no random access) and
        File (has footer, supports random access). A .arrow/.ipc file could
        be either. We try Stream first (more memory-efficient), then fall
        back to File format.
        """
        try:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                yield from batch.to_pylist()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            # Not a stream — try IPC File format
            if hasattr(source, "seek"):
                source.seek(0)
            reader = pa.ipc.open_file(source)
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single Arrow IPC or Feather file."""

        def arrow_generator():
            try:
                if path.startswith("s3://"):
                    fs = pafs.S3FileSystem()
                    s3_path = path[len("s3://") :]
                    source = fs.open_input_file(s3_path)
                else:
                    source = path

                if path.endswith(".feather"):
                    table = pa.feather.read_table(source)
                    for batch in table.to_batches():
                        yield from batch.to_pylist()
                else:
                    yield from ArrowDatasetLoader._iter_arrow_batches(source)
            except Exception as e:
                raise DataPrepError(f"Error loading Arrow file {path}: {e}") from e

        return arrow_generator
