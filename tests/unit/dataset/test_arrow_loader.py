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
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.feather
import pyarrow.ipc

from amzn_nova_forge.dataset.dataset_loader import ArrowDatasetLoader
from amzn_nova_forge.dataset.operations.base import DataPrepError


class TestArrowDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        self.table = pa.table(
            {
                "name": pa.array(["Alice", "Bob", "Charlie"]),
                "age": pa.array([30, 25, 35]),
            }
        )

    def _write_arrow_ipc(self, path):
        """Write an Arrow IPC stream file."""
        with pa.OSFile(path, "wb") as f:
            writer = pa.ipc.new_stream(f, self.table.schema)
            writer.write_table(self.table)
            writer.close()

    def _write_feather(self, path):
        """Write a Feather file."""
        pa.feather.write_feather(self.table, path)

    def test_all_extensions_yield_correct_records(self):
        """Test .arrow, .feather, .ipc all produce correct records."""
        cases = [
            (".arrow", self._write_arrow_ipc),
            (".feather", self._write_feather),
            (".ipc", self._write_arrow_ipc),
        ]
        for ext, writer in cases:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                writer(f.name)
                try:
                    loader = ArrowDatasetLoader()
                    loader.load(f.name)
                    result = list(loader.dataset())
                    self.assertEqual(len(result), 3, f"Failed for extension {ext}")
                finally:
                    os.unlink(f.name)

    def test_corrupted_file_raises_data_prep_error(self):
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            f.write(b"this is not a valid arrow file")
            f.flush()
            try:
                loader = ArrowDatasetLoader()
                loader.load(f.name)
                with self.assertRaises(DataPrepError) as ctx:
                    list(loader.dataset())
                self.assertIn(f.name, str(ctx.exception))
            finally:
                os.unlink(f.name)

    def test_load_path_set_after_load(self):
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            self._write_arrow_ipc(f.name)
            try:
                loader = ArrowDatasetLoader()
                self.assertIsNone(loader._load_path)
                loader.load(f.name)
                # resolve_path resolves symlinks, so compare resolved paths
                expected = str(Path(f.name).resolve())
                self.assertEqual(loader._load_path, expected)
            finally:
                os.unlink(f.name)

    def test_ipc_file_format_fallback(self):
        """A .arrow file in IPC File format (not Stream) should load via fallback."""
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            # Write IPC File format (has footer, supports random access)
            writer = pa.ipc.new_file(f.name, self.table.schema)
            writer.write_table(self.table)
            writer.close()
            try:
                loader = ArrowDatasetLoader()
                loader.load(f.name)
                result = list(loader.dataset())
                self.assertEqual(len(result), 3)
                self.assertEqual(result[0]["name"], "Alice")
            finally:
                os.unlink(f.name)

    def test_lazy_iteration_partial_consumption(self):
        """Partial consumption should not require loading all batches."""
        large_table = pa.table(
            {
                "id": pa.array(list(range(10000))),
                "value": pa.array([f"val_{i}" for i in range(10000)]),
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            with pa.OSFile(f.name, "wb") as out:
                writer = pa.ipc.new_stream(out, large_table.schema)
                # Write in small batches to enable lazy iteration
                for batch in large_table.to_batches(max_chunksize=100):
                    writer.write_batch(batch)
                writer.close()
            try:
                loader = ArrowDatasetLoader()
                loader.load(f.name)
                gen = loader.dataset()
                first_five = [next(gen) for _ in range(5)]
                self.assertEqual(len(first_five), 5)
                self.assertEqual(first_five[0]["id"], 0)
            finally:
                os.unlink(f.name)

    @patch("amzn_nova_forge.dataset.dataset_loader.pafs.S3FileSystem")
    def test_s3_loading(self, mock_s3fs_cls):
        """Test S3 loading with mocked S3FileSystem."""
        mock_fs = MagicMock()
        mock_s3fs_cls.return_value = mock_fs

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            self._write_arrow_ipc(f.name)
            try:
                mock_fs.open_input_file.return_value = pa.OSFile(f.name, "r")

                loader = ArrowDatasetLoader()
                loader.load("s3://my-bucket/data/file.arrow")
                result = list(loader.dataset())

                self.assertEqual(len(result), 3)
                mock_s3fs_cls.assert_called_once()
                mock_fs.open_input_file.assert_called_once_with("my-bucket/data/file.arrow")
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
