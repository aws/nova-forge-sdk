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
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.dataset_loader import (
    ArrowDatasetLoader,
    JSONLDatasetLoader,
)
from amzn_nova_forge.dataset.operations.base import OperationResult
from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod
from amzn_nova_forge.dataset.operations.utils import (
    _write_parquet_part,
    convert_to_s3_parquet,
    upload_local_file_to_s3,
)


class TestWriteParquetPart(unittest.TestCase):
    def test_writes_valid_parquet_to_s3(self):
        mock_client = MagicMock()
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        _write_parquet_part(mock_client, "bucket", "prefix/", 0, records)

        mock_client.put_object.assert_called_once()
        kw = mock_client.put_object.call_args[1]
        self.assertEqual(kw["Bucket"], "bucket")
        self.assertEqual(kw["Key"], "prefix/part_00000.parquet")
        table = pq.read_table(io.BytesIO(kw["Body"]))
        self.assertEqual(table.num_rows, 2)


class TestConvertToS3Parquet(unittest.TestCase):
    @patch("boto3.client")
    def test_converts_generator_to_parquet_parts(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        records = [{"id": i, "text": f"record_{i}"} for i in range(25)]
        result = convert_to_s3_parquet(lambda: iter(records), "s3://bucket/data/", batch_size=10)

        self.assertTrue(result.startswith("s3://bucket/_forge_converted_input/"))
        self.assertEqual(mock_client.put_object.call_count, 3)

    def test_raises_without_s3_base(self):
        with self.assertRaises(ValueError):
            convert_to_s3_parquet(lambda: iter([]), "/local/path/")


class TestUploadLocalFileToS3(unittest.TestCase):
    @patch("boto3.client")
    def test_uploads_local_file(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(b'{"id": 1}\n')
            f.flush()
            try:
                result = upload_local_file_to_s3(f.name, "s3://bucket/output/")
                self.assertTrue(result.startswith("s3://bucket/"))
                self.assertTrue(result.endswith(".jsonl"))
                mock_client.put_object.assert_called_once()
            finally:
                os.unlink(f.name)

    def test_raises_without_s3_base(self):
        with self.assertRaises(ValueError):
            upload_local_file_to_s3("/local/file.jsonl", "/local/output/")


class TestExecuteFilterConversion(unittest.TestCase):
    """Test that execute() passes DataState to operations for conversion/upload."""

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_arrow_passes_state_to_operation(self, mock_get_op):
        """Arrow loader + execute() should pass DataState with arrow format to the operation."""
        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/output/", format="parquet", location=DataLocation.S3
            ),
        )
        mock_get_op.return_value = mock_op

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            table = pa.table({"text": ["hello"]})
            writer = pa.ipc.new_stream(f.name, table.schema)
            writer.write_table(table)
            writer.close()
            try:
                loader = ArrowDatasetLoader()
                loader.load(f.name)
                loader.filter(
                    method=FilterMethod.DEFAULT_TEXT_FILTER,
                    output_path="s3://bucket/output/",
                ).execute()
                # Verify the operation received a DataState via kwargs
                call_kwargs = mock_op.execute.call_args[1]
                state = call_kwargs["state"]
                self.assertIsInstance(state, DataState)
                self.assertEqual(state.format, "arrow")
                self.assertEqual(state.location, DataLocation.LOCAL)
            finally:
                os.unlink(f.name)

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_local_jsonl_passes_state_to_operation(self, mock_get_op):
        """Local JSONL loader + execute() should pass DataState with jsonl format to the operation."""
        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/output/", format="jsonl", location=DataLocation.S3
            ),
        )
        mock_get_op.return_value = mock_op

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"text": "hello"}\n')
            f.flush()
            try:
                loader = JSONLDatasetLoader()
                loader.load(f.name)
                loader.filter(
                    method=FilterMethod.DEFAULT_TEXT_FILTER,
                    output_path="s3://bucket/output/",
                ).execute()
                # Verify the operation received a DataState via kwargs
                call_kwargs = mock_op.execute.call_args[1]
                state = call_kwargs["state"]
                self.assertIsInstance(state, DataState)
                self.assertEqual(state.format, "jsonl")
                self.assertEqual(state.location, DataLocation.LOCAL)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
