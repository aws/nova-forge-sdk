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
"""Unit tests for NovaForgeFilterOperationBase.prepare_input().

Tests the actual branching logic on a real subclass (DefaultTextFilterOperation),
not through mocked get_filter_operation().
"""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
    DefaultTextFilterOperation,
)


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.op = DefaultTextFilterOperation()
        self.dummy_gen = lambda: iter([{"text": "hello"}])

    @patch("amzn_nova_forge.dataset.operations.filter_operation.convert_to_s3_parquet")
    def test_arrow_format_triggers_conversion(self, mock_convert):
        """Arrow format (not Glue-compatible) should trigger convert_to_s3_parquet."""
        mock_convert.return_value = "s3://bucket/_forge_converted_input/abc/"
        state = DataState(
            path="s3://bucket/data.arrow",
            format="arrow",
            location=DataLocation.S3,
            generator=self.dummy_gen,
        )
        result = self.op.prepare_input(state, output_path="s3://bucket/output/")
        mock_convert.assert_called_once_with(self.dummy_gen, "s3://bucket/output/")
        self.assertEqual(result.format, "parquet")
        self.assertEqual(result.location, DataLocation.S3)
        self.assertEqual(result.path, "s3://bucket/_forge_converted_input/abc/")

    @patch("amzn_nova_forge.dataset.operations.filter_operation.convert_to_s3_parquet")
    def test_csv_format_triggers_conversion(self, mock_convert):
        """CSV format (not Glue-compatible) should trigger convert_to_s3_parquet."""
        mock_convert.return_value = "s3://bucket/_forge_converted_input/def/"
        state = DataState(
            path="/tmp/data.csv",
            format="csv",
            location=DataLocation.LOCAL,
            generator=self.dummy_gen,
        )
        result = self.op.prepare_input(state, output_path="s3://bucket/output/")
        mock_convert.assert_called_once()
        self.assertEqual(result.format, "parquet")

    @patch("amzn_nova_forge.dataset.operations.filter_operation.upload_local_file_to_s3")
    def test_local_jsonl_triggers_upload(self, mock_upload):
        """Local JSONL (Glue-compatible but local) should trigger upload."""
        mock_upload.return_value = "s3://bucket/_forge_uploaded_input/abc/data.jsonl"
        state = DataState(
            path="/tmp/data.jsonl",
            format="jsonl",
            location=DataLocation.LOCAL,
            generator=self.dummy_gen,
        )
        result = self.op.prepare_input(state, output_path="s3://bucket/output/")
        mock_upload.assert_called_once_with("/tmp/data.jsonl", "s3://bucket/output/")
        self.assertEqual(result.format, "jsonl")
        self.assertEqual(result.location, DataLocation.S3)
        self.assertEqual(result.path, "s3://bucket/_forge_uploaded_input/abc/data.jsonl")

    def test_s3_parquet_passthrough(self):
        """S3 Parquet (Glue-compatible, already on S3) should pass through unchanged."""
        state = DataState(
            path="s3://bucket/data.parquet",
            format="parquet",
            location=DataLocation.S3,
            generator=self.dummy_gen,
        )
        result = self.op.prepare_input(state, output_path="s3://bucket/output/")
        self.assertIs(result, state)

    def test_s3_jsonl_passthrough(self):
        """S3 JSONL should pass through unchanged."""
        state = DataState(
            path="s3://bucket/data.jsonl",
            format="jsonl",
            location=DataLocation.S3,
            generator=self.dummy_gen,
        )
        result = self.op.prepare_input(state, output_path="s3://bucket/output/")
        self.assertIs(result, state)

    def test_conversion_without_s3_output_raises(self):
        """Arrow conversion without S3 output_path should raise ValueError."""
        state = DataState(
            path="/tmp/data.arrow",
            format="arrow",
            location=DataLocation.LOCAL,
            generator=self.dummy_gen,
        )
        with self.assertRaises(ValueError) as ctx:
            self.op.prepare_input(state, output_path="/tmp/output/")
        self.assertIn("S3", str(ctx.exception))

    def test_upload_without_s3_output_raises(self):
        """Local upload without S3 output_path should raise ValueError."""
        state = DataState(
            path="/tmp/data.jsonl",
            format="jsonl",
            location=DataLocation.LOCAL,
            generator=self.dummy_gen,
        )
        with self.assertRaises(ValueError) as ctx:
            self.op.prepare_input(state, output_path="/tmp/output/")
        self.assertIn("S3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
