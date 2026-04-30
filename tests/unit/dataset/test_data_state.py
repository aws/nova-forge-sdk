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
"""Unit tests for DataState, OutputPathResolver, and loader _get_format()."""

import unittest
from enum import Enum
from unittest.mock import MagicMock

from amzn_nova_forge.dataset.data_state import (
    DataLocation,
    DataState,
    OutputPathResolver,
    PathSuffix,
)
from amzn_nova_forge.dataset.dataset_loader import (
    ArrowDatasetLoader,
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
    ParquetDatasetLoader,
)


class TestDataStateFromLoader(unittest.TestCase):
    """Tests for DataState.from_loader() with different loader types."""

    def _make_loader(self, load_path, fmt, dataset=None):
        loader = MagicMock()
        loader._load_path = load_path
        loader._get_format.return_value = fmt
        loader.dataset = dataset or (lambda: iter([]))
        return loader

    def test_s3_jsonl_loader(self):
        loader = self._make_loader("s3://bucket/data.jsonl", "jsonl")
        state = DataState.from_loader(loader)
        self.assertEqual(state.path, "s3://bucket/data.jsonl")
        self.assertEqual(state.format, "jsonl")
        self.assertEqual(state.location, DataLocation.S3)
        self.assertIs(state.generator, loader.dataset)

    def test_local_parquet_loader(self):
        loader = self._make_loader("/tmp/data.parquet", "parquet")
        state = DataState.from_loader(loader)
        self.assertEqual(state.path, "/tmp/data.parquet")
        self.assertEqual(state.format, "parquet")
        self.assertEqual(state.location, DataLocation.LOCAL)

    def test_local_arrow_loader(self):
        loader = self._make_loader("/tmp/data.arrow", "arrow")
        state = DataState.from_loader(loader)
        self.assertEqual(state.format, "arrow")
        self.assertEqual(state.location, DataLocation.LOCAL)

    def test_s3_csv_loader(self):
        loader = self._make_loader("s3://bucket/data.csv", "csv")
        state = DataState.from_loader(loader)
        self.assertEqual(state.format, "csv")
        self.assertEqual(state.location, DataLocation.S3)

    def test_none_load_path_raises_valueerror(self):
        loader = self._make_loader(None, "jsonl")
        with self.assertRaises(ValueError) as ctx:
            DataState.from_loader(loader)
        self.assertIn("No data source provided", str(ctx.exception))


class TestGetFormat(unittest.TestCase):
    """Tests for _FORMAT class attribute and _get_format() on each loader."""

    def test_jsonl_format(self):
        self.assertEqual(JSONLDatasetLoader()._get_format(), "jsonl")

    def test_json_format(self):
        self.assertEqual(JSONDatasetLoader()._get_format(), "json")

    def test_csv_format(self):
        self.assertEqual(CSVDatasetLoader()._get_format(), "csv")

    def test_parquet_format(self):
        self.assertEqual(ParquetDatasetLoader()._get_format(), "parquet")

    def test_arrow_format(self):
        self.assertEqual(ArrowDatasetLoader()._get_format(), "arrow")


class TestOutputPathResolver(unittest.TestCase):
    """Tests for OutputPathResolver.resolve_prefix() and resolve_path()."""

    def test_resolve_prefix_string_method_artifact_suffix(self):
        resolver = OutputPathResolver("s3://bucket/raw/train.jsonl", "2026-04-22_18-30-28")
        prefix = resolver.resolve_prefix("analyze", suffix=PathSuffix.ARTIFACT)
        self.assertEqual(prefix, "train/2026-04-22_18-30-28/analyze_artifact")

    def test_resolve_prefix_enum_method_default_suffix(self):
        class FakeMethod(Enum):
            DEFAULT_TEXT_FILTER = "default_text_filter"

        resolver = OutputPathResolver("/home/user/data/train.jsonl", "2026-04-22_18-30-28")
        prefix = resolver.resolve_prefix(FakeMethod.DEFAULT_TEXT_FILTER)
        self.assertEqual(prefix, "train/2026-04-22_18-30-28/default_text_filter_output")

    def test_resolve_path_includes_parent_and_trailing_slash(self):
        resolver = OutputPathResolver("s3://bucket/raw/train.jsonl", "2026-04-22_18-30-28")
        path = resolver.resolve_path("analyze", suffix=PathSuffix.ARTIFACT)
        self.assertEqual(path, "s3://bucket/raw/train/2026-04-22_18-30-28/analyze_artifact/")

    def test_session_id_fallback(self):
        resolver = OutputPathResolver("s3://bucket/train.jsonl")
        self.assertRegex(resolver._session_id, r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


if __name__ == "__main__":
    unittest.main()
