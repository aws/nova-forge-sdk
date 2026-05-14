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

import pytest

from amzn_nova_forge.dataset.arrow_dataset_loader import ArrowDatasetLoader
from amzn_nova_forge.dataset.csv_dataset_loader import CSVDatasetLoader
from amzn_nova_forge.dataset.data_state import (
    DataLocation,
    DataState,
    OutputPathResolver,
    PathSuffix,
)
from amzn_nova_forge.dataset.json_dataset_loader import JSONDatasetLoader
from amzn_nova_forge.dataset.jsonl_dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.parquet_dataset_loader import ParquetDatasetLoader


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

    def test_local_path_rebases_to_dataprep_bucket(self):
        """Local path should be rebased to the data-prep bucket."""
        resolver = OutputPathResolver("/home/user/data/train.jsonl", "2026-05-04_12-00-00")

        path = resolver.resolve_path("default_text_filter")
        self.assertTrue(path.startswith("s3://"), f"Expected S3 path, got: {path}")
        self.assertIn("train", path)

    def test_hf_path_rebases_to_dataprep_bucket(self):
        """HuggingFace path should be rebased to the data-prep bucket."""
        resolver = OutputPathResolver("hf://fake-org/fake-dataset/train", "2026-05-04_12-00-00")

        path = resolver.resolve_path("default_text_filter")
        self.assertTrue(path.startswith("s3://"), f"Expected S3 path, got: {path}")
        self.assertIn("train", path)

    def test_s3_path_not_rebased(self):
        """S3 path should NOT be rebased — uses the original parent."""
        resolver = OutputPathResolver("s3://my-bucket/data/train.jsonl", "2026-05-04_12-00-00")
        path = resolver.resolve_path("default_text_filter")
        self.assertTrue(
            path.startswith("s3://my-bucket/"), f"Expected original bucket, got: {path}"
        )


@pytest.mark.parametrize(
    "path, expected_location",
    [
        ("s3://bucket/data.jsonl", DataLocation.S3),
        ("/tmp/local/data.jsonl", DataLocation.LOCAL),
        ("hf://foo/bar/train_sft", DataLocation.HUGGINGFACE),
    ],
    ids=["s3", "local", "huggingface"],
)
def test_from_loader_location_detection(path, expected_location):
    """DataState.from_loader() maps path prefixes to the correct DataLocation."""
    loader = MagicMock()
    loader._load_path = path
    loader._get_format.return_value = "jsonl"
    loader.dataset = lambda: iter([])

    state = DataState.from_loader(loader)

    assert state.location == expected_location
    assert state.path == path


if __name__ == "__main__":
    unittest.main()
