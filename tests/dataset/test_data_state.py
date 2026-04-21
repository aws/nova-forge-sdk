"""Unit tests for DataState and loader _get_format()."""

import unittest
from unittest.mock import MagicMock

from amzn_nova_forge.dataset.data_state import (
    DataLocation,
    DataState,
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


if __name__ == "__main__":
    unittest.main()
