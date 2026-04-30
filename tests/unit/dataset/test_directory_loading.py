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
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.file_utils import is_directory, scan_s3_directory
from amzn_nova_forge.dataset.operations.base import DataPrepError


class TestDirectoryLoading(unittest.TestCase):
    def setUp(self):
        self.records_a = [{"id": 1, "text": "aaa"}, {"id": 2, "text": "bbb"}]
        self.records_b = [{"id": 3, "text": "ccc"}]
        self.records_c = [{"id": 4, "text": "ddd"}, {"id": 5, "text": "eee"}]

    def _write_jsonl(self, path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_valid_files_loaded_in_lexicographic_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl(os.path.join(tmpdir, "c.jsonl"), self.records_c)
            self._write_jsonl(os.path.join(tmpdir, "a.jsonl"), self.records_a)
            self._write_jsonl(os.path.join(tmpdir, "b.jsonl"), self.records_b)
            loader = JSONLDatasetLoader()
            loader.load(tmpdir)
            results = list(loader.dataset())
            self.assertEqual(results, self.records_a + self.records_b + self.records_c)

    def test_invalid_files_raise_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl(os.path.join(tmpdir, "data.jsonl"), self.records_a)
            with open(os.path.join(tmpdir, "stray.csv"), "w") as f:
                f.write("id,text\n1,hello\n")
            with self.assertRaises(DataPrepError) as ctx:
                JSONLDatasetLoader().load(tmpdir)
            self.assertIn("stray.csv", str(ctx.exception))

    def test_empty_directory_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(DataPrepError) as ctx:
                JSONLDatasetLoader().load(tmpdir)
            self.assertIn(".jsonl", str(ctx.exception))

    def test_load_path_set_to_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl(os.path.join(tmpdir, "data.jsonl"), self.records_a)
            loader = JSONLDatasetLoader()
            loader.load(tmpdir)
            self.assertEqual(loader._load_path, str(Path(tmpdir).resolve()))

    def test_lazy_iteration_processes_files_sequentially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_jsonl(os.path.join(tmpdir, "a.jsonl"), self.records_a)
            self._write_jsonl(os.path.join(tmpdir, "b.jsonl"), self.records_b)
            gen = JSONLDatasetLoader().load(tmpdir).dataset()
            self.assertEqual(next(gen), self.records_a[0])
            self.assertEqual(next(gen), self.records_a[1])
            self.assertEqual(next(gen), self.records_b[0])

    def test_s3_directory_discovery(self):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/train/a.jsonl"},
                    {"Key": "data/train/b.jsonl"},
                ]
            }
        ]
        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        with patch("boto3.client", return_value=mock_client):
            files = scan_s3_directory("s3://bucket/data/train/", {".jsonl"})
        self.assertEqual(
            files,
            [
                "s3://bucket/data/train/a.jsonl",
                "s3://bucket/data/train/b.jsonl",
            ],
        )


class TestHiddenFilesIgnored(unittest.TestCase):
    def test_hidden_and_metadata_files_ignored(self):
        records = [{"id": 1, "text": "aaa"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, ".DS_Store"), "w") as f:
                f.write("")
            with open(os.path.join(tmpdir, "_SUCCESS"), "w") as f:
                f.write("")
            loader = JSONLDatasetLoader()
            loader.load(tmpdir)
            results = list(loader.dataset())
            assert results == records


class TestScanS3DirectoryErrorPaths(unittest.TestCase):
    def test_unexpected_extensions_raises_error(self):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/train/a.jsonl"},
                    {"Key": "data/train/b.csv"},
                ]
            }
        ]
        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        with patch("boto3.client", return_value=mock_client):
            with self.assertRaises(DataPrepError) as ctx:
                scan_s3_directory("s3://bucket/data/train/", {".jsonl"})
            self.assertIn("b.csv", str(ctx.exception))

    def test_no_matching_files_raises_error(self):
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/train/readme.txt"},
                ]
            }
        ]
        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        with patch("boto3.client", return_value=mock_client):
            with self.assertRaises(DataPrepError) as ctx:
                scan_s3_directory("s3://bucket/data/train/", {".jsonl"})
            self.assertIn(".jsonl", str(ctx.exception))


class TestIsDirectory(unittest.TestCase):
    def test_s3_uri_with_trailing_slash(self):
        self.assertTrue(is_directory("s3://bucket/prefix/"))

    def test_s3_uri_without_trailing_slash(self):
        self.assertFalse(is_directory("s3://bucket/file.parquet"))

    def test_local_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(is_directory(tmpdir))

    def test_local_file(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertFalse(is_directory(f.name))


if __name__ == "__main__":
    unittest.main()
