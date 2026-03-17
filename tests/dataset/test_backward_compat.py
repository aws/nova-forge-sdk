"""
Tests that verify backward compatibility with the old DatasetLoader API.

All deprecated call patterns should continue to work and produce correct results,
while emitting a deprecation log warning via the package logger.
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from amzn_nova_forge.dataset.dataset_loader import (
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
)
from amzn_nova_forge.model.model_enums import Model, TrainingMethod

LOGGER_NAME = "nova_forge_sdk"


class TestBackwardCompatTransform(unittest.TestCase):
    """Test that old transform() call patterns still work."""

    def test_positional_args(self):
        """transform(TrainingMethod.X, Model.Y) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.transform(TrainingMethod.SFT_LORA, Model.NOVA_LITE)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        # Should still produce valid output
        data = list(loader.dataset())
        self.assertGreater(len(data), 0)

    def test_positional_args_with_eval_task(self):
        """transform(TrainingMethod.X, Model.Y, eval_task) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.transform(TrainingMethod.SFT_LORA, Model.NOVA_LITE, None)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        data = list(loader.dataset())
        self.assertGreater(len(data), 0)

    def test_named_method_arg(self):
        """transform(method=TrainingMethod.X, model=Model.Y) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.transform(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        data = list(loader.dataset())
        self.assertGreater(len(data), 0)

    def test_constructor_column_mappings(self):
        """JSONLDatasetLoader(question="q", answer="a") should work."""
        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader = CSVDatasetLoader(question="question", answer="answer")
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        loader.load("tests/test_data/csv_train_test_data.csv")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING):
            loader.transform(TrainingMethod.SFT_LORA, Model.NOVA_LITE)

        data = list(loader.dataset())
        self.assertGreater(len(data), 0)


class TestBackwardCompatValidate(unittest.TestCase):
    """Test that old validate() call patterns still work."""

    def test_positional_args(self):
        """validate(TrainingMethod.X, Model.Y) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

    def test_positional_args_with_eval_task(self):
        """validate(TrainingMethod.X, Model.Y, eval_task) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE, None)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

    def test_named_method_arg(self):
        """validate(method=TrainingMethod.X, model=Model.Y) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/sft_train_samples_converse.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))


class TestBackwardCompatSaveAndSplit(unittest.TestCase):
    """Test that old save_data() and split_data() still work."""

    def test_save_data(self):
        """save_data() should work and emit deprecation log warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = JSONDatasetLoader()
            loader.dataset = lambda: iter([{"key": "value"}])
            save_path = Path(tmpdir) / "output.json"

            with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
                result = loader.save_data(str(save_path))
            self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

            self.assertEqual(result, str(save_path))
            with open(save_path) as f:
                self.assertEqual(json.load(f), {"key": "value"})

    def test_split_data(self):
        """split_data() should work and emit deprecation log warning."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/rft_test_data.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            train, val, test = loader.split_data()
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        self.assertGreater(len(list(train.dataset())), 0)
        self.assertGreater(len(list(val.dataset())), 0)
        self.assertGreater(len(list(test.dataset())), 0)

    def test_split_data_positional_args(self):
        """split_data(0.7, 0.2, 0.1) should work."""
        loader = JSONLDatasetLoader()
        loader.load("tests/test_data/rft_test_data.jsonl")

        with self.assertLogs(LOGGER_NAME, level=logging.WARNING) as cm:
            train, val, test = loader.split_data(0.7, 0.2, 0.1)
        self.assertTrue(any("deprecated" in msg.lower() for msg in cm.output))

        total = (
            len(list(train.dataset()))
            + len(list(val.dataset()))
            + len(list(test.dataset()))
        )
        self.assertEqual(total, 10)
