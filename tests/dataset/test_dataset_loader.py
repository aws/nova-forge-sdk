import json
import unittest
from unittest.mock import patch

import fsspec  # type: ignore

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        """Sets up variables that might get reused throughout the test suite for easy access."""
        with open("tests/test_data/sft_train_samples_converse.jsonl", "r") as f:
            self.test_data = [json.loads(line.strip()) for line in f]
            self.converse_first_row = self.test_data[0]

        self.openai_rft_first_row = {
            "id": "chem-001",
            "messages": [
                {"content": "You are a helpful chemistry assistant", "role": "system"},
                {
                    "content": "Predict hydrogen bond donors and acceptors for this "
                    "SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    "role": "user",
                },
            ],
            "reference_answer": {"acceptor_bond_counts": 4, "donor_bond_counts": 2},
        }

        self.openai_rft_first_row_transformed = {
            "messages": [
                {"content": "You are a helpful AI assistant.", "role": "system"},
                {
                    "content": "Predict hydrogen bond donors and acceptors for this "
                    "SMILES: CCN(CC)CCC(=O)c1sc(N)nc1C",
                    "role": "user",
                },
            ],
            "reference_answer": {"acceptor_bond_counts": 4, "donor_bond_counts": 2},
        }

        self.generic_to_converse_first_row = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [
                {
                    "text": "You are a helpful assistant who answers the question based on the task assigned"
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Who was the 16th President of the United States?"}
                    ],
                },
                {"role": "assistant", "content": [{"text": "Abraham Lincoln"}]},
            ],
        }

    @patch("fsspec.open")
    def test_load_json_dataset_from_s3(self, mock_fsspec_open):
        # Mock using fsspec.open to read from s3.
        mock_fsspec_open.return_value = open(
            "tests/test_data/sft_train_samples_converse.jsonl", "r"
        )

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(
            path="s3://sdk-test-bucket-read/sft_train_samples_converse.jsonl"
        )

        mock_fsspec_open.assert_called_once_with(
            "s3://sdk-test-bucket-read/sft_train_samples_converse.jsonl",
            "r",
            encoding="utf-8-sig",
        )
        self.assertEqual(dataset_loader.raw_dataset[0], self.converse_first_row)

    def test_load_csv_dataset(self):
        dataset_loader = CSVDatasetLoader()
        dataset_loader.load(path="tests/test_data/csv_train_test_data.csv")
        dataset_loader.show(n=1)

        expected_first_row = {
            "question": "Who was the 16th President of the United States?",
            "answer": "Abraham Lincoln",
        }
        self.assertEqual(dataset_loader.raw_dataset[0], expected_first_row)

    def test_load_json_dataset(self):
        dataset_loader = JSONDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.json")
        dataset_loader.show(n=1)

        self.assertEqual(dataset_loader.raw_dataset[0], self.converse_first_row)

    def test_converse_format_to_converse_format_sft(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            TrainingMethod.SFT_LORA, Model.NOVA_LITE
        )  # Already in converse form - no change.
        self.assertEqual(dataset_loader.transformed_dataset[0], self.converse_first_row)

    def test_openai_format_to_openai_format_rft(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            TrainingMethod.RFT, Model.NOVA_LITE_2
        )  # Already in openai format - no change.
        self.assertEqual(
            dataset_loader.transformed_dataset[0], self.openai_rft_first_row
        )

    def test_generic_format_to_converse_format(self):
        dataset_loader = CSVDatasetLoader(question="question", answer="answer")
        dataset_loader.load(path="tests/test_data/csv_train_test_data.csv")
        dataset_loader.show(n=1)

        dataset_loader.transform(TrainingMethod.SFT_LORA, Model.NOVA_LITE)
        dataset_loader.show(n=1)

        self.assertEqual(
            dataset_loader.transformed_dataset[0], self.generic_to_converse_first_row
        )
        self.assertNotEqual(
            dataset_loader.raw_dataset[0], self.generic_to_converse_first_row
        )  # Check that original dataset isn't changed.

    def test_generic_format_to_openai_format(self):
        dataset_loader = JSONLDatasetLoader(
            question="question", reference_answer="reference_answer"
        )
        dataset_loader.load(path="tests/test_data/unstructured_rft_data.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(TrainingMethod.RFT, Model.NOVA_LITE_2)
        dataset_loader.show(n=1)

        self.assertEqual(
            dataset_loader.transformed_dataset[0], self.openai_rft_first_row_transformed
        )
        self.assertNotEqual(
            dataset_loader.raw_dataset[0], self.openai_rft_first_row_transformed
        )

    def test_split_data_default_values(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        expected_train, expected_val, expected_test = 8, 1, 1

        train, val, test = dataset_loader.split_data()

        # Assert split sizes match expected sizes
        self.assertEqual(
            len(train.raw_dataset),
            expected_train,
            f"Train size {len(train.raw_dataset)} doesn't match expected size {expected_train}",
        )
        self.assertEqual(
            len(val.raw_dataset),
            expected_val,
            f"Validation size {len(val.raw_dataset)} doesn't match expected size {expected_val}",
        )
        self.assertEqual(
            len(test.raw_dataset),
            expected_test,
            f"Test size {len(test.raw_dataset)} doesn't match expected size {expected_test}",
        )

    def test_split_data_custom_values(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        expected_train, expected_val, expected_test = 7, 2, 1

        train, val, test = dataset_loader.split_data(0.7, 0.2, 0.1)

        # Assert split sizes match expected sizes
        self.assertEqual(
            len(train.raw_dataset),
            expected_train,
            f"Train size {len(train.raw_dataset)} doesn't match expected size {expected_train}",
        )
        self.assertEqual(
            len(val.raw_dataset),
            expected_val,
            f"Validation size {len(val.raw_dataset)} doesn't match expected size {expected_val}",
        )
        self.assertEqual(
            len(test.raw_dataset),
            expected_test,
            f"Test size {len(test.raw_dataset)} doesn't match expected size {expected_test}",
        )
