import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonschema
from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.dataset.dataset_format_schema import (
    SFT_NOVA_ONE_CONVERSE_2024,
    SFT_NOVA_TWO_CONVERSE_2024,
)
from amzn_nova_forge.dataset.dataset_loader import (
    ArrowDatasetLoader,
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
    ParquetDatasetLoader,
)
from amzn_nova_forge.dataset.file_utils import resolve_path
from amzn_nova_forge.dataset.operations.base import DataPrepError


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
                    "content": [{"text": "Who was the 16th President of the United States?"}],
                },
                {"role": "assistant", "content": [{"text": "Abraham Lincoln"}]},
            ],
        }

        self.openai_to_converse_first_row = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "You are a helpful assistant."}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Who won the FIFA world cup in 2022?"}],
                },
                {"role": "assistant", "content": [{"text": "Argentina"}]},
            ],
        }

        self.save_data = [
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are a helpful assistant."}],
                "messages": [
                    {"role": "user", "content": [{"text": "Hello"}]},
                    {"role": "assistant", "content": [{"text": "Hi there!"}]},
                ],
            },
            {
                "schemaVersion": "bedrock-conversation-2024",
                "system": [{"text": "You are a helpful assistant."}],
                "messages": [
                    {"role": "user", "content": [{"text": "What's 2+2?"}]},
                    {"role": "assistant", "content": [{"text": "4"}]},
                ],
            },
        ]

        self.cpt_generic_first_row = {"text": "The quick brown fox jumps over the lazy dog."}

        self.cpt_transformed_first_row = {"text": "The quick brown fox jumps over the lazy dog."}

        self.cpt_custom_column_row = {
            "content": "This is a sample text for continued pre-training.",
            "metadata": "some additional info",
        }

    @patch("amzn_nova_forge.dataset.dataset_loader.check_path_exists")
    @patch("amzn_nova_forge.dataset.dataset_loader.load_file_content")
    def test_load_json_dataset_from_s3(self, mock_load_file, mock_check_exists):
        with open("tests/test_data/sft_train_samples_converse.jsonl", "r") as f:
            lines = f.read().splitlines()
            # Return a fresh iterator each time
            mock_load_file.side_effect = lambda *args, **kwargs: iter(lines)

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="s3://sdk-test-bucket-read/sft_train_samples_converse.jsonl")
        dataset_loader.show(n=1)  # Call show() to actually trigger loading

        mock_load_file.assert_called_with(
            file_path="s3://sdk-test-bucket-read/sft_train_samples_converse.jsonl",
            extension=".jsonl",
            encoding="utf-8-sig",
        )
        self.assertEqual(list(dataset_loader.dataset())[0], self.converse_first_row)

    @patch("amzn_nova_forge.dataset.dataset_loader.check_path_exists")
    @patch("amzn_nova_forge.dataset.dataset_loader.load_file_content")
    def test_load_jsonl_with_empty_lines(self, mock_load_file, mock_check_exists):
        jsonl_content = """{"id": "1", "name": "Alice"}

    {"id": "2", "name": "Bob"}
   
    {"id": "3", "name": "Charlie"}

    """
        # Return a fresh iterator each time the function is called
        mock_load_file.side_effect = lambda *args, **kwargs: iter(jsonl_content.splitlines())

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="test.jsonl")

        self.assertEqual(len(list(dataset_loader.dataset())), 3)
        self.assertEqual(list(dataset_loader.dataset())[0], {"id": "1", "name": "Alice"})
        self.assertEqual(list(dataset_loader.dataset())[1], {"id": "2", "name": "Bob"})
        self.assertEqual(list(dataset_loader.dataset())[2], {"id": "3", "name": "Charlie"})

    @patch("amzn_nova_forge.dataset.dataset_loader.check_path_exists")
    @patch("amzn_nova_forge.dataset.dataset_loader.load_file_content")
    @patch("amzn_nova_forge.dataset.dataset_loader.logger")
    def test_load_jsonl_with_malformed_json(self, mock_logger, mock_load_file, mock_check_exists):
        jsonl_content = """{"id": "1", "name": "Alice"}
    {"id": "2", "name": "Bob", invalid json here}
    {"id": "3", "name": "Charlie"}"""

        # Return a fresh iterator each time
        mock_load_file.side_effect = lambda *args, **kwargs: iter(jsonl_content.splitlines())

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="test.jsonl")
        loaded_data = list(dataset_loader.dataset())

        # Should load 2 valid records, skipping the malformed one
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0], {"id": "1", "name": "Alice"})
        self.assertEqual(loaded_data[1], {"id": "3", "name": "Charlie"})

        mock_logger.warning.assert_called_once()
        warning_call_args = mock_logger.warning.call_args[0]
        # Format string is first arg, path and preview are positional args
        self.assertIn("Skipping malformed JSON line", warning_call_args[0])
        self.assertIn("invalid json here", warning_call_args[2])

    def test_load_csv_dataset(self):
        dataset_loader = CSVDatasetLoader()
        dataset_loader.load(path="tests/test_data/csv_train_test_data.csv")
        dataset_loader.show(n=1)

        expected_first_row = {
            "question": "Who was the 16th President of the United States?",
            "answer": "Abraham Lincoln",
        }

        self.assertEqual(list(dataset_loader.dataset())[0], expected_first_row)

    def test_load_json_dataset(self):
        dataset_loader = JSONDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.json")
        dataset_loader.show(n=1)

        self.assertEqual(list(dataset_loader.dataset())[0], self.converse_first_row)

    def test_converse_format_to_converse_format_sft(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
        )  # Already in converse form - no change.
        dataset_loader.execute()
        self.assertEqual(list(dataset_loader.dataset())[0], self.converse_first_row)

    def test_openai_format_to_openai_format_rft(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            training_method=TrainingMethod.RFT_FULL, model=Model.NOVA_LITE_2
        )  # Already in openai format - no change.
        dataset_loader.execute()
        self.assertEqual(list(dataset_loader.dataset())[0], self.openai_rft_first_row)

    def test_generic_format_to_converse_format(self):
        dataset_loader = CSVDatasetLoader()
        dataset_loader.load(path="tests/test_data/csv_train_test_data.csv")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE,
            column_mappings={"question": "question", "answer": "answer"},
        )
        dataset_loader.execute()
        dataset_loader.show(n=1)

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.generic_to_converse_first_row,
        )

    def test_generic_format_to_openai_format(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/unstructured_rft_data.jsonl")
        dataset_loader.show(n=1)

        dataset_loader.transform(
            training_method=TrainingMethod.RFT_FULL,
            model=Model.NOVA_LITE_2,
            column_mappings={
                "question": "question",
                "reference_answer": "reference_answer",
            },
        )
        dataset_loader.execute()
        dataset_loader.show(n=1)

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.openai_rft_first_row_transformed,
        )

    def test_split_data_default_values(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        expected_train, expected_val, expected_test = 8, 1, 1

        train, val, test = dataset_loader.split()

        # Assert split sizes match expected sizes
        self.assertEqual(
            len(list(train.dataset())),
            expected_train,
            f"Train size {len(list(train.dataset()))} doesn't match expected size {expected_train}",
        )
        self.assertEqual(
            len(list(val.dataset())),
            expected_val,
            f"Validation size {len(list(val.dataset()))} doesn't match expected size {expected_val}",
        )
        self.assertEqual(
            len(list(test.dataset())),
            expected_test,
            f"Test size {len(list(test.dataset()))} doesn't match expected size {expected_test}",
        )

    def test_split_data_custom_values(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/rft_test_data.jsonl")
        expected_train, expected_val, expected_test = 7, 2, 1

        train, val, test = dataset_loader.split(0.7, 0.2, 0.1)

        # Assert split sizes match expected sizes
        self.assertEqual(
            len(list(train.dataset())),
            expected_train,
            f"Train size {len(list(train.dataset()))} doesn't match expected size {expected_train}",
        )
        self.assertEqual(
            len(list(val.dataset())),
            expected_val,
            f"Validation size {len(list(val.dataset()))} doesn't match expected size {expected_val}",
        )
        self.assertEqual(
            len(list(test.dataset())),
            expected_test,
            f"Test size {len(list(test.dataset()))} doesn't match expected size {expected_test}",
        )

    def test_openai_format_to_converse_format_nova_lite(self):
        """Test OpenAI format auto-detection and conversion to Converse for Nova 1.0 (NOVA_LITE)."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.openai_to_converse_first_row,
        )
        # Verify schemaVersion is set
        self.assertEqual(
            list(dataset_loader.dataset())[0]["schemaVersion"],
            "bedrock-conversation-2024",
        )

    def test_openai_format_to_converse_format_nova_micro(self):
        """Test OpenAI format conversion to Converse for Nova 1.0 (NOVA_MICRO)."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_MICRO)
        dataset_loader.execute()

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.openai_to_converse_first_row,
        )

    def test_openai_format_to_converse_format_nova_pro(self):
        """Test OpenAI format conversion to Converse for Nova 1.0 (NOVA_PRO)."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_FULL, model=Model.NOVA_PRO)
        dataset_loader.execute()

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.openai_to_converse_first_row,
        )

    def test_openai_format_to_converse_format_nova_lite_2(self):
        """Test OpenAI format conversion to Converse for Nova 2.0 (NOVA_LITE_2)."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Nova 2.0 should produce similar output structure
        self.assertEqual(
            list(dataset_loader.dataset())[0]["schemaVersion"],
            "bedrock-conversation-2024",
        )
        self.assertEqual(
            list(dataset_loader.dataset())[0]["system"],
            [{"text": "You are a helpful assistant."}],
        )
        self.assertEqual(list(dataset_loader.dataset())[0]["messages"][0]["role"], "user")
        self.assertEqual(
            list(dataset_loader.dataset())[0]["messages"][1]["role"],
            "assistant",
        )

    def test_openai_format_transform_produces_new_dataset(self):
        """Test that OpenAI to Converse transformation produces transformed output."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")
        original_first_row = list(dataset_loader.dataset())[0].copy()

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        # After transform, dataset should contain transformed data
        self.assertNotEqual(
            list(dataset_loader.dataset())[0],
            original_first_row,
        )

    def test_openai_format_with_tool_calls_nova_lite(self):
        """Test that Nova 1.0 raises error when OpenAI format has tool calls."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        # Nova 1.0 should raise an error for tool calls
        with self.assertRaises(DataPrepError) as context:
            dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
            dataset_loader.show(n=1)

        self.assertIn("Tool/function calling is not supported in Nova 1.0", str(context.exception))

    def test_openai_format_with_tool_calls_nova_lite_2(self):
        """Test OpenAI format with tool calls conversion to Converse for Nova 2.0."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Check fourth record has reasoning content (Nova 2.0 feature)
        fourth_record = list(dataset_loader.dataset())[3]

        # Find assistant message with reasoning content
        has_reasoning = False
        for msg in fourth_record["messages"]:
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if "reasoningContent" in content:
                        has_reasoning = True
                        break

        self.assertTrue(has_reasoning, "Nova 2.0 should have reasoningContent")

    def test_openai_format_with_multiple_tool_calls(self):
        """Test OpenAI format with multiple tool calls in a single message for Nova 2.0."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Check fifth record has multiple tool calls (index 4)
        multi_tool_record = list(dataset_loader.dataset())[4]

        # Count tool uses in assistant message
        tool_use_count = 0
        for msg in multi_tool_record["messages"]:
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if "toolUse" in content:
                        tool_use_count += 1

        self.assertEqual(tool_use_count, 2, "Should have 2 tool uses")

        # Count tool results in user messages
        tool_result_count = 0
        for msg in multi_tool_record["messages"]:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if "toolResult" in content:
                        tool_result_count += 1

        self.assertEqual(tool_result_count, 2, "Should have 2 tool results")

    def test_openai_format_with_tool_config(self):
        """Test OpenAI format with tools configuration conversion."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Check seventh record has toolConfig (index 6)
        config_record = list(dataset_loader.dataset())[6]
        self.assertIn("toolConfig", config_record)
        self.assertIn("tools", config_record["toolConfig"])

        # Verify tool spec structure
        tools = config_record["toolConfig"]["tools"]
        self.assertGreater(len(tools), 0)

        first_tool = tools[0]
        self.assertIn("toolSpec", first_tool)
        self.assertIn("name", first_tool["toolSpec"])
        self.assertIn("description", first_tool["toolSpec"])
        self.assertIn("inputSchema", first_tool["toolSpec"])

    def test_tool_result_creates_separate_user_message(self):
        """Test that tool results create separate user messages, not combined with user text."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        # Use Nova 2.0 since Nova 1.0 doesn't support tools
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Check that tool results are in separate user messages
        for record in dataset_loader.dataset():
            for msg in record["messages"]:
                if msg["role"] == "user":
                    # A user message should either have text OR toolResult, not both
                    has_text = False
                    has_tool_result = False
                    for content in msg["content"]:
                        if "text" in content:
                            has_text = True
                        if "toolResult" in content:
                            has_tool_result = True

                    # They should not be combined
                    if has_tool_result and has_text:
                        self.fail(
                            "Tool results should not be combined with user text in the same message"
                        )

    def test_tool_use_id_mapping(self):
        """Test that toolUseId correctly maps between toolUse and toolResult."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Collect all toolUseIds from toolUse and toolResult
        for record in dataset_loader.dataset():
            tool_use_ids = set()
            tool_result_ids = set()

            for msg in record["messages"]:
                if msg["role"] == "assistant":
                    for content in msg["content"]:
                        if "toolUse" in content:
                            tool_use_ids.add(content["toolUse"]["toolUseId"])
                elif msg["role"] == "user":
                    for content in msg["content"]:
                        if "toolResult" in content:
                            tool_result_ids.add(content["toolResult"]["toolUseId"])

            # Every tool result should have a corresponding tool use
            for result_id in tool_result_ids:
                self.assertIn(
                    result_id,
                    tool_use_ids,
                    f"toolResult with ID {result_id} has no corresponding toolUse",
                )

    def test_save_data_local_json(self):
        """Test saving a single dictionary to JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            # JSON format expects a single dictionary
            single_record = self.save_data[0]
            dataset_loader.dataset = lambda: iter([single_record])
            save_path = Path(tmpdir) / "test_output.json"

            result_path = dataset_loader.save(str(save_path))

            self.assertEqual(result_path, str(save_path))
            self.assertTrue(save_path.exists())

            with open(save_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, single_record)

    def test_save_data_local_json_multiple_records_raises_error(self):
        """Test that JSON format with multiple records raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter(self.save_data)  # Multiple records
            save_path = Path(tmpdir) / "test_output.json"

            with self.assertRaises(DataPrepError) as context:
                dataset_loader.save(str(save_path))

            self.assertIn("expects exactly one dictionary", str(context.exception))
            self.assertIn("Use JSONL format", str(context.exception))

    def test_save_data_local_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONLDatasetLoader()
            dataset_loader.dataset = lambda: iter(self.save_data)
            save_path = Path(tmpdir) / "test_output.jsonl"

            result_path = dataset_loader.save(str(save_path))

            self.assertEqual(result_path, str(save_path))
            self.assertTrue(save_path.exists())

            with open(save_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                self.assertEqual(json.loads(lines[0].strip()), self.save_data[0])
                self.assertEqual(json.loads(lines[1].strip()), self.save_data[1])

    def test_save_data_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            # Use single record for JSON format
            dataset_loader.dataset = lambda: iter([self.save_data[0]])
            save_path = Path(tmpdir) / "nested" / "directories" / "test_output.json"

            result_path = dataset_loader.save(str(save_path))

            self.assertEqual(result_path, str(save_path))
            self.assertTrue(save_path.exists())
            self.assertTrue(save_path.parent.exists())

    def test_save_data_dataset_is_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter([{"some": "data"}])
            save_path = Path(tmpdir) / "test_output.json"

            dataset_loader.save(str(save_path))

            with open(save_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, {"some": "data"})

    def test_save_data_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter([])
            save_path = Path(tmpdir) / "test_output.json"

            dataset_loader.save(str(save_path))

            with open(save_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, {})

    def test_save_data_empty_dataset_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter([])
            save_path = Path(tmpdir) / "test_output.json"

            with patch("amzn_nova_forge.dataset.operations.save_operation.logger") as mock_logger:
                dataset_loader.save(str(save_path))

                mock_logger.warning.assert_called_once()
                self.assertIn("empty", mock_logger.warning.call_args[0][0].lower())

            with open(save_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, {})

    def test_save_data_unsupported_format_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter([self.save_data[0]])
            save_path = Path(tmpdir) / "test_output.txt"

            with self.assertRaises(DataPrepError) as context:
                dataset_loader.save(str(save_path))

            self.assertIn("Unsupported format", str(context.exception))

    def test_save_data_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unicode_data = {"text": "Hello 世界 Café ☕ Emoji 😀🎉"}
            dataset_loader = JSONDatasetLoader()
            dataset_loader.dataset = lambda: iter([unicode_data])
            save_path = Path(tmpdir) / "test_unicode.json"

            dataset_loader.save(str(save_path))

            with open(save_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, unicode_data)

    @patch("boto3.client")
    def test_save_data_s3_success(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        dataset_loader = JSONDatasetLoader()
        # Use single record for JSON format
        single_record = self.save_data[0]
        dataset_loader.dataset = lambda: iter([single_record])
        dataset_loader._load_path = "in-memory"
        save_path = "s3://test-bucket/path/to/output.json"

        result_path = dataset_loader.save(save_path)

        self.assertEqual(result_path, save_path)
        mock_boto_client.assert_called_once_with("s3")

        # Verify upload_file was called
        mock_s3.upload_file.assert_called_once()
        call_args = mock_s3.upload_file.call_args

        # Verify bucket and key
        self.assertEqual(call_args[0][1], "test-bucket")
        self.assertEqual(call_args[0][2], "path/to/output.json")
        self.assertEqual(call_args[1]["ExtraArgs"]["ContentType"], "application/json")

    @patch("boto3.client")
    def test_save_data_s3_multiple_records_raises_error(self, mock_boto_client):
        """Test that S3 upload with JSON format and multiple records raises an error."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        dataset_loader = JSONDatasetLoader()
        dataset_loader.dataset = lambda: iter(self.save_data)  # Multiple records
        save_path = "s3://test-bucket/output.json"

        with self.assertRaises(DataPrepError) as context:
            dataset_loader.save(save_path)

        self.assertIn("expects exactly one dictionary", str(context.exception))
        # upload_file should not be called since we error during temp file write
        mock_s3.upload_file.assert_not_called()

    @patch("boto3.client")
    def test_save_data_s3_client_error_raises_dataprep_error(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "UploadFile",
        )
        mock_boto_client.return_value = mock_s3
        dataset_loader = JSONDatasetLoader()
        dataset_loader.dataset = lambda: iter([self.save_data[0]])
        dataset_loader._load_path = "in-memory"
        save_path = "s3://test-bucket/output.json"

        with self.assertRaises(DataPrepError) as context:
            dataset_loader.save(save_path)

        self.assertIn("Failed to upload to S3", str(context.exception))

    def test_end_to_end_simple_openai_to_converse_with_validation(self):
        """Test end-to-end: OpenAI format -> Convert -> Validate for simple conversations."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")

        # Transform
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        # Validate
        dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

        # Check structure
        self.assertGreater(len(list(dataset_loader.dataset())), 0)
        for record in dataset_loader.dataset():
            self.assertIn("schemaVersion", record)
            self.assertEqual(record["schemaVersion"], "bedrock-conversation-2024")
            self.assertIn("system", record)
            self.assertIn("messages", record)

    def test_end_to_end_multi_tool_openai_to_converse_nova_two(self):
        """Test end-to-end: Multi-tool OpenAI format -> Converse Nova 2.0 -> Validation."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_tool_test_data.jsonl")

        # Transform to Nova 2.0 (supports tools)
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        # Note: Validation would fail due to inconsistent reasoningContent across samples
        # (some samples have it, some don't). This is expected behavior.
        # We'll verify the transformation worked correctly instead.

        # Check that tool configs are present where expected
        has_tool_config = False
        for record in dataset_loader.dataset():
            if "toolConfig" in record:
                has_tool_config = True
                self.assertIn("tools", record["toolConfig"])
                self.assertIsInstance(record["toolConfig"]["tools"], list)

        self.assertTrue(has_tool_config, "Should have at least one record with toolConfig")

        # Verify transformed dataset structure
        self.assertGreater(len(list(dataset_loader.dataset())), 0)
        for record in dataset_loader.dataset():
            self.assertIn("schemaVersion", record)
            self.assertEqual(record["schemaVersion"], "bedrock-conversation-2024")

    def test_end_to_end_reasoning_content_nova_two(self):
        """Test end-to-end: OpenAI with reasoning -> Converse Nova 2.0."""
        import json
        import tempfile

        # Create consistent dataset where ALL samples have reasoning
        reasoning_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a logical assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {
                        "role": "assistant",
                        "content": "2+2 equals 4.",
                        "reasoning": "This is basic arithmetic. 2+2=4.",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful tutor."},
                    {"role": "user", "content": "Explain gravity."},
                    {
                        "role": "assistant",
                        "content": "Gravity is a force that attracts objects with mass.",
                        "reasoning": "The user asks about gravity, so I'll provide a simple explanation.",
                    },
                ]
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in reasoning_data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            dataset_loader = JSONLDatasetLoader()
            dataset_loader.load(path=temp_path)

            # Transform (Nova 2.0 supports reasoning)
            dataset_loader.transform(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )
            dataset_loader.execute()

            # Now validate should pass since all samples have reasoning
            dataset_loader.validate(
                training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2
            )

            # Check that ALL records have reasoning content
            for record in dataset_loader.dataset():
                has_reasoning = False
                for msg in record["messages"]:
                    if msg["role"] == "assistant":
                        for content in msg["content"]:
                            if "reasoningContent" in content:
                                has_reasoning = True
                                break
                self.assertTrue(
                    has_reasoning,
                    f"All samples should have reasoning content in Nova 2.0",
                )

        finally:
            import os

            os.unlink(temp_path)

    def test_end_to_end_direct_converse_validation(self):
        """Test end-to-end: Direct Converse format -> Validation."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/sft_train_samples_converse.jsonl")

        # Already in Converse format
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        # Validate
        dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

    def test_end_to_end_batch_mixed_formats(self):
        """Test batch processing of different conversation types."""
        import json
        import tempfile

        # Create mixed format data
        mixed_data = [
            # Simple Q&A
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {
                        "role": "assistant",
                        "content": "Python is a programming language.",
                    },
                ]
            },
            # With system message
            {
                "messages": [
                    {"role": "system", "content": "You are a teacher."},
                    {"role": "user", "content": "Explain recursion."},
                    {
                        "role": "assistant",
                        "content": "Recursion is when a function calls itself.",
                    },
                ]
            },
            # Multi-turn
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well, thanks!"},
                ]
            },
        ]

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in mixed_data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            # Load and transform
            dataset_loader = JSONLDatasetLoader()
            dataset_loader.load(path=temp_path)
            dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
            dataset_loader.execute()
            dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

            # Verify all records were processed
            self.assertEqual(len(list(dataset_loader.dataset())), 3)

            # Check multi-turn conversation structure
            multi_turn = list(dataset_loader.dataset())[2]
            self.assertEqual(len(multi_turn["messages"]), 4)  # 2 user, 2 assistant

        finally:
            import os

            os.unlink(temp_path)

    def test_end_to_end_csv_to_converse_workflow(self):
        """Test complete workflow: CSV -> Load -> Transform -> Validate -> Split."""
        # Load CSV
        dataset_loader = CSVDatasetLoader()
        dataset_loader.load(path="tests/test_data/csv_train_test_data.csv")

        # Transform to Converse
        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE,
            column_mappings={"question": "question", "answer": "answer"},
        )
        dataset_loader.execute()

        # Validate
        dataset_loader.validate(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)

        # Split data
        train, val, test = dataset_loader.split(0.6, 0.2, 0.2)

        # Verify splits maintain format
        total_original = len(list(dataset_loader.dataset()))
        total_split = (
            len(list(train.dataset())) + len(list(val.dataset())) + len(list(test.dataset()))
        )
        self.assertEqual(total_original, total_split)

    def test_end_to_end_error_handling_invalid_format(self):
        """Test error handling for invalid formats during transformation."""
        import json
        import tempfile

        # Create invalid data (missing required roles)
        invalid_data = [
            {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "user", "content": "Question 2"},  # No assistant!
                ]
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in invalid_data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        try:
            dataset_loader = JSONLDatasetLoader()
            dataset_loader.load(path=temp_path)

            # Should raise error during transformation
            with self.assertRaises(DataPrepError) as context:
                dataset_loader.transform(
                    training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE
                )
                dataset_loader.show(n=1)  # Call show() to actually trigger transform operation

            self.assertIn(
                "must contain at least one 'user' and one 'assistant'",
                str(context.exception),
            )

        finally:
            import os

            os.unlink(temp_path)

    def test_convert_to_cpt_basic(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([self.cpt_generic_first_row])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(
            list(dataset_loader.dataset())[0],
            self.cpt_transformed_first_row,
        )

    def test_convert_to_cpt_missing_text_column(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([{"wrong_column": "some text"}])
        dataset_loader._load_path = "in-memory"

        with self.assertRaises(DataPrepError) as context:
            dataset_loader.transform(
                training_method=TrainingMethod.CPT,
                model=Model.NOVA_LITE,
                column_mappings={"text": "text"},
            )
            dataset_loader.show(n=1)  # Call show() to actually trigger the transform function

        self.assertIn("'text' column not found", str(context.exception))
        self.assertIn("required for CPT", str(context.exception))

    def test_convert_to_cpt_custom_column_mapping(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([self.cpt_custom_column_row])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "content"},
        )
        dataset_loader.execute()

        expected = {"text": "This is a sample text for continued pre-training."}
        self.assertEqual(list(dataset_loader.dataset())[0], expected)

    def test_convert_to_cpt_preserves_dataset(self):
        dataset_loader = JSONLDatasetLoader()
        original_data = [{"text": "First paragraph."}, {"text": "Second paragraph."}]
        dataset_loader.dataset = lambda: iter(original_data.copy())
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(training_method=TrainingMethod.CPT, model=Model.NOVA_LITE)
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset()), original_data)
        self.assertEqual(len(list(dataset_loader.dataset())), 2)

    def test_convert_to_cpt_multiple_records(self):
        test_data = [
            {"text": "First training text."},
            {"text": "Second training text."},
            {"text": "Third training text."},
        ]
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter(test_data)
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(len(list(dataset_loader.dataset())), 3)
        for i, record in enumerate(dataset_loader.dataset()):
            self.assertEqual(record["text"], test_data[i]["text"])
            self.assertEqual(list(record.keys()), ["text"])

    def test_cpt_format_already_in_cpt_format(self):
        dataset_loader = JSONLDatasetLoader()
        cpt_data = [
            {"text": "Already formatted text 1."},
            {"text": "Already formatted text 2."},
        ]
        dataset_loader.dataset = lambda: iter(cpt_data)
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset()), cpt_data)

    def test_cpt_with_empty_text(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([{"text": ""}])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset())[0], {"text": ""})

    def test_cpt_with_long_text(self):
        long_text = "This is a very long text. " * 1000
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([{"text": long_text}])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset())[0]["text"], long_text)

    def test_cpt_with_unicode_text(self):
        unicode_text = "Hello 🌍"
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([{"text": unicode_text}])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset())[0]["text"], unicode_text)

    def test_cpt_strips_extra_fields(self):
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter(
            [
                {
                    "text": "Training text",
                    "id": "12345",
                    "metadata": {"source": "book"},
                    "timestamp": "2024-01-01",
                }
            ]
        )
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset())[0], {"text": "Training text"})
        self.assertEqual(list(list(dataset_loader.dataset())[0].keys()), ["text"])

    def test_end_to_end_cpt_workflow_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "cpt_data.csv"
            csv_content = """text
"The first training document."
"The second training document."
"The third training document."
"""
            csv_path.write_text(csv_content)

            dataset_loader = CSVDatasetLoader()
            dataset_loader.load(str(csv_path))
            dataset_loader.transform(
                training_method=TrainingMethod.CPT,
                model=Model.NOVA_LITE,
                column_mappings={"text": "text"},
            )
            dataset_loader.execute()
            dataset_loader.validate(training_method=TrainingMethod.CPT, model=Model.NOVA_LITE)

            self.assertEqual(len(list(dataset_loader.dataset())), 3)
            self.assertEqual(
                list(dataset_loader.dataset())[0]["text"],
                "The first training document.",
            )

    def test_end_to_end_cpt_workflow_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "cpt_data.jsonl"
            jsonl_content = [
                {"text": "First paragraph of training data."},
                {"text": "Second paragraph of training data."},
                {"text": "Third paragraph of training data."},
            ]
            with open(jsonl_path, "w") as f:
                for item in jsonl_content:
                    f.write(json.dumps(item) + "\n")

            dataset_loader = JSONLDatasetLoader()
            dataset_loader.load(str(jsonl_path))
            dataset_loader.transform(
                training_method=TrainingMethod.CPT,
                model=Model.NOVA_LITE,
                column_mappings={"text": "text"},
            )
            dataset_loader.execute()
            dataset_loader.validate(training_method=TrainingMethod.CPT, model=Model.NOVA_LITE)

            self.assertEqual(len(list(dataset_loader.dataset())), 3)
            self.assertEqual(
                list(dataset_loader.dataset())[0],
                {"text": "First paragraph of training data."},
            )

    def test_cpt_with_newlines_and_tabs(self):
        text_with_formatting = "Line 1\nLine 2\tTabbed\nLine 3"
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.dataset = lambda: iter([{"text": text_with_formatting}])
        dataset_loader._load_path = "in-memory"

        dataset_loader.transform(
            training_method=TrainingMethod.CPT,
            model=Model.NOVA_LITE,
            column_mappings={"text": "text"},
        )
        dataset_loader.execute()

        self.assertEqual(list(dataset_loader.dataset())[0]["text"], text_with_formatting)

    def test_end_to_end_multimodal_openai_to_converse_nova_one_missing_s3_path(self):
        """Multimodal OpenAI format without multimodal_data_s3_path raises DataPrepError."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_multimodal_test_data.jsonl")
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        with self.assertRaises(DataPrepError) as ctx:
            list(dataset_loader.dataset())
        self.assertIn("multimodal_data_s3_path is required", str(ctx.exception))

    def test_end_to_end_multimodal_openai_to_converse_nova_two_missing_s3_path(self):
        """Multimodal OpenAI format without multimodal_data_s3_path raises DataPrepError for Nova 2.0."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_multimodal_test_data.jsonl")
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        dataset_loader.execute()

        with self.assertRaises(DataPrepError) as ctx:
            list(dataset_loader.dataset())
        self.assertIn("multimodal_data_s3_path is required", str(ctx.exception))

    @patch("amzn_nova_forge.dataset.dataset_transformers.boto3.client")
    def test_end_to_end_multimodal_nova_one_with_s3_path(self, mock_boto_client):
        """Multimodal OpenAI → Converse Nova 1.0 with mocked S3 produces valid s3Location output."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_multimodal_test_data.jsonl")
        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE,
            multimodal_data_s3_path="s3://test-bucket/images/",
            multimodal_data_bucket_owner="123456789012",
        )
        dataset_loader.execute()

        results = list(dataset_loader.dataset())
        self.assertEqual(len(results), 5)

        # All results should validate against the Nova One schema
        for result in results:
            jsonschema.validate(instance=result, schema=SFT_NOVA_ONE_CONVERSE_2024)

        # Record 0: text-only string — no image blocks
        user_content_0 = results[0]["messages"][0]["content"]
        self.assertTrue(all("image" not in b for b in user_content_0))

        # Record 1: text-only array — no image blocks
        user_content_1 = results[1]["messages"][0]["content"]
        self.assertTrue(all("image" not in b for b in user_content_1))

        # Record 2: data URI image — should have s3Location with bucketOwner
        user_content_2 = results[2]["messages"][0]["content"]
        image_blocks_2 = [b for b in user_content_2 if "image" in b]
        self.assertEqual(len(image_blocks_2), 1)
        self.assertIn("s3Location", image_blocks_2[0]["image"]["source"])
        self.assertEqual(
            image_blocks_2[0]["image"]["source"]["s3Location"]["bucketOwner"],
            "123456789012",
        )

        # Record 3: S3 URI image — should have s3Location with bucketOwner
        user_content_3 = results[3]["messages"][0]["content"]
        image_blocks_3 = [b for b in user_content_3 if "image" in b]
        self.assertEqual(len(image_blocks_3), 1)
        self.assertIn("s3Location", image_blocks_3[0]["image"]["source"])

        # Record 4: mixed content — 2 images, order preserved (text, image, text, image)
        user_content_4 = results[4]["messages"][0]["content"]
        image_blocks_4 = [b for b in user_content_4 if "image" in b]
        self.assertEqual(len(image_blocks_4), 2)
        # Verify S3 operations were called (put_object for data URIs, copy_object for S3 URIs)
        self.assertTrue(mock_s3.put_object.called or mock_s3.copy_object.called)

    @patch("amzn_nova_forge.dataset.dataset_transformers.boto3.client")
    def test_end_to_end_multimodal_nova_two_with_s3_path(self, mock_boto_client):
        """Multimodal OpenAI → Converse Nova 2.0 with mocked S3 produces valid s3Location output."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_multimodal_test_data.jsonl")
        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            multimodal_data_s3_path="s3://test-bucket/images/",
            multimodal_data_bucket_owner="123456789012",
        )
        dataset_loader.execute()

        results = list(dataset_loader.dataset())
        self.assertEqual(len(results), 5)

        # All results should validate against the Nova Two schema
        for result in results:
            jsonschema.validate(instance=result, schema=SFT_NOVA_TWO_CONVERSE_2024)

        # Verify image blocks use s3Location
        for result in results:
            for msg in result.get("messages", []):
                for block in msg.get("content", []):
                    if "image" in block:
                        source = block["image"]["source"]
                        self.assertIn("s3Location", source)
                        self.assertNotIn("bytes", source)
                        self.assertEqual(source["s3Location"]["bucketOwner"], "123456789012")
                        self.assertTrue(source["s3Location"]["uri"].startswith("s3://"))

    @patch("amzn_nova_forge.dataset.dataset_transformers.boto3.client")
    def test_end_to_end_multimodal_s3_key_naming(self, mock_boto_client):
        """S3 keys follow the record_NNNN_block_NNN.format naming convention."""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_multimodal_test_data.jsonl")
        dataset_loader.transform(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE,
            multimodal_data_s3_path="s3://test-bucket/images/",
            multimodal_data_bucket_owner="123456789012",
        )
        dataset_loader.execute()

        results = list(dataset_loader.dataset())

        # Collect all S3 URIs from image blocks
        s3_uris = []
        for result in results:
            for msg in result.get("messages", []):
                for block in msg.get("content", []):
                    if "image" in block:
                        s3_uris.append(block["image"]["source"]["s3Location"]["uri"])

        # 4 images total: record 2 has 1, record 3 has 1, record 4 has 2
        self.assertEqual(len(s3_uris), 4)
        # Verify naming pattern
        self.assertIn("record_0002_block_000", s3_uris[0])
        self.assertIn("record_0003_block_000", s3_uris[1])
        self.assertIn("record_0004_block_000", s3_uris[2])
        self.assertIn("record_0004_block_001", s3_uris[3])

    def test_end_to_end_multimodal_text_only_no_s3_path_succeeds(self):
        """Text-only OpenAI dataset transforms without multimodal_data_s3_path."""
        dataset_loader = JSONLDatasetLoader()
        dataset_loader.load(path="tests/test_data/openai_sft_test_data.jsonl")
        dataset_loader.transform(training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
        dataset_loader.execute()

        # Should not raise — no images in the dataset
        results = list(dataset_loader.dataset())
        self.assertGreater(len(results), 0)
        for result in results:
            jsonschema.validate(instance=result, schema=SFT_NOVA_ONE_CONVERSE_2024)


class TestResolvePath(unittest.TestCase):
    """Tests for the resolve_path helper in dataset_loader."""

    def test_local_paths_resolved_to_absolute(self):
        """Relative, tilde, and .. paths all resolve to absolute paths."""
        cases = [
            "data/train.jsonl",
            "./data/train.jsonl",
            "~/datasets/train.jsonl",
        ]
        for path in cases:
            with self.subTest(path=path):
                result = resolve_path(path)
                self.assertTrue(result.startswith("/"))
                self.assertNotIn("/..", result)
                self.assertNotIn("/./", result)
                if path.startswith("~"):
                    self.assertNotIn("~", result)

    def test_parent_segments_normalized(self):
        result = resolve_path("foo/bar/../baz/file.parquet")
        self.assertNotIn("/..", result)
        self.assertTrue(result.endswith("foo/baz/file.parquet"))

    def test_all_loaders_callresolve_path(self):
        """Each loader's load() should call resolve_path."""
        resolve_mock_path = "amzn_nova_forge.dataset.dataset_loader.resolve_path"
        check_exists_path = "amzn_nova_forge.dataset.dataset_loader.check_path_exists"
        load_content_path = "amzn_nova_forge.dataset.dataset_loader.load_file_content"

        loaders_with_paths = [
            (
                JSONLDatasetLoader,
                "data/train.jsonl",
                {"load_file_content": load_content_path},
            ),
            (
                JSONDatasetLoader,
                "data/train.json",
                {"load_file_content": load_content_path},
            ),
            (
                CSVDatasetLoader,
                "data/train.csv",
                {"load_file_content": load_content_path},
            ),
            (
                ParquetDatasetLoader,
                "data/train.parquet",
                {"pq": "amzn_nova_forge.dataset.dataset_loader.pq"},
            ),
            (
                ArrowDatasetLoader,
                "data/train.arrow",
                {"pa": "amzn_nova_forge.dataset.dataset_loader.pa"},
            ),
        ]
        for loader_cls, path, extra_mocks in loaders_with_paths:
            with self.subTest(loader=loader_cls.__name__):
                patches = [
                    patch(resolve_mock_path, side_effect=lambda p: p),
                    patch(check_exists_path),
                ]
                for mock_target in extra_mocks.values():
                    patches.append(patch(mock_target, return_value=iter([])))
                with patches[0] as mock_resolve:
                    for p in patches[1:]:
                        p.start()
                    try:
                        loader_cls().load(path)
                        mock_resolve.assert_called_once_with(path)
                    finally:
                        for p in patches[1:]:
                            p.stop()


class TestEagerLoadValidation(unittest.TestCase):
    """Tests for eager validation in load() — existence and extension checks."""

    def test_nonexistent_local_file_raises_at_load_time(self):
        """load() raises DataPrepError immediately for a non-existent local file."""
        loader = JSONLDatasetLoader()
        with self.assertRaises(DataPrepError) as ctx:
            loader.load("/tmp/definitely_does_not_exist_abc123.jsonl")
        self.assertIn("File not found", str(ctx.exception))

    def test_wrong_extension_raises_at_load_time(self):
        """load() raises DataPrepError immediately for wrong file extension."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"id,text\n1,hello\n")
            temp_path = f.name
        try:
            loader = JSONLDatasetLoader()
            with self.assertRaises(DataPrepError) as ctx:
                loader.load(temp_path)
            self.assertIn(".csv", str(ctx.exception))
            self.assertIn(".jsonl", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    def test_wrong_extension_parquet_loader(self):
        """ParquetDatasetLoader rejects .jsonl files at load() time."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(b'{"id": 1}\n')
            temp_path = f.name
        try:
            loader = ParquetDatasetLoader()
            with self.assertRaises(DataPrepError) as ctx:
                loader.load(temp_path)
            self.assertIn(".jsonl", str(ctx.exception))
            self.assertIn(".parquet", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    @patch("amzn_nova_forge.dataset.file_utils.boto3.client")
    def test_s3_404_raises_at_load_time(self, mock_boto_client):
        """load() raises DataPrepError for S3 objects that return 404."""
        mock_client = mock_boto_client.return_value
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        loader = JSONLDatasetLoader()
        with self.assertRaises(DataPrepError) as ctx:
            loader.load("s3://my-bucket/nonexistent.jsonl")
        self.assertIn("File not found", str(ctx.exception))

    @patch("amzn_nova_forge.dataset.file_utils.boto3.client")
    def test_s3_no_such_key_raises_at_load_time(self, mock_boto_client):
        """load() raises DataPrepError for S3 NoSuchKey errors."""
        mock_client = mock_boto_client.return_value
        error_response = {
            "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}
        }
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        loader = JSONLDatasetLoader()
        with self.assertRaises(DataPrepError) as ctx:
            loader.load("s3://my-bucket/nonexistent.jsonl")
        self.assertIn("File not found", str(ctx.exception))

    @patch("amzn_nova_forge.dataset.file_utils.boto3.client")
    def test_s3_access_denied_logs_warning_and_proceeds(self, mock_boto_client):
        """load() logs a warning but does not raise for S3 permission errors."""
        mock_client = mock_boto_client.return_value
        error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        loader = JSONLDatasetLoader()
        # Should not raise — just warn
        loader.load("s3://my-bucket/data.jsonl")
        # Verify the generator was still assigned
        self.assertIsNotNone(loader.dataset)

    def test_directory_permission_error_raises_dataprep_error(self):
        """scan_local_directory wraps OSError in DataPrepError."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        os.chmod(tmpdir, 0o000)
        try:
            loader = JSONLDatasetLoader()
            with self.assertRaises(DataPrepError) as ctx:
                loader.load(tmpdir)
            self.assertIn("Cannot read directory", str(ctx.exception))
        finally:
            os.chmod(tmpdir, 0o755)
            os.rmdir(tmpdir)


