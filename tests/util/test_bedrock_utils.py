"""Unit tests for Bedrock utility functions.

This module contains tests for utility functions in bedrock.py including:
- parse_bedrock_recipe_config
- get_customization_type
- resolve_base_model_identifier
"""

import unittest
from unittest.mock import mock_open, patch

from amzn_nova_forge.model.model_enums import TrainingMethod
from amzn_nova_forge.util.bedrock import (
    get_customization_type,
    parse_bedrock_recipe_config,
    resolve_base_model_identifier,
)


class TestBedrockUtils(unittest.TestCase):
    """Unit tests for Bedrock utility functions."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_parse_sft_recipe_extracts_hyperparameters_as_strings(
        self, mock_yaml_load, mock_file
    ):
        """Test that SFT hyperparameters are extracted as strings.

        Validates: SFT hyperparameters must be strings for Bedrock API
        """
        # Mock SFT recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-sft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            },
            "training_config": {
                "method": "sft_lora",
                "epochCount": 10,
                "batchSize": 128,
                "learningRate": 0.0001,
                "maxPromptLength": 8192,
                "trainer": {  # Nested dict - should be excluded
                    "peft": {"peft_scheme": "lora"}
                },
            },
        }

        result = parse_bedrock_recipe_config(
            "/path/to/recipe.yaml", TrainingMethod.SFT_LORA
        )

        # Verify hyperparameters are extracted
        self.assertIn("hyperparameters", result)
        hyperparameters = result["hyperparameters"]

        # Verify all values are strings
        self.assertIsInstance(hyperparameters["epochCount"], str)
        self.assertEqual(hyperparameters["epochCount"], "10")

        self.assertIsInstance(hyperparameters["batchSize"], str)
        self.assertEqual(hyperparameters["batchSize"], "128")

        self.assertIsInstance(hyperparameters["learningRate"], str)
        self.assertEqual(hyperparameters["learningRate"], "0.0001")

        self.assertIsInstance(hyperparameters["maxPromptLength"], str)
        self.assertEqual(hyperparameters["maxPromptLength"], "8192")

        # Verify nested dicts are excluded
        self.assertNotIn("trainer", hyperparameters)

        # Verify method field is excluded
        self.assertNotIn("method", hyperparameters)

        # Verify RFT hyperparameters are empty for SFT job
        self.assertEqual(result["rft_hyperparameters"], {})

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_parse_rft_recipe_extracts_hyperparameters_as_native_types(
        self, mock_yaml_load, mock_file
    ):
        """Test that RFT hyperparameters are extracted as native types (int, float, str).

        This is critical - Bedrock API requires RFT hyperparameters to be native types,
        NOT strings like SFT hyperparameters.

        Validates: RFT hyperparameters must be native types for Bedrock API
        """
        # Mock RFT recipe config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-rft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            },
            "training_config": {
                "method": "rft_lora",
                "rft": {
                    "epochCount": 2,  # int
                    "batchSize": 64,  # int
                    "learningRate": 0.0001,  # float
                    "maxPromptLength": 8192,  # int
                    "trainingSamplePerPrompt": 8,  # int
                    "inferenceMaxTokens": 2048,  # int
                    "evalInterval": 10,  # int
                    "reasoningEffort": "low",  # str
                    "graderConfig": {  # Nested dict - should be excluded
                        "lambdaArn": "arn:aws:lambda:..."
                    },
                },
            },
        }

        result = parse_bedrock_recipe_config(
            "/path/to/recipe.yaml", TrainingMethod.RFT_LORA
        )

        # Verify RFT hyperparameters are extracted
        self.assertIn("rft_hyperparameters", result)
        rft_hyperparameters = result["rft_hyperparameters"]

        # Verify integer values remain as integers
        self.assertIsInstance(rft_hyperparameters["epochCount"], int)
        self.assertEqual(rft_hyperparameters["epochCount"], 2)

        self.assertIsInstance(rft_hyperparameters["batchSize"], int)
        self.assertEqual(rft_hyperparameters["batchSize"], 64)

        self.assertIsInstance(rft_hyperparameters["maxPromptLength"], int)
        self.assertEqual(rft_hyperparameters["maxPromptLength"], 8192)

        self.assertIsInstance(rft_hyperparameters["trainingSamplePerPrompt"], int)
        self.assertEqual(rft_hyperparameters["trainingSamplePerPrompt"], 8)

        self.assertIsInstance(rft_hyperparameters["inferenceMaxTokens"], int)
        self.assertEqual(rft_hyperparameters["inferenceMaxTokens"], 2048)

        self.assertIsInstance(rft_hyperparameters["evalInterval"], int)
        self.assertEqual(rft_hyperparameters["evalInterval"], 10)

        # Verify float values remain as floats
        self.assertIsInstance(rft_hyperparameters["learningRate"], float)
        self.assertEqual(rft_hyperparameters["learningRate"], 0.0001)

        # Verify string values remain as strings
        self.assertIsInstance(rft_hyperparameters["reasoningEffort"], str)
        self.assertEqual(rft_hyperparameters["reasoningEffort"], "low")

        # Verify graderConfig is excluded
        self.assertNotIn("graderConfig", rft_hyperparameters)

        # Verify SFT hyperparameters are empty for RFT job
        self.assertEqual(result["hyperparameters"], {})

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_parse_recipe_without_training_config(self, mock_yaml_load, mock_file):
        """Test that missing training_config returns empty hyperparameters.

        Validates: Graceful handling of missing training_config
        """
        # Mock recipe without training_config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            }
        }

        result = parse_bedrock_recipe_config(
            "/path/to/recipe.yaml", TrainingMethod.SFT_LORA
        )

        # Verify empty hyperparameters are returned
        self.assertEqual(result["hyperparameters"], {})
        self.assertEqual(result["rft_hyperparameters"], {})

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_parse_rft_recipe_without_rft_config(self, mock_yaml_load, mock_file):
        """Test that RFT job without rft config returns empty RFT hyperparameters.

        Validates: Graceful handling of missing rft config
        """
        # Mock RFT recipe without rft config
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-rft-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            },
            "training_config": {
                "method": "rft_lora",
                # Missing rft config
            },
        }

        result = parse_bedrock_recipe_config(
            "/path/to/recipe.yaml", TrainingMethod.RFT_LORA
        )

        # Verify empty RFT hyperparameters are returned
        self.assertEqual(result["rft_hyperparameters"], {})

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_parse_recipe_returns_full_config(self, mock_yaml_load, mock_file):
        """Test that parse_bedrock_recipe_config returns full recipe config.

        Validates: Full recipe config is returned for reference
        """
        # Mock recipe config
        recipe_config = {
            "run": {
                "name": "test-job",
                "model_type": "amazon.nova-micro-v1:0:128k",
            },
            "training_config": {
                "method": "sft_lora",
                "epochCount": 10,
            },
        }
        mock_yaml_load.return_value = recipe_config

        result = parse_bedrock_recipe_config(
            "/path/to/recipe.yaml", TrainingMethod.SFT_LORA
        )

        # Verify full recipe config is returned
        self.assertIn("recipe_config", result)
        self.assertEqual(result["recipe_config"], recipe_config)

    def test_sft_lora_maps_to_fine_tuning(self):
        """Test that SFT_LORA maps to FINE_TUNING."""
        result = get_customization_type(TrainingMethod.SFT_LORA)
        self.assertEqual(result, "FINE_TUNING")

    def test_rft_lora_maps_to_reinforcement_fine_tuning(self):
        """Test that RFT_LORA maps to REINFORCEMENT_FINE_TUNING."""
        result = get_customization_type(TrainingMethod.RFT_LORA)
        self.assertEqual(result, "REINFORCEMENT_FINE_TUNING")

    def test_unsupported_method_raises_value_error(self):
        """Test that unsupported methods raise ValueError with helpful message."""
        with self.assertRaises(ValueError) as context:
            get_customization_type(TrainingMethod.SFT_FULL)

        error_message = str(context.exception)
        self.assertIn("sft_full", error_message)
        self.assertIn("not supported on Bedrock", error_message)
        self.assertIn("Supported methods:", error_message)

    def test_explicit_identifier_is_returned(self):
        """Test that explicit base_model_identifier is returned without parsing recipe."""
        explicit_id = "arn:aws:bedrock:us-east-1::foundation-model/custom-model:0"
        result = resolve_base_model_identifier("dummy_path.yaml", explicit_id)
        self.assertEqual(result, explicit_id)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_resolve_from_recipe_nova_micro(self, mock_yaml_load, mock_file):
        """Test resolving NOVA_MICRO from recipe."""
        mock_yaml_load.return_value = {
            "run": {
                "model_type": "amazon.nova-micro-v1:0:128k",
            }
        }

        result = resolve_base_model_identifier("/path/to/recipe.yaml")
        self.assertIn("nova-micro", result)
        self.assertIn("foundation-model", result)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_resolve_from_recipe_nova_lite(self, mock_yaml_load, mock_file):
        """Test resolving NOVA_LITE from recipe."""
        mock_yaml_load.return_value = {
            "run": {
                "model_type": "amazon.nova-lite-v1:0:300k",
            }
        }

        result = resolve_base_model_identifier("/path/to/recipe.yaml")
        self.assertIn("nova-lite", result)
        self.assertIn("foundation-model", result)

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_missing_model_type_raises_error(self, mock_yaml_load, mock_file):
        """Test that missing model_type raises ValueError."""
        mock_yaml_load.return_value = {
            "run": {
                "name": "test-job",
                # Missing model_type
            }
        }

        with self.assertRaises(ValueError) as context:
            resolve_base_model_identifier("/path/to/recipe.yaml")

        self.assertIn("model_type", str(context.exception))

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_unsupported_model_type_raises_error(self, mock_yaml_load, mock_file):
        """Test that unsupported model_type raises ValueError."""
        mock_yaml_load.return_value = {
            "run": {
                "model_type": "unsupported-model",
            }
        }

        with self.assertRaises(ValueError) as context:
            resolve_base_model_identifier("/path/to/recipe.yaml")

        error_message = str(context.exception)
        self.assertIn("unsupported-model", error_message)
        self.assertIn("not supported on Bedrock", error_message)


if __name__ == "__main__":
    unittest.main()
