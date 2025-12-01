"""
Unit tests for eval_constraints module.
"""

import unittest
from unittest.mock import patch

from amzn_nova_customization_sdk.model.model_enums import Model
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask
from amzn_nova_customization_sdk.validation.evaluation_validator import (
    EVAL_CONSTRAINTS,
    EvaluationValidator,
)


class TestEvaluationValidator(unittest.TestCase):
    """Test the EvaluationValidator class."""

    def test_get_available_subtasks_nonexistent_task(self):
        """Test get_available_subtasks for tasks without subtasks."""
        # Tasks not in EVAL_AVAILABLE_SUBTASKS should return empty list
        subtasks = EvaluationValidator.get_available_subtasks(EvaluationTask.GPQA)
        self.assertEqual(subtasks, [])

        subtasks = EvaluationValidator.get_available_subtasks(EvaluationTask.IFEVAL)
        self.assertEqual(subtasks, [])

    def test_get_constraints_existing_model(self):
        """Test get_constraints for supported models."""
        for model in [Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_PRO]:
            constraints = EvaluationValidator.get_constraints(model)
            self.assertIsNotNone(constraints)
            self.assertEqual(constraints, EVAL_CONSTRAINTS[model])

    def test_get_constraints_nonexistent_model(self):
        """Test get_constraints for unsupported models."""
        # Create a mock model that doesn't exist in EVAL_CONSTRAINTS
        with patch(
            "amzn_nova_customization_sdk.validation.evaluation_validator.EVAL_CONSTRAINTS",
            {},
        ):
            constraints = EvaluationValidator.get_constraints(Model.NOVA_MICRO)
            self.assertIsNone(constraints)

    def test_validate_byod_valid(self):
        """Test validation for valid BYOD configuration."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.GEN_QA,
                data_s3_path="s3://bucket/path",
                model=Model.NOVA_MICRO,
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_byod_invalid_task(self):
        """Test validation for invalid BYOD task."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="s3://bucket/path",
                model=Model.NOVA_MICRO,
            )
        self.assertIn(
            "BYOD evaluation must use following eval task:", str(context.exception)
        )

    def test_validate_subtask_valid(self):
        """Test validation for valid subtasks."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                subtask="abstract_algebra",
            )

            EvaluationValidator.validate(
                eval_task=EvaluationTask.BBH,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                subtask="boolean_expressions",
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_subtask_invalid_for_task(self):
        """Test validation for invalid subtasks."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                subtask="invalid_subtask",
            )

        self.assertIn("Invalid subtask", str(context.exception))

    def test_validate_subtask_not_supported(self):
        """Test validation for tasks that don't support subtasks."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.GPQA,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                subtask="some_subtask",
            )
        self.assertIn("Task gpqa does not support subtasks", str(context.exception))

    def test_validate_instance_constraints_valid(self):
        """Test validation for valid instance constraints."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_instance_constraints_invalid_type(self):
        """Test validation for invalid instance type."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                instance_type="ml.invalid.type",
                instance_count=1,
            )
        self.assertIn(
            "Instance type 'ml.invalid.type' is not supported", str(context.exception)
        )

    def test_validate_instance_constraints_disallowed_type(self):
        """Test validation for valid instance type which is not allowed for this model."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_LITE,
                instance_type="ml.g5.48xlarge",
                instance_count=1,
            )
        self.assertIn(
            "Instance type 'ml.g5.48xlarge' is not supported", str(context.exception)
        )

    def test_validate_instance_constraints_invalid_count(self):
        """Test validation for invalid instance count."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                instance_type="ml.g5.12xlarge",
                instance_count=200,
            )
        self.assertIn("Instance count 200 is not supported", str(context.exception))

    def test_validate_overrides_full_valid(self):
        """Test validation with valid overrides."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                    "top_logprobs": 10,
                },
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_overrides_partial_valid(self):
        """Test validation with valid overrides."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={"max_new_tokens": 2048},
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.evaluation_validator.logger")
    def test_validate_overrides_with_unknown_field_valid(self, mock_logger):
        """Test validation with valid overrides."""
        # Should not raise any exception
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={"max_new_tokens": 2048, "unknown_field": 10},
            )

            mock_logger.info.assert_called_with(
                "Unknown field 'unknown_field' in overrides, will be ignored later"
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_overrides_type_errors(self):
        """Test validation with type errors in overrides."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={"max_new_tokens": "invalid_string"},
            )
        self.assertIn(
            "Field max_new_tokens expects int, got str", str(context.exception)
        )

    def test_validate_overrides_top_p_range_errors(self):
        """Test validation with top_p range errors in overrides."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={"top_p": 1.5},
            )
        self.assertIn(
            "Field top_p must be a float between 0.0 and 1.0, got 1.5",
            str(context.exception),
        )

    def test_validate_overrides_temperature_range_errors(self):
        """Test validation with temperature range errors in overrides."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={"temperature": -0.1},
            )
        self.assertIn(
            "Field temperature must be a positive float, got -0.1",
            str(context.exception),
        )

    def test_validate_combined_errors(self):
        """Test validation with multiple errors."""
        # This should test the error accumulation logic
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                subtask="invalid_subtask",
                model=Model.NOVA_MICRO,
                instance_type="ml.invalid.type",
                instance_count=1,
            )

        error_message = str(context.exception)
        self.assertIn("Invalid subtask", error_message)
        self.assertIn("Instance type", error_message)

    def test_validate_overrides_multiple_errors(self):
        """Test validation with multiple override errors."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                overrides={
                    "max_new_tokens": "invalid",
                    "top_p": 2.0,
                    "temperature": -1.0,
                },
            )
        error_message = str(context.exception)
        self.assertIn("Field max_new_tokens expects int, got str", error_message)
        self.assertIn(
            "Field top_p must be a float between 0.0 and 1.0, got 2.0", error_message
        )
        self.assertIn(
            "Field temperature must be a positive float, got -1.0", error_message
        )

    def test_validate_empty_data_path(self):
        """Test validation with empty data_s3_path."""
        # Should not trigger BYOD validation
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU, data_s3_path="", model=Model.NOVA_MICRO
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_processor_config_valid(self):
        """Test validation with valid processor_config."""
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.GEN_QA,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                processor_config={
                    "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test"
                },
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_processor_config_missing_lambda_arn(self):
        """Test validation with processor_config missing lambda_arn."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_MICRO,
                processor_config={"aggregation": "average"},
            )
        self.assertIn(
            "processor_config must contain a lambda_arn", str(context.exception)
        )

    def test_validate_rl_env_config_v1_valid(self):
        """Test validation with valid rl_env_config for v1 model."""
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_LITE,
                rl_env_config={
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:reward"
                },
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_rl_env_config_v1_missing_reward_lambda_arn(self):
        """Test validation with rl_env_config missing reward_lambda_arn for v1 model."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_LITE,
                rl_env_config={"mode": "single_turn"},
            )
        self.assertIn(
            "rl_env_config must contain a reward_lambda_arn for model version=Version.ONE",
            str(context.exception),
        )

    def test_validate_rl_env_config_v2_valid(self):
        """Test validation with valid rl_env_config for v2 model."""
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_LITE_2,
                rl_env_config={
                    "single_turn": {
                        "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:single_turn"
                    }
                },
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_rl_env_config_v2_missing_lambda_arn(self):
        """Test validation with rl_env_config missing lambda_arn in single_turn for v2 model."""
        with self.assertRaises(ValueError) as context:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.MMLU,
                data_s3_path="",
                model=Model.NOVA_LITE_2,
                rl_env_config={"single_turn": {"mode": "single_turn"}},
            )
        self.assertIn(
            "rl_env_config.single_turn must contain a lambda_arn for model version=Version.TWO",
            str(context.exception),
        )

    def test_validate_combined_processor_and_rl_env_configs(self):
        """Test validation with both processor_config and rl_env_config."""
        try:
            EvaluationValidator.validate(
                eval_task=EvaluationTask.GEN_QA,
                data_s3_path="",
                model=Model.NOVA_LITE_2,
                processor_config={
                    "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test"
                },
                rl_env_config={
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:reward"
                },
            )
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")


if __name__ == "__main__":
    unittest.main()
