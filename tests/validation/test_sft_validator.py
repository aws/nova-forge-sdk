import unittest
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.validation.base_validator import Constraints
from amzn_nova_customization_sdk.validation.sft_validator import (
    SFTValidator,
)


class TestSFTValidator(unittest.TestCase):
    def setUp(self):
        """Set up common mocks for AWS services used in validation."""
        # Start patches
        self.boto3_patch = patch(
            "amzn_nova_customization_sdk.validation.base_validator.boto3.client"
        )
        self.sagemaker_role_patch = patch("sagemaker.get_execution_role")
        self.cluster_info_patch = patch(
            "amzn_nova_customization_sdk.validation.base_validator.get_cluster_instance_info"
        )

        # Get mock objects
        self.mock_boto3_client = self.boto3_patch.start()
        self.mock_get_execution_role = self.sagemaker_role_patch.start()
        self.mock_get_cluster_info = self.cluster_info_patch.start()

        # Configure SageMaker execution role mock
        self.mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Configure IAM client mock
        self.mock_iam_client = Mock()
        self.mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }

        # Configure SageMaker client mock
        self.mock_sagemaker_client = Mock()

        # Configure cluster info mock
        self.mock_get_cluster_info.return_value = [
            {
                "instance_group_name": "worker-group",
                "instance_type": "ml.g5.12xlarge",
                "current_count": 4,
                "target_count": 4,
                "status": "InService",
            }
        ]

        # Configure boto3 client factory
        def mock_client_factory(service, **kwargs):
            return {
                "sagemaker": self.mock_sagemaker_client,
                "iam": self.mock_iam_client,
            }[service]

        self.mock_boto3_client.side_effect = mock_client_factory

    def tearDown(self):
        """Clean up patches."""
        self.boto3_patch.stop()
        self.sagemaker_role_patch.stop()
        self.cluster_info_patch.stop()

    def test_get_constraints_valid(self):
        constraints = SFTValidator.get_constraints(
            Platform.SMTJ, Model.NOVA_LITE, TrainingMethod.SFT_LORA
        )
        self.assertIsNotNone(constraints)
        self.assertIsInstance(constraints, Constraints)

    def test_validate_valid_configuration(self):
        # Should not raise any exception
        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_invalid_instance_type(self):
        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.invalid.type",
                instance_count=1,
            )

        self.assertIn("not supported", str(context.exception))
        self.assertIn("ml.invalid.type", str(context.exception))

    def test_validate_invalid_instance_count(self):
        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=999,
            )

        self.assertIn("not supported", str(context.exception))
        self.assertIn("999", str(context.exception))

    def test_validate_max_length_override_valid(self):
        overrides = {"max_length": 4096}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_length_override_too_small(self):
        overrides = {"max_length": 512}  # Below minimum allowed value

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("max_length=512", str(context.exception))
        self.assertIn("outside valid range", str(context.exception))

    def test_validate_max_length_override_too_large(self):
        overrides = {"max_length": 99999}  # Above maximum allowed value

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("max_length=99999", str(context.exception))
        self.assertIn("outside valid range", str(context.exception))

    def test_validate_max_length_override_wrong_type(self):
        overrides = {"max_length": "4096"}  # String instead of int

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects int", str(context.exception))

    def test_validate_global_batch_size_override_valid(self):
        overrides = {"global_batch_size": 32}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_global_batch_size_override_invalid(self):
        overrides = {"global_batch_size": 128}  # Not in allowed options

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("global_batch_size=128", str(context.exception))
        self.assertIn("not valid", str(context.exception))

    def test_validate_global_batch_size_override_wrong_type(self):
        overrides = {"global_batch_size": 32.0}  # Float instead of int

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects int", str(context.exception))

    def test_validate_max_epochs_override_valid(self):
        overrides = {"max_epochs": 10}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_epochs_override_wrong_type(self):
        overrides = {"max_epochs": 10.5}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("max_epochs", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_hidden_dropout_override_valid(self):
        overrides = {"hidden_dropout": 0.5}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_hidden_dropout_override_out_of_range(self):
        overrides = {"hidden_dropout": 1.5}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("hidden_dropout", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_hidden_dropout_override_wrong_type(self):
        overrides = {"hidden_dropout": "0.5"}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects float", str(context.exception))

    def test_validate_attention_dropout_override_valid(self):
        overrides = {"attention_dropout": 0.3}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_attention_dropout_override_out_of_range(self):
        overrides = {"attention_dropout": 1.2}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("attention_dropout", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_attention_dropout_override_wrong_type(self):
        overrides = {"attention_dropout": "invalid"}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects float", str(context.exception))

    def test_validate_ffn_dropout_override_valid(self):
        overrides = {"ffn_dropout": 0.2}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_ffn_dropout_override_out_of_range(self):
        overrides = {"ffn_dropout": -0.1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("ffn_dropout", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_ffn_dropout_override_wrong_type(self):
        overrides = {"ffn_dropout": 0}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects float", str(context.exception))

    def test_validate_lr_override_valid(self):
        overrides = {"lr": 0.001}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_lr_override_wrong_type(self):
        overrides = {"lr": 1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("lr", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_eps_override_valid(self):
        overrides = {"eps": 1e-8}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_eps_override_wrong_type(self):
        overrides = {"eps": 1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("eps", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_weight_decay_override_valid(self):
        overrides = {"weight_decay": 0.01}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_weight_decay_override_out_of_range(self):
        overrides = {"weight_decay": 1.5}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("weight_decay", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_weight_decay_override_wrong_type(self):
        overrides = {"weight_decay": 1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects float", str(context.exception))

    def test_validate_betas_override_valid(self):
        overrides = {"betas": [0.9, 0.999]}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_betas_override_wrong_type(self):
        overrides = {"betas": (0.9, 0.999)}  # Tuple instead of list

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("betas", str(context.exception))
        self.assertIn("expects list", str(context.exception))

    def test_validate_betas_override_wrong_length(self):
        overrides = {"betas": [0.9]}  # Only one value

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("exactly two values", str(context.exception))

    def test_validate_betas_override_out_of_range(self):
        overrides = {"betas": [0.9, 1.5]}  # Second value out of range

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_betas_override_wrong_order(self):
        overrides = {"betas": [0.999, 0.9]}  # First value should be less than second

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("first value", str(context.exception))
        self.assertIn("less than the second", str(context.exception))

    def test_validate_warmup_steps_override_valid(self):
        overrides = {"warmup_steps": 100}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_warmup_steps_override_wrong_type(self):
        overrides = {"warmup_steps": 100.5}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("warmup_steps", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_constant_steps_override_valid(self):
        overrides = {"constant_steps": 50}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_constant_steps_override_wrong_type(self):
        overrides = {"constant_steps": "invalid"}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("constant_steps", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_min_lr_override_valid(self):
        overrides = {"min_lr": 0.0001}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_min_lr_override_wrong_type(self):
        overrides = {"min_lr": 1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("min_lr", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_alpha_override_valid(self):
        overrides = {"alpha": 64}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_alpha_override_invalid_value(self):
        overrides = {"alpha": 100}  # Not in allowed values

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("alpha", str(context.exception))
        self.assertIn("must be one of", str(context.exception))

    def test_validate_alpha_override_wrong_type(self):
        overrides = {"alpha": 64.0}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("alpha", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_adapter_dropout_override_valid(self):
        overrides = {"adapter_dropout": 0.1}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adapter_dropout_override_out_of_range(self):
        overrides = {"adapter_dropout": 1.1}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("adapter_dropout", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_adapter_dropout_override_wrong_type(self):
        overrides = {"adapter_dropout": "invalid"}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("expects float", str(context.exception))

    def test_validate_loraplus_lr_ratio_override_valid(self):
        overrides = {"loraplus_lr_ratio": 16.0}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_loraplus_lr_ratio_override_out_of_range(self):
        overrides = {"loraplus_lr_ratio": 150.0}  # Above maximum of 100.0

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("loraplus_lr_ratio", str(context.exception))
        self.assertIn("between 0.0 and 100.0", str(context.exception))

    def test_validate_loraplus_lr_ratio_override_wrong_type(self):
        overrides = {"loraplus_lr_ratio": 16}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        self.assertIn("loraplus_lr_ratio", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_max_steps_override_valid(self):
        overrides = {"max_steps": 1000}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_steps_override_wrong_type(self):
        overrides = {"max_steps": 1000.5}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )

        self.assertIn("max_steps", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_adam_beta1_override_valid(self):
        overrides = {"adam_beta1": 0.9}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adam_beta1_override_wrong_type(self):
        overrides = {"adam_beta1": 9}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )

        self.assertIn("adam_beta1", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_adam_beta2_override_valid(self):
        overrides = {"adam_beta2": 0.999}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adam_beta2_override_wrong_type(self):
        overrides = {"adam_beta2": 999}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )

        self.assertIn("adam_beta2", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_reasoning_enabled_override_valid(self):
        overrides = {"reasoning_enabled": False}

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_reasoning_enabled_override_invalid(self):
        overrides = {"reasoning_enabled": "invalid"}

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
            )

        self.assertIn("reasoning_enabled", str(context.exception))
        self.assertIn(
            "Override 'reasoning_enabled' expects bool",
            str(context.exception),
        )

    def test_validate_multiple_overrides(self):
        overrides = {
            "max_length": 4096,
            "global_batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.01,
        }

        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_multiple_errors_aggregated(self):
        overrides = {
            "max_length": 99999,  # Out of range
            "global_batch_size": 128,  # Invalid option
            "hidden_dropout": 1.5,  # Out of range
        }

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                overrides=overrides,
            )

        error_msg = str(context.exception)
        # Should contain multiple error messages
        self.assertIn("max_length", error_msg)
        self.assertIn("global_batch_size", error_msg)
        self.assertIn("hidden_dropout", error_msg)


if __name__ == "__main__":
    unittest.main()
