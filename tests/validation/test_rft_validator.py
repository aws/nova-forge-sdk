import unittest
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.validation.base_validator import Constraints
from amzn_nova_customization_sdk.validation.rft_validator import RFTValidator


class TestRFTValidatorSMHP(unittest.TestCase):
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
                "instance_type": "ml.p5.48xlarge",
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
        constraints = RFTValidator.get_constraints(
            Platform.SMTJ, Model.NOVA_LITE_2, TrainingMethod.RFT_LORA
        )
        self.assertIsNotNone(constraints)
        self.assertIsInstance(constraints, Constraints)

    def test_validate_valid_configuration(self):
        # Should not raise any exception
        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_invalid_instance_type(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.invalid.type",
                instance_count=2,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("not supported", str(context.exception))
        self.assertIn("ml.invalid.type", str(context.exception))

    def test_validate_invalid_instance_count(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=999,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("not supported", str(context.exception))
        self.assertIn("999", str(context.exception))

    def test_validate_nova_1_invalid(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE,
                method=TrainingMethod.RFT,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn(
            "RFT is currently only supported on Nova Lite 2.0", str(context.exception)
        )

    def test_validate_missing_lambda_arn(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                rft_lambda_arn=None,
            )

        self.assertIn(
            "You must provide a valid Lambda function ARN for 'rft_lambda_arn' when performing RFT training.",
            str(context.exception),
        )

    def test_validate_invalid_lambda_arn(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                rft_lambda_arn="not a lambda ARN",
            )

        self.assertIn(
            "You must provide a valid Lambda function ARN for 'rft_lambda_arn' when performing RFT training.",
            str(context.exception),
        )

    def test_validate_global_batch_size_override_valid(self):
        overrides = {"global_batch_size": 256}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_global_batch_size_override_wrong_type(self):
        overrides = {"global_batch_size": 32.0}  # Float instead of int

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("global_batch_size", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_lr_override_valid(self):
        overrides = {"lr": 7e-7}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_lr_override_wrong_type(self):
        overrides = {"lr": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("lr", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_weight_decay_override_valid(self):
        overrides = {"weight_decay": 0.01}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_weight_decay_override_out_of_range(self):
        overrides = {"weight_decay": 1.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("weight_decay", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_weight_decay_override_wrong_type(self):
        overrides = {"weight_decay": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("weight_decay", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_alpha_override_valid(self):
        overrides = {"alpha": 32}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_alpha_override_invalid_value(self):
        overrides = {"alpha": 100}  # Not in allowed values

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("alpha", str(context.exception))
        self.assertIn("must be one of", str(context.exception))

    def test_validate_alpha_override_wrong_type(self):
        overrides = {"alpha": 64.0}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("alpha", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_loraplus_lr_ratio_override_valid(self):
        overrides = {"loraplus_lr_ratio": 64.0}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_loraplus_lr_ratio_override_out_of_range(self):
        overrides = {"loraplus_lr_ratio": 150.0}  # Above maximum of 100.0

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("loraplus_lr_ratio", str(context.exception))
        self.assertIn("between 0.0 and 100.0", str(context.exception))

    def test_validate_loraplus_lr_ratio_override_wrong_type(self):
        overrides = {"loraplus_lr_ratio": 16}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("loraplus_lr_ratio", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_generation_replicas_override_valid(self):
        overrides = {"generation_replicas": 2}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_generation_replicas_override_invalid(self):
        overrides = {"generation_replicas": 0}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("generation_replicas", str(context.exception))
        self.assertIn("must be at least 1", str(context.exception))

    def test_validate_generation_replicas_override_wrong_type(self):
        overrides = {"generation_replicas": 4.0}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("generation_replicas", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_rollout_worker_replicas_override_valid(self):
        overrides = {"rollout_worker_replicas": 1}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_rollout_worker_replicas_override_invalid(self):
        overrides = {"rollout_worker_replicas": -1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("rollout_worker_replicas", str(context.exception))
        self.assertIn("must be at least 1", str(context.exception))

    def test_validate_rollout_worker_replicas_override_wrong_type(self):
        overrides = {"rollout_worker_replicas": "1"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("rollout_worker_replicas", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_max_steps_override_valid(self):
        overrides = {"max_steps": 100}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_steps_override_wrong_type(self):
        overrides = {"max_steps": 1000.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("max_steps", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_max_length_override_valid(self):
        overrides = {"max_length": 10240}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_length_override_wrong_type(self):
        overrides = {"max_length": "2048"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("max_length", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_reasoning_effort_override_valid(self):
        overrides = {"reasoning_effort": "low"}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_reasoning_effort_override_wrong_type(self):
        overrides = {"reasoning_effort": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("reasoning_effort", str(context.exception))
        self.assertIn("expects str", str(context.exception))

    def test_validate_shuffle_override_valid_true(self):
        overrides = {"shuffle": True}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_shuffle_override_valid_false(self):
        overrides = {"shuffle": False}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_shuffle_override_invalid(self):
        overrides = {"shuffle": "yes"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn(
            "Override 'shuffle' expects bool",
            str(context.exception),
        )

    def test_validate_type_override_non_overrideable(self):
        overrides = {"type": "not overrideable"}

        with self.assertLogs(logger, level="INFO") as log:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        self.assertTrue(any("'type' is not overrideable" in msg for msg in log.output))

    def test_validate_age_tolerance_override_valid(self):
        overrides = {"age_tolerance": 2}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_age_tolerance_override_wrong_type(self):
        overrides = {"age_tolerance": 5.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("age_tolerance", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_number_generation_override_valid(self):
        overrides = {"number_generation": 16}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_number_generation_override_wrong_type(self):
        overrides = {"number_generation": "10"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("number_generation", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_max_new_tokens_override_valid(self):
        overrides = {"max_new_tokens": 8192}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_new_tokens_override_wrong_type(self):
        overrides = {"max_new_tokens": 512.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("max_new_tokens", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_set_random_seed_override_valid(self):
        overrides = {"set_random_seed": True}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_set_random_seed_override_invalid(self):
        overrides = {"set_random_seed": "invalid"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn(
            "Override 'set_random_seed' expects bool",
            str(context.exception),
        )

    def test_validate_temperature_override_valid(self):
        overrides = {"temperature": 1}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_temperature_override_wrong_type(self):
        overrides = {"temperature": 1.0}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("temperature", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_save_top_k_valid(self):
        overrides = {"save_top_k": 5}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_save_top_k_wrong_type(self):
        overrides = {"save_top_k": 40.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("save_top_k", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_lambda_concurrency_limit_override_valid(self):
        overrides = {"lambda_concurrency_limit": 100}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_lambda_concurrency_limit_override_wrong_type(self):
        overrides = {"lambda_concurrency_limit": "100"}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("lambda_concurrency_limit", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_save_steps_override_valid(self):
        overrides = {"save_steps": 20}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_save_steps_override_wrong_type(self):
        overrides = {"save_steps": 100.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("save_steps", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_clip_ratio_high_override_valid(self):
        overrides = {"clip_ratio_high": 0.2}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_clip_ratio_high_override_wrong_type(self):
        overrides = {"clip_ratio_high": 2}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("clip_ratio_high", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_ent_coeff_override_valid(self):
        overrides = {"ent_coeff": 0.001}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_ent_coeff_override_wrong_type(self):
        overrides = {"ent_coeff": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("ent_coeff", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_loss_scale_override_valid(self):
        overrides = {"loss_scale": 1}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_loss_scale_override_wrong_type(self):
        overrides = {"loss_scale": 1024.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("loss_scale", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_refit_freq_override_valid(self):
        overrides = {"refit_freq": 4}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_refit_freq_override_wrong_type(self):
        overrides = {"refit_freq": 50.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("refit_freq", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_adam_beta1_override_valid(self):
        overrides = {"adam_beta1": 0.9}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adam_beta1_override_wrong_type(self):
        overrides = {"adam_beta1": 9}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("adam_beta1", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_adam_beta2_override_valid(self):
        overrides = {"adam_beta2": 0.999}

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adam_beta2_override_wrong_type(self):
        overrides = {"adam_beta2": 999}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("adam_beta2", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_multiple_overrides(self):
        overrides = {
            "global_batch_size": 512,
            "lr": 0.001,
            "weight_decay": 0.05,
        }

        try:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_multiple_errors_aggregated(self):
        overrides = {
            "shuffle": "shuffle",
            "weight_decay": 1.5,
        }

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        error_msg = str(context.exception)
        self.assertIn("shuffle", error_msg)
        self.assertIn("weight_decay", error_msg)


class TestRFTValidatorSMTJ(unittest.TestCase):
    def test_validate_min_lr_override_valid(self):
        overrides = {"min_lr": 0.0}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_min_lr_override_wrong_type(self):
        overrides = {"min_lr": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("min_lr", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_max_epochs_override_valid(self):
        overrides = {"max_epochs": 5}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_max_epochs_override_wrong_type(self):
        overrides = {"max_epochs": 2.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("max_epochs", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_entropy_coeff_override_valid(self):
        overrides = {"entropy_coeff": 0.01}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_entropy_coeff_override_wrong_type(self):
        overrides = {"entropy_coeff": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("entropy_coeff", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_kl_loss_coef_override_valid(self):
        overrides = {"kl_loss_coef": 0.002}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_kl_loss_coef_override_wrong_type(self):
        overrides = {"kl_loss_coef": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("kl_loss_coef", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_adapter_dropout_override_valid(self):
        overrides = {"adapter_dropout": 0.1}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_adapter_dropout_override_out_of_range(self):
        overrides = {"adapter_dropout": 1.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("adapter_dropout", str(context.exception))
        self.assertIn("between 0.0 and 1.0", str(context.exception))

    def test_validate_adapter_dropout_override_wrong_type(self):
        overrides = {"adapter_dropout": 1}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("adapter_dropout", str(context.exception))
        self.assertIn("expects float", str(context.exception))

    def test_validate_top_k_override_valid(self):
        overrides = {"top_k": 5}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_top_k_override_wrong_type(self):
        overrides = {"top_k": 5.5}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("top_k", str(context.exception))
        self.assertIn("expects int", str(context.exception))

    def test_validate_smtj_global_batch_size_constraints(self):
        overrides = {"global_batch_size": 64}

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_smtj_global_batch_size_invalid(self):
        overrides = {"global_batch_size": 512}

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("global_batch_size", str(context.exception))
        self.assertIn("not valid", str(context.exception))

    def test_validate_smtj_instance_count_valid(self):
        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_smtj_instance_count_invalid(self):
        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=2,  # Not allowed for SMTJ
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        self.assertIn("not supported", str(context.exception))
        self.assertIn("2", str(context.exception))

    def test_validate_smtj_p5en_instance_type_valid(self):
        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5en.48xlarge",
                instance_count=4,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_smtj_multiple_platform_specific_overrides(self):
        overrides = {
            "max_epochs": 3,
            "min_lr": 1e-7,
            "entropy_coeff": 0.005,
            "kl_loss_coef": 0.002,
            "adapter_dropout": 0.1,
            "global_batch_size": 128,
        }

        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    def test_validate_smtj_multiple_errors(self):
        overrides = {
            "max_epochs": 3.5,
            "adapter_dropout": 1.5,
            "min_lr": "invalid",
        }

        with self.assertRaises(ValueError) as context:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )

        error_msg = str(context.exception)
        self.assertIn("max_epochs", error_msg)
        self.assertIn("adapter_dropout", error_msg)
        self.assertIn("min_lr", error_msg)

    def test_validate_optimizer_override_non_overrideable_smtj(self):
        overrides = {"optimizer": "sgd"}

        with self.assertLogs(logger, level="INFO") as log:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                overrides=overrides,
                rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
            )
        self.assertTrue(
            any("'optimizer' is not overrideable" in msg for msg in log.output)
        )


if __name__ == "__main__":
    unittest.main()
