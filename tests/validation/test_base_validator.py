import unittest
from unittest.mock import MagicMock, Mock, create_autospec, patch

from amzn_nova_customization_sdk.manager.runtime_manager import SMTJRuntimeManager
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.model.nova_model_customizer import NovaModelCustomizer
from amzn_nova_customization_sdk.util.recipe import RecipePath
from amzn_nova_customization_sdk.validation.base_validator import (
    BaseValidator,
    Constraints,
    InstanceTypeConstraints,
)
from amzn_nova_customization_sdk.validation.rft_validator import RFTValidator
from amzn_nova_customization_sdk.validation.sft_validator import SFTValidator


class MockRecipePath(RecipePath):
    """Mock RecipePath that acts as a context manager for tests"""

    def __init__(self, path: str):
        super().__init__(path, temp=False)

    def close(self):
        # Override to prevent actual cleanup in tests
        pass


class TestInstanceTypeConstraints(unittest.TestCase):
    def test_instance_type_constraints_initialization(self):
        constraints = InstanceTypeConstraints(
            allowed_counts={1, 2, 4},
            max_length_range=(1024, 8192),
            global_batch_size_options={16, 32, 64},
        )

        self.assertEqual(constraints.allowed_counts, {1, 2, 4})
        self.assertEqual(constraints.max_length_range, (1024, 8192))
        self.assertEqual(constraints.global_batch_size_options, {16, 32, 64})


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.constraints = Constraints(
            instance_type_constraints={
                "ml.g5.12xlarge": InstanceTypeConstraints(
                    allowed_counts={1},
                    max_length_range=(1024, 8192),
                    global_batch_size_options={16, 32, 64},
                ),
                "ml.p5.48xlarge": InstanceTypeConstraints(
                    allowed_counts={2, 4},
                    max_length_range=(1024, 16384),
                    global_batch_size_options={16, 32, 64},
                ),
            }
        )

    def test_get_all_instance_types(self):
        instance_types = self.constraints.get_all_instance_types()
        self.assertEqual(instance_types, {"ml.g5.12xlarge", "ml.p5.48xlarge"})

    def test_get_allowed_counts_for_type_valid(self):
        allowed_counts = self.constraints.get_allowed_counts_for_type("ml.g5.12xlarge")
        self.assertEqual(allowed_counts, {1})

    def test_get_allowed_counts_for_type_invalid(self):
        allowed_counts = self.constraints.get_allowed_counts_for_type("ml.invalid.type")
        self.assertIsNone(allowed_counts)


class TestBaseValidator(unittest.TestCase):
    def setUp(self):
        self.mock_infra = Mock()
        self.mock_infra.cluster_name = "test-cluster"
        self.mock_infra.execution_role = (
            "arn:aws:iam::123456789012:role/TestExecutionRole"
        )

    @patch(
        "amzn_nova_customization_sdk.validation.base_validator.get_cluster_instance_info"
    )
    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    @patch("sagemaker.get_execution_role")
    def test_validate_smhp_infrastructure_success(
        self, mock_get_execution_role, mock_boto3_client, mock_get_cluster_info
    ):
        # Mock successful cluster info
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        # Mock SageMaker execution role
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Mock successful IAM permissions
        mock_sagemaker_client = Mock()
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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

        def mock_client_factory(service, **kwargs):
            return {"sagemaker": mock_sagemaker_client, "iam": mock_iam_client}[service]

        mock_boto3_client.side_effect = mock_client_factory

        try:
            SFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                infra=self.mock_infra,
            )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    @patch(
        "amzn_nova_customization_sdk.validation.base_validator.get_cluster_instance_info"
    )
    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    @patch("sagemaker.get_execution_role")
    def test_validate_smhp_missing_instance_type(
        self, mock_get_execution_role, mock_boto3_client, mock_get_cluster_info
    ):
        # Mock cluster with different instance type
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group-1",
                    "instance_type": "ml.g5.12xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                },
                {
                    "instance_group_name": "worker-group-2",
                    "instance_type": "ml.g5.24xlarge",
                    "current_count": 2,
                    "target_count": 2,
                    "status": "InService",
                },
            ],
        }

        # Mock SageMaker execution role
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )

        # Mock successful IAM permissions and SageMaker cluster access
        mock_sagemaker_client = Mock()
        # TODO: Get rid of this, the above mock is sufficient.
        mock_sagemaker_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
            "InstanceGroups": [
                {
                    "InstanceGroupName": "worker-group-1",
                    "InstanceType": "ml.g5.12xlarge",
                    "CurrentCount": 4,
                    "TargetCount": 4,
                    "Status": "InService",
                },
                {
                    "InstanceGroupName": "worker-group-2",
                    "InstanceType": "ml.g5.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "Status": "InService",
                },
            ],
        }
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "sagemaker": mock_sagemaker_client,
            "iam": mock_iam_client,
        }[service]

        with self.assertRaises(ValueError) as context:
            SFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.p5.48xlarge",
                instance_count=4,  # Use valid count to bypass training validation
                infra=self.mock_infra,
            )

        self.assertIn(
            "Instance type 'ml.p5.48xlarge' not available", str(context.exception)
        )
        self.assertIn(
            "Available types: ['ml.g5.12xlarge', 'ml.g5.24xlarge']",
            str(context.exception),
        )

    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    def test_validate_iam_permissions_directly(self, mock_boto3_client):
        """Test IAM validation method directly."""
        # Mock successful IAM permissions
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
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

        mock_boto3_client.return_value = mock_iam_client

        with patch("sagemaker.get_execution_role") as mock_get_role:
            mock_get_role.return_value = "arn:aws:iam::123456789012:role/ValidRole"

            errors = []
            BaseValidator._validate_iam_permissions(errors, self.mock_infra)

            # Should not add any errors
            self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    def test_validate_iam_permissions_invalid_trust_policy(self, mock_boto3_client):
        """Test IAM validation with invalid trust policy."""
        # Mock IAM role with invalid trust policy
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "ec2.amazonaws.com"
                            },  # Wrong service
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }

        mock_boto3_client.return_value = mock_iam_client

        with patch("sagemaker.get_execution_role") as mock_get_role:
            mock_get_role.return_value = "arn:aws:iam::123456789012:role/InvalidRole"

            errors = []
            BaseValidator._validate_iam_permissions(errors, self.mock_infra)

            # Should add error about trust policy
            self.assertEqual(len(errors), 1)
            self.assertIn("does not trust sagemaker.amazonaws.com service", errors[0])

    @patch(
        "amzn_nova_customization_sdk.validation.base_validator.get_cluster_instance_info"
    )
    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    def test_validate_infrastructure_directly(
        self, mock_boto3_client, mock_get_cluster_info
    ):
        """Test infrastructure validation method directly."""
        # Mock successful cluster info
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 4,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        mock_sagemaker_client = Mock()
        mock_boto3_client.return_value = mock_sagemaker_client

        errors = []
        BaseValidator._validate_infrastructure(
            self.mock_infra, "ml.p5.48xlarge", 2, errors
        )

        # Should not add any errors
        self.assertEqual(len(errors), 0)

    @patch(
        "amzn_nova_customization_sdk.validation.base_validator.get_cluster_instance_info"
    )
    @patch("amzn_nova_customization_sdk.validation.base_validator.boto3.client")
    def test_validate_infrastructure_insufficient_capacity(
        self, mock_boto3_client, mock_get_cluster_info
    ):
        """Test infrastructure validation with insufficient capacity."""
        # Mock cluster with insufficient capacity
        mock_get_cluster_info.return_value = {
            "normal_instance_groups": [],
            "restricted_instance_groups": [
                {
                    "instance_group_name": "worker-group",
                    "instance_type": "ml.p5.48xlarge",
                    "current_count": 1,
                    "target_count": 4,
                    "status": "InService",
                }
            ],
        }

        mock_sagemaker_client = Mock()
        mock_boto3_client.return_value = mock_sagemaker_client

        errors = []
        BaseValidator._validate_infrastructure(
            self.mock_infra, "ml.p5.48xlarge", 2, errors
        )

        # Should add error about insufficient capacity
        self.assertEqual(len(errors), 1)
        self.assertIn("Insufficient capacity", errors[0])
        self.assertIn("Required: 2, Maximum available: 1", errors[0])


class TestValidationConfig(unittest.TestCase):
    """Test validation configuration functionality."""

    def test_get_validation_config_defaults(self):
        """Test that default validation config is correct."""
        config = BaseValidator._get_default_validation_config()
        expected = {"iam": True, "infra": True}
        self.assertEqual(config, expected)

    def test_validation_config_iam_disabled(self):
        """Test that IAM validation is skipped when disabled."""
        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                validation_config={"iam": False, "infra": False},
            )
        except ValueError as e:
            # Should not fail due to IAM validation
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))

    def test_validation_config_infra_disabled(self):
        """Test that infrastructure validation is skipped when disabled."""
        # Create a mock infra object that would normally cause validation errors
        mock_infra = type(
            "MockInfra",
            (),
            {"cluster_name": "non-existent-cluster", "namespace": "test"},
        )()

        try:
            SFTValidator.validate(
                platform=Platform.SMHP,
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                infra=mock_infra,
                validation_config={"iam": False, "infra": False},
            )
        except ValueError as e:
            # Should not fail due to infrastructure validation
            self.assertNotIn("not available in cluster", str(e))
            self.assertNotIn("Failed to validate cluster", str(e))

    def test_validation_config_partial_merge_with_defaults(self):
        """Test that partial validation config is merged with defaults."""
        try:
            SFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                instance_type="ml.g5.12xlarge",
                instance_count=1,
                validation_config={
                    "iam": False
                },  # infra not specified, should default to True
            )
        except ValueError as e:
            # Should not fail due to IAM validation (disabled)
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))


class TestRFTValidatorIntegration(unittest.TestCase):
    """Test RFT validator integration with BaseValidator."""

    def test_rft_validator_inheritance(self):
        """Test that RFT validator properly inherits from BaseValidator."""
        self.assertTrue(issubclass(RFTValidator, BaseValidator))

    def test_rft_validator_with_lambda_arn(self):
        """Test RFT validator with required lambda ARN parameter."""
        try:
            RFTValidator.validate(
                platform=Platform.SMTJ,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.RFT,
                instance_type="ml.p5.48xlarge",
                instance_count=4,
                rft_lambda_arn="arn:aws:lambda:us-west-2:123456789012:function:test-function",
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("RFT validate() raised ValueError unexpectedly!")


class TestE2EValidation(unittest.TestCase):
    """Test E2E validation flow through NovaModelCustomizer."""

    def setUp(self):
        self.data_s3_path = "s3://test-bucket/data"

    @patch("boto3.client")
    @patch("amzn_nova_customization_sdk.model.nova_model_customizer.SFTRecipeBuilder")
    def test_validation_config_propagation_to_recipe_builder(
        self, mock_recipe_builder_class, mock_boto3_client
    ):
        """Test that validation_config is properly propagated from NovaModelCustomizer to SFTRecipeBuilder."""
        # Setup AWS mocks
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}  # Bucket exists

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_boto3_client.side_effect = client_side_effect

        # Setup recipe builder mock
        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build.return_value = MockRecipePath("mock_recipe")
        mock_recipe_builder_class.return_value = mock_recipe_builder

        # Mock infra execute method with proper attributes
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.execute.return_value = "test-job-id"
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.instance_count = 1

        # Create customizer with validation config
        validation_config = {"iam": False, "infra": True}
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
        )
        customizer.validation_config = validation_config

        # Call train method
        customizer.train(job_name="test-job")

        # Verify SFTRecipeBuilder was called with validation_config
        mock_recipe_builder.build.assert_called_once()
        call_kwargs = mock_recipe_builder.build.call_args[1]
        self.assertEqual(call_kwargs["validation_config"], validation_config)

    @patch("boto3.client")
    @patch("amzn_nova_customization_sdk.model.nova_model_customizer.SFTRecipeBuilder")
    def test_validation_config_none_propagation(
        self, mock_recipe_builder_class, mock_boto3_client
    ):
        """Test that None validation_config is properly propagated."""
        # Setup AWS mocks
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}  # Bucket exists

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_boto3_client.side_effect = client_side_effect

        # Setup recipe builder mock
        mock_recipe_builder = MagicMock()
        mock_recipe_builder.build.return_value = MockRecipePath("mock_recipe")
        mock_recipe_builder_class.return_value = mock_recipe_builder

        # Mock infra execute method with proper attributes
        mock_infra = create_autospec(SMTJRuntimeManager)
        mock_infra.execute.return_value = "test-job-id"
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.instance_count = 1

        # Create customizer without validation config (should be None)
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
        )
        # validation_config not specified, should be passed as None to indicate default value

        # Call train method
        customizer.train(job_name="test-job")

        # Verify SFTRecipeBuilder was called with None validation_config
        mock_recipe_builder.build.assert_called_once()
        call_kwargs = mock_recipe_builder.build.call_args[1]
        self.assertIsNone(call_kwargs["validation_config"])


if __name__ == "__main__":
    unittest.main()
