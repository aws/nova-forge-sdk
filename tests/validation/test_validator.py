import unittest
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_customization_sdk.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_customization_sdk.validation.validator import (
    CLUSTER_NAME_REGEX,
    JOB_NAME_REGEX,
    NAMESPACE_REGEX,
    Validator,
)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.mock_smhp_infra = Mock(spec=SMHPRuntimeManager)
        self.mock_smhp_infra.cluster_name = "test-cluster"
        self.mock_smhp_infra.instance_type = "ml.p5.48xlarge"
        self.mock_smhp_infra.instance_count = 4
        self.mock_smhp_infra.region = "us-east-1"

        self.mock_smtj_infra = Mock(spec=SMTJRuntimeManager)
        self.mock_smtj_infra.execution_role = (
            "arn:aws:iam::123456789012:role/TestExecutionRole"
        )
        self.mock_smtj_infra.instance_type = "ml.p5.48xlarge"
        self.mock_smtj_infra.region = "us-east-1"

    @patch("amzn_nova_customization_sdk.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    @patch("sagemaker.get_execution_role")
    @patch(
        "amzn_nova_customization_sdk.manager.runtime_manager.SMHPRuntimeManager.required_calling_role_permissions"
    )
    def test_validate_smhp_infrastructure_success(
        self,
        mock_required_permissions,
        mock_get_execution_role,
        mock_boto3_client,
        mock_get_cluster_info,
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

        # Mock STS client for calling role permissions
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service, **kwargs):
            return {
                "sagemaker": mock_sagemaker_client,
                "iam": mock_iam_client,
                "sts": mock_sts_client,
            }[service]

        mock_boto3_client.side_effect = mock_client_factory

        try:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMHPRuntimeManager.setup"
            ):
                smhp_infra = SMHPRuntimeManager(
                    "ml.p5.48xlarge", 4, "test-cluster", "kubeflow"
                )

                Validator.validate(
                    platform=Platform.SMHP,
                    method=TrainingMethod.SFT_LORA,
                    infra=smhp_infra,
                    recipe={},
                    overrides_template={},
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
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
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMHPRuntimeManager.setup"
            ):
                smhp_infra = SMHPRuntimeManager(
                    "ml.p5.48xlarge", 4, "test-cluster", "kubeflow"
                )

                Validator.validate(
                    platform=Platform.SMHP,
                    method=TrainingMethod.SFT_LORA,
                    infra=smhp_infra,
                    recipe={},
                    overrides_template={},
                )

        self.assertIn(
            "Instance type 'ml.p5.48xlarge' not available", str(context.exception)
        )
        self.assertIn(
            "Available types: ['ml.g5.12xlarge', 'ml.g5.24xlarge']",
            str(context.exception),
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_iam_permissions_directly(self, mock_boto3_client):
        """Test IAM validation method directly."""
        # Mock IAM client
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
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::test-bucket",
                            "arn:aws:s3:::test-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        # Mock STS client to return same account (making it same-account role)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "iam":
                return mock_iam_client
            elif service_name == "sts":
                return mock_sts_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch(
                    "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.setup"
                ):
                    mock_get_role.return_value = (
                        "arn:aws:iam::123456789012:role/ValidRole"
                    )

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge", 1, "arn:aws:iam::123456789012:role/ValidRole"
                    )
                    smtj_infra.execution_role = (
                        "arn:aws:iam::123456789012:role/ValidRole"
                    )

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should not add any errors
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_iam_permissions_invalid_trust_policy(self, mock_boto3_client):
        """Test IAM validation with invalid trust policy."""
        # Mock IAM client with invalid trust policy
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
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": []}
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        # Mock STS client to return same account (making it same-account role)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "iam":
                return mock_iam_client
            elif service_name == "sts":
                return mock_sts_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch(
                    "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.setup"
                ):
                    mock_get_role.return_value = (
                        "arn:aws:iam::123456789012:role/InvalidRole"
                    )

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::123456789012:role/InvalidRole",
                    )
                    smtj_infra.execution_role = (
                        "arn:aws:iam::123456789012:role/InvalidRole"
                    )

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should add error about trust policy AND missing permissions
                self.assertEqual(len(errors), 2)
            # Check that both trust policy and permissions errors are present
            error_text = " ".join(errors)
            self.assertIn("does not trust sagemaker.amazonaws.com service", error_text)
            self.assertIn("missing required permissions", error_text)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_success(self, mock_boto3_client):
        """Test IAM validation for cross-account role with successful assumption."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock successful assume role
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test-key",
                "SecretAccessKey": "test-secret",
                "SessionToken": "test-token",
            }
        }

        # Mock IAM client with assumed credentials
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
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::cross-account-bucket",
                            "arn:aws:s3:::cross-account-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch(
                    "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.setup"
                ):
                    mock_get_role.return_value = (
                        "arn:aws:iam::222222222222:role/CrossAccountRole"
                    )

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::222222222222:role/CrossAccountRole",
                    )
                    smtj_infra.execution_role = (
                        "arn:aws:iam::222222222222:role/CrossAccountRole"
                    )

                errors = []
                Validator._validate_iam_permissions(
                    errors,
                    smtj_infra,
                    data_s3_path="s3://test-bucket/data.jsonl",
                    output_s3_path="s3://test-bucket/output/",
                )

                # Should not add any errors for successful cross-account validation
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_assume_fail(
        self, mock_boto3_client
    ):
        """Test IAM validation for cross-account role when assume role fails."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock failed assume role
        mock_sts_client.assume_role.side_effect = Exception("AccessDenied")

        # Mock IAM client for calling role permissions
        mock_iam_client = Mock()
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch(
                    "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.setup"
                ):
                    cross_account_role = (
                        "arn:aws:iam::222222222222:role/CrossAccountRole"
                    )
                    mock_get_role.return_value = cross_account_role

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge", 1, self.mock_smtj_infra.execution_role
                    )
                    smtj_infra.execution_role = cross_account_role

                    errors = []
                    Validator._validate_iam_permissions(
                        errors,
                        smtj_infra,
                        data_s3_path="s3://test-bucket/data.jsonl",
                        output_s3_path="s3://test-bucket/output/",
                    )

                    # Should not add any errors - silently skip validation for cross-account
                    self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_iam_permissions_cross_account_limited_permissions(
        self, mock_boto3_client
    ):
        """Test IAM validation for cross-account role with limited IAM permissions."""
        # Mock STS client for cross-account detection (different accounts)
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Account": "111111111111",
            "Arn": "arn:aws:sts::111111111111:assumed-role/TestRole/session",
        }

        # Mock successful assume role
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test-key",
                "SecretAccessKey": "test-secret",
                "SessionToken": "test-token",
            }
        }

        # Mock IAM client that can get role but not policies
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
        # Fail on policy operations
        mock_iam_client.list_role_policies.side_effect = Exception("AccessDenied")

        # Mock IAM simulation for calling role permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        def mock_client_factory(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "iam":
                return mock_iam_client
            return Mock()

        mock_boto3_client.side_effect = mock_client_factory

        with patch("sagemaker.get_execution_role") as mock_get_role:
            with patch(
                "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.required_calling_role_permissions",
                return_value=[],
            ):
                with patch(
                    "amzn_nova_customization_sdk.manager.runtime_manager.SMTJRuntimeManager.setup"
                ):
                    mock_get_role.return_value = (
                        "arn:aws:iam::222222222222:role/CrossAccountRole"
                    )

                    smtj_infra = SMTJRuntimeManager(
                        "ml.p5.48xlarge",
                        1,
                        "arn:aws:iam::222222222222:role/CrossAccountRole",
                    )
                    smtj_infra.execution_role = (
                        "arn:aws:iam::222222222222:role/CrossAccountRole"
                    )

                    errors = []
                    Validator._validate_iam_permissions(
                        errors,
                        smtj_infra,
                        data_s3_path="s3://test-bucket/data.jsonl",
                        output_s3_path="s3://test-bucket/output/",
                    )

                # Should not add any errors - validates trust policy only
                self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
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
        Validator._validate_infrastructure(self.mock_smhp_infra, errors)

        # Should not add any errors
        self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.get_cluster_instance_info")
    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
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

        # Update mock to require more instances than available
        self.mock_smhp_infra.instance_count = 2

        errors = []
        Validator._validate_infrastructure(self.mock_smhp_infra, errors)

        # Should add error about insufficient capacity
        self.assertEqual(len(errors), 1)
        self.assertIn("Insufficient capacity", errors[0])
        self.assertIn("Required: 2, Maximum available: 1", errors[0])


class TestValidationConfig(unittest.TestCase):
    """Test validation configuration functionality."""

    def test_get_validation_config_defaults(self):
        """Test that default validation config is correct."""
        config = Validator._get_default_validation_config()
        expected = {"iam": True, "infra": True}
        self.assertEqual(config, expected)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validation_config_iam_disabled(self, mock_boto3_client):
        """Test that IAM validation is skipped when disabled."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                infra=mock_infra,
                method=TrainingMethod.SFT_LORA,
                recipe={},
                overrides_template={},
                validation_config={"iam": False, "infra": False},
            )
        except ValueError as e:
            # Should not fail due to IAM validation
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))

    def test_validation_config_infra_disabled(self):
        """Test that infrastructure validation is skipped when disabled."""
        # Create a mock infra object that would normally cause validation errors
        mock_infra = Mock(spec=SMHPRuntimeManager)
        mock_infra.cluster_name = "non-existent-cluster"
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMHP,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config={"iam": False, "infra": False},
            )
        except ValueError as e:
            # Should not fail due to infrastructure validation
            self.assertNotIn("not available in cluster", str(e))
            self.assertNotIn("Failed to validate cluster", str(e))

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validation_config_partial_merge_with_defaults(self, mock_boto3_client):
        """Test that partial validation config is merged with defaults."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.g5.12xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config={
                    "iam": False
                },  # infra not specified, should default to True
            )
        except ValueError as e:
            # Should not fail due to IAM validation (disabled)
            self.assertNotIn("Failed to validate IAM permissions", str(e))
            self.assertNotIn("Failed to get SageMaker execution role", str(e))


class TestRFTValidation(unittest.TestCase):
    """Test RFT validator integration with Validator."""

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_rft_validator_with_lambda_arn(self, mock_boto3_client):
        """Test RFT validator with required lambda ARN parameter."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                rft_lambda_arn="arn:aws:lambda:us-west-2:123456789012:function:test-function",
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("RFT validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_rft_validator_with_invalid_lambda_arn(self, mock_boto3_client):
        """Test RFT validator fails with invalid lambda ARN."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config={"iam": False, "infra": False},
                rft_lambda_arn="not a lambda arn",
            )

        self.assertIn("must be a valid Lambda function ARN", str(context.exception))

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_rft_validator_without_lambda_arn(self, mock_boto3_client):
        """Test RFT validator fails without lambda ARN."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.RFT_LORA,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn(
            "'rft_lambda_arn' is a required parameter", str(context.exception)
        )


class TestEvaluationValidation(unittest.TestCase):
    """Test evaluation-specific validation."""

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_evaluation_with_valid_task(self, mock_boto3_client):
        """Test evaluation with valid task."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_evaluation_with_valid_byod_task(self, mock_boto3_client):
        """Test evaluation with valid task."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                data_s3_path="data_s3_path",
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_invalid_byod_task(self, mock_boto3_client):
        """Test eval validator fails with invalid BYOD eval task"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                data_s3_path="data_s3_path",
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn(
            "BYOD evaluation must use one of the following eval tasks",
            str(context.exception),
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_evaluation_with_valid_subtask(self, mock_boto3_client):
        """Test evaluation with valid subtask."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                subtask="abstract_algebra",
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_invalid_subtask(self, mock_boto3_client):
        """Test eval validator fails with invalid subtask"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                subtask="invalid",
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn("Invalid subtask", str(context.exception))

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_unsupported_subtask(self, mock_boto3_client):
        """Test eval validator fails with unsupported subtask"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                subtask="abstract_algebra",
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn("does not support subtasks", str(context.exception))

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_evaluation_with_valid_processor_config(self, mock_boto3_client):
        """Test evaluation with valid processor_config."""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                processor_config={
                    "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:my-function"
                },
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_processor_config_but_not_needed(
        self, mock_boto3_client
    ):
        """Test eval validator fails when processor config is provided but not needed"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                processor_config={"lambda_arn": "my lambda"},
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn(
            "processor_config is only supported for gen_qa task", str(context.exception)
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_processor_config_missing_lambda_arn(
        self, mock_boto3_client
    ):
        """Test eval validator fails when processor config is provided but is missing lambda_arn"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.GEN_QA,
                processor_config={"lambda_task": "lambda_task"},
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn(
            "processor_config must contain a lambda_arn", str(context.exception)
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_rl_env_config_missing_reward_lambda_arn(
        self, mock_boto3_client
    ):
        """Test eval validator fails when rl_env_config is provided but is missing reward_lambda_arn"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.RFT_EVAL,
                rl_env_config={"something": "something"},
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn("rl_env must contain a reward_lambda_arn", str(context.exception))

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_rl_env_config_invalid_task(self, mock_boto3_client):
        """Test eval validator fails when rl_env_config is provided but for invalid task"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.MMLU,
                rl_env_config={"reward_lambda_arn": "reward_lambda_arn"},
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn(
            "rl_env_config is only supported for rft_eval task", str(context.exception)
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_eval_validator_with_valid_rl_env_config(self, mock_boto3_client):
        """Test eval validator succeeds when rl_env_config is provided"""
        mock_infra = Mock(spec=SMTJRuntimeManager)
        mock_infra.instance_type = "ml.p5.48xlarge"
        mock_infra.region = "us-east-1"

        try:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                recipe={},
                overrides_template={},
                eval_task=EvaluationTask.RFT_EVAL,
                rl_env_config={
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:my-function"
                },
                validation_config={"iam": False, "infra": False},
            )
        except ValueError:
            self.fail("Evaluation validate() raised ValueError unexpectedly!")


class TestRecipeValidation(unittest.TestCase):
    """Test cases for recipe validation logic in Validator._validate_recipe method"""

    def setUp(self):
        self.mock_infra = Mock(spec=SMTJRuntimeManager)
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.region = "us-east-1"

    def test_validate_recipe_namespace_key_is_skipped(self):
        """Test that 'namespace' key is skipped during validation"""
        recipe = {}
        overrides_template = {
            "namespace": {"type": "string", "required": True, "default": "kubeflow"}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_instance_type_valid(self):
        """Test instance_type validation with valid value"""
        recipe = {}

        overrides_template = {
            "instance_type": {"enum": ["ml.g5.48xlarge", "ml.p5.48xlarge"]}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_instance_type_invalid(self):
        """Test instance_type validation with invalid value"""
        recipe = {}

        overrides_template = {
            "instance_type": {"enum": ["ml.g5.48xlarge", "ml.p5.48xlarge"]}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.g5.12xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Instance type 'ml.g5.12xlarge' is not supported", errors[0])
        self.assertIn("ml.g5.48xlarge", errors[0])
        self.assertIn("ml.p5.48xlarge", errors[0])

    def test_validate_recipe_instance_type_without_enum(self):
        """Test instance_type validation when no instance types are specified"""
        recipe = {}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template={},
            instance_type="ml.g5.12xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_required_validation_missing(self):
        """Test type validation fails with missing required field"""
        recipe = {}
        overrides_template = {"max_steps": {"required": True}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn(
            "'max_steps' is required, but was not found in your recipe", errors[0]
        )

    def test_validate_recipe_type_validation_string(self):
        """Test type validation for string type"""
        recipe = {"run": {"name": "my-training-run"}}
        overrides_template = {"name": {"type": "string"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_integer(self):
        """Test type validation for integer type"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_boolean(self):
        """Test type validation for boolean type"""
        recipe = {"training_config": {"reasoning_enabled": True}}
        overrides_template = {"reasoning_enabled": {"type": "boolean"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_wrong_type(self):
        """Test type validation fails with wrong type"""
        recipe = {"training_config": {"max_steps": "100"}}
        overrides_template = {"max_steps": {"type": "integer"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' expects integer", errors[0])
        self.assertIn("You provided str", errors[0])

    def test_validate_recipe_unknown_type_in_metadata(self):
        """Test validation handles unknown type in metadata gracefully"""
        recipe = {"run": {"custom_field": "value"}}
        overrides_template = {"custom_field": {"type": "unknown_type"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_enum_validation_valid(self):
        """Test enum validation with valid value"""
        recipe = {"run": {"replicas": 4}}
        overrides_template = {"replicas": {"enum": [4, 8, 16, 32]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_enum_validation_invalid(self):
        """Test enum validation with invalid value"""
        recipe = {"run": {"replicas": 2}}
        overrides_template = {"replicas": {"type": "integer", "enum": [4, 8, 16, 32]}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'replicas' must be one of [4, 8, 16, 32]", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_min_validation_valid(self):
        """Test minimum value validation with valid value"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_min_validation_invalid(self):
        """Test minimum value validation with invalid value"""
        recipe = {"training_config": {"max_steps": 2}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' must be at least 4", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_min_validation_exact_boundary(self):
        """Test minimum value validation at exact boundary"""
        recipe = {"training_config": {"max_steps": 4}}
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_max_validation_valid(self):
        """Test maximum value validation with valid value"""
        recipe = {"training_config": {"max_steps": 100}}
        overrides_template = {"max_steps": {"type": "integer", "max": 500}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_max_validation_invalid(self):
        """Test maximum value validation with invalid value"""
        recipe = {"training_config": {"max_steps": 2}}
        overrides_template = {"max_steps": {"type": "integer", "max": 1}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("'max_steps' must be no greater than 1", errors[0])
        self.assertIn("You provided 2", errors[0])

    def test_validate_recipe_max_validation_exact_boundary(self):
        """Test maximum value validation at exact boundary"""
        recipe = {"training_config": {"max_steps": 4}}
        overrides_template = {"max_steps": {"type": "integer", "max": 4}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_multiple_constraints_all_valid(self):
        """Test recipe with multiple constraints that are all valid"""
        recipe = {
            "run": {"replicas": 8, "name": "my-run"},
            "training_config": {"max_steps": 100, "reasoning_enabled": True},
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "name": {"type": "string"},
            "max_steps": {"type": "integer", "min": 4, "max": 100000},
            "reasoning_enabled": {"type": "boolean"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_multiple_constraints_multiple_violations(self):
        """Test recipe with multiple constraint violations"""
        recipe = {
            "run": {"replicas": 3, "name": 123},
            "training_config": {"max_steps": 2},
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "name": {"type": "string"},
            "max_steps": {"type": "integer", "min": 4},
            "lr": {"required": True},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Should have 4 errors: replicas enum, name type, max_steps min, missing lr
        self.assertEqual(len(errors), 4)
        error_text = " ".join(errors)
        self.assertIn("replicas", error_text)
        self.assertIn("name", error_text)
        self.assertIn("max_steps", error_text)
        self.assertIn("lr", error_text)

    def test_validate_recipe_nested_recipe_structure(self):
        """Test validation works with nested recipe structure"""
        recipe = {
            "run": {"replicas": 4},
            "training_config": {
                "max_steps": 100,
                "optim_config": {"lr": 0.00001, "weight_decay": 0.0},
            },
        }
        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32]},
            "max_steps": {"type": "integer", "min": 4, "max": 100000},
            "lr": {"type": "float", "min": 0, "max": 1},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_type_validation_prevents_further_checks(self):
        """Test that type validation failure prevents enum/min/max checks"""
        recipe = {"training_config": {"max_steps": "not_a_number"}}
        overrides_template = {
            "max_steps": {"type": "integer", "min": 4, "max": 100000, "enum": [4, 8]}
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Should only have type error, not enum/min/max errors (due to continue statement)
        self.assertEqual(len(errors), 1)
        self.assertIn("expects integer", errors[0])
        self.assertNotIn("must be one of", errors[0])
        self.assertNotIn("must be at least", errors[0])

    def test_validate_recipe_with_real_world_template(self):
        """Test validation with realistic overrides template"""
        recipe = {
            "run": {
                "name": "my-full-rank-sft-run",
                "data_s3_path": "s3://my-bucket/train.jsonl",
                "output_s3_path": "s3://my-bucket/output/",
                "replicas": 4,
            },
            "training_config": {
                "max_steps": 100,
                "global_batch_size": 64,
                "reasoning_enabled": True,
                "max_length": 8192,
            },
        }

        overrides_template = {
            "replicas": {"type": "integer", "enum": [4, 8, 16, 32], "default": 4},
            "name": {"type": "string", "required": True},
            "data_s3_path": {"type": "string", "required": True},
            "output_s3_path": {"type": "string", "required": True},
            "global_batch_size": {"type": "integer", "enum": [32, 64, 128, 256]},
            "reasoning_enabled": {"type": "boolean", "default": True},
            "max_steps": {"type": "integer", "min": 4, "max": 100000, "default": 100},
            "max_context_length": {"type": "integer", "min": 1, "max": 131072},
            "instance_type": {
                "type": "string",
                "enum": ["ml.p5.48xlarge", "ml.p5en.48xlarge"],
            },
            "namespace": {"type": "string", "default": "kubeflow"},
        }

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    def test_validate_recipe_boolean_type_with_integer(self):
        """Test that boolean type validation catches integer values"""
        recipe = {"training_config": {"reasoning_enabled": 1}}
        overrides_template = {"reasoning_enabled": {"type": "boolean"}}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        # Note: In Python, bool is a subclass of int, so isinstance(1, bool) is False
        # but isinstance(True, int) is True. This test verifies the validation catches this.
        self.assertEqual(len(errors), 1)
        self.assertIn("expects boolean", errors[0])

    def test_validate_recipe_empty_overrides_template(self):
        """Test validation with empty overrides template."""
        recipe = {
            "run": {"name": "my-run"},
            "training_config": {"max_steps": 100},
        }
        overrides_template = {}

        errors = []
        Validator._validate_recipe(
            recipe=recipe,
            overrides_template=overrides_template,
            instance_type="ml.p5.48xlarge",
            errors=errors,
            method=TrainingMethod.SFT_LORA,
        )

        self.assertEqual(len(errors), 0)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_recipe_integration_with_full_validate_method(
        self, mock_boto3_client
    ):
        """Test recipe validation as part of full validate() method"""
        recipe = {"training_config": {"max_steps": 2}}  # Below minimum
        overrides_template = {"max_steps": {"type": "integer", "min": 4}}

        with self.assertRaises(ValueError) as context:
            Validator.validate(
                platform=Platform.SMTJ,
                method=TrainingMethod.SFT_LORA,
                infra=self.mock_infra,
                recipe=recipe,
                overrides_template=overrides_template,
                validation_config={"iam": False, "infra": False},
            )

        self.assertIn("'max_steps' must be at least 4", str(context.exception))


class TestCallingRolePermissionsValidation(unittest.TestCase):
    """Test cases for _validate_calling_role_permissions method"""

    @patch("boto3.client")
    def test_validate_calling_role_permissions_success(self, mock_boto3_client):
        """Test successful permission validation"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation response - permission allowed
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [
            ("sagemaker:CreateTrainingJob", "*"),
            ("iam:PassRole", "*"),
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # Verify correct calls were made
        mock_sts_client.get_caller_identity.assert_called_once()
        self.assertEqual(
            mock_iam_client.simulate_principal_policy.call_count, 3
        )  # 1 test + 2 permissions

    @patch("boto3.client")
    def test_validate_calling_role_permissions_denied(self, mock_boto3_client):
        """Test permission validation with denied permissions"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation response - permission denied
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "implicitDeny"}]
        }

        errors = []
        required_permissions = [("route53:CreateHostedZone", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have one error
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "Missing required calling role permission: route53:CreateHostedZone",
            errors[0],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_direct_role_arn(self, mock_boto3_client):
        """Test with direct role ARN (not assumed role)"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response with direct role ARN
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:role/DirectRole",
            "Account": "123456789012",
        }

        # Mock IAM simulation response
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [("sagemaker:ListClusters", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should use the direct ARN for simulation
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/DirectRole",
            ActionNames=["sagemaker:ListClusters"],
            ResourceArns=["*"],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_simulation_error(
        self, mock_boto3_client
    ):
        """Test handling of IAM simulation errors"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS response
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation error
        mock_iam_client.simulate_principal_policy.side_effect = Exception(
            "Simulation failed"
        )

        errors = []
        required_permissions = [("sagemaker:CreateTrainingJob", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have error about simulation failure
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "Cannot run iam:SimulatePrincipalPolicy to validate calling role permissions: Simulation failed",
            errors[0],
        )

    @patch("boto3.client")
    def test_validate_calling_role_permissions_sts_error(self, mock_boto3_client):
        """Test handling of STS errors"""
        # Mock clients
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        # Mock STS error
        mock_sts_client.get_caller_identity.side_effect = Exception("STS failed")

        errors = []
        required_permissions = [("sagemaker:CreateTrainingJob", "*")]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have error about calling role validation failure
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "Failed to validate calling role permissions: STS failed", errors[0]
        )


class TestPermissionValidationMethods(unittest.TestCase):
    """Test cases for permission validation helper methods and formats"""

    def test_check_policy_json_permissions_grants_required_actions(self):
        """Test successful policy JSON permission checking"""
        policies = [
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject"],
                        "Resource": "*",
                    }
                ]
            }
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_supports_wildcards(self):
        """Test policy JSON permission checking with wildcards"""
        policies = [
            {"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "*"}]}
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_supports_action_prefix_wildcards(self):
        """Test policy JSON permission checking with action prefix wildcards like iam:Get*"""
        policies = [
            {"Statement": [{"Effect": "Allow", "Action": "iam:Get*", "Resource": "*"}]}
        ]

        required_permissions = ["iam:GetRole", "iam:GetPolicy", "iam:GetUser"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        # Should now support prefix wildcards like iam:Get*
        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_action_prefix_wildcards_no_false_positives(
        self,
    ):
        """Test that action prefix wildcards don't match unrelated actions"""
        policies = [
            {"Statement": [{"Effect": "Allow", "Action": "iam:Get*", "Resource": "*"}]}
        ]

        required_permissions = ["iam:CreateRole", "s3:GetObject", "iam:PutRole"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        # iam:Get* should not match iam:CreateRole, s3:GetObject, or iam:PutRole
        self.assertEqual(len(missing), 3)
        self.assertIn("iam:CreateRole", missing)
        self.assertIn("s3:GetObject", missing)
        self.assertIn("iam:PutRole", missing)

    def test_check_policy_json_permissions_supports_infix_wildcards(self):
        """Test policy JSON permission checking with infix wildcards like s3:*Object"""
        policies = [
            {
                "Statement": [
                    {"Effect": "Allow", "Action": "s3:*Object", "Resource": "*"}
                ]
            }
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        # Should support infix wildcards like s3:*Object
        self.assertEqual(len(missing), 0)

    def test_check_policy_json_permissions_infix_wildcards_no_false_positives(self):
        """Test that infix wildcards don't match unrelated actions"""
        policies = [
            {
                "Statement": [
                    {"Effect": "Allow", "Action": "s3:*Object", "Resource": "*"}
                ]
            }
        ]

        required_permissions = ["s3:ListBucket", "s3:GetBucketLocation", "iam:GetRole"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        # s3:*Object should not match s3:ListBucket, s3:GetBucketLocation, or iam:GetRole
        self.assertEqual(len(missing), 3)
        self.assertIn("s3:ListBucket", missing)
        self.assertIn("s3:GetBucketLocation", missing)
        self.assertIn("iam:GetRole", missing)

    def test_check_policy_json_permissions_identifies_missing_actions(self):
        """Test policy JSON permission checking with missing permissions"""
        policies = [
            {
                "Statement": [
                    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "*"}
                ]
            }
        ]

        required_permissions = ["s3:GetObject", "s3:PutObject", "iam:PassRole"]
        missing = Validator._check_policy_json_permissions(
            policies, required_permissions
        )

        self.assertEqual(len(missing), 2)
        self.assertIn("s3:PutObject", missing)
        self.assertIn("iam:PassRole", missing)

    @patch("boto3.client")
    def test_validate_permissions_with_specific_resource_arn(self, mock_boto3_client):
        """Test permission validation using SimulatePrincipalPolicy with specific resource ARN"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        errors = []
        required_permissions = [
            ("sagemaker:CreateTrainingJob", "arn:aws:sagemaker:*:*:training-job/*")
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/TestRole",
            ActionNames=["sagemaker:CreateTrainingJob"],
            ResourceArns=["arn:aws:sagemaker:*:*:training-job/*"],
        )

    @patch("boto3.client")
    def test_validate_permissions_with_dynamic_resource_generation(
        self, mock_boto3_client
    ):
        """Test permission validation using lambda functions to generate resource ARNs"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "allowed"}]
        }

        # Create mock infra
        mock_infra = MagicMock()
        mock_infra.region = "us-west-2"
        mock_infra.cluster_name = "test-cluster"

        errors = []
        required_permissions = [
            (
                "sagemaker:DescribeCluster",
                lambda infra: f"arn:aws:sagemaker:{infra.region}:*:cluster/{infra.cluster_name}",
            )
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, mock_infra, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        mock_iam_client.simulate_principal_policy.assert_called_with(
            PolicySourceArn="arn:aws:iam::123456789012:role/TestRole",
            ActionNames=["sagemaker:DescribeCluster"],
            ResourceArns=["arn:aws:sagemaker:us-west-2:*:cluster/test-cluster"],
        )

    @patch("boto3.client")
    def test_validate_permissions_using_policy_document_parsing(
        self, mock_boto3_client
    ):
        """Test permission validation using JSON policy document parsing for simple strings"""
        mock_iam_client = MagicMock()
        mock_sts_client = MagicMock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy retrieval for JSON parsing
        mock_iam_client.list_role_policies.return_value = {
            "PolicyNames": ["InlinePolicy"]
        }
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": "arn:aws:iam::123456789012:role/specific-role",
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        errors = []
        required_permissions = [
            "iam:PassRole"
        ]  # Simple string - should use JSON parsing

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        self.assertEqual(len(errors), 0)
        # Should call policy retrieval methods for JSON parsing
        mock_iam_client.list_role_policies.assert_called_with(RoleName="TestRole")
        mock_iam_client.get_role_policy.assert_called_with(
            RoleName="TestRole", PolicyName="InlinePolicy"
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_permissions_with_insufficient_resource_permissions(
        self, mock_boto3_client
    ):
        """Test that validation fails when policy has specific resources but not the required ones"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy with specific resources that don't match what's needed
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["S3Policy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            "arn:aws:s3:::wrong-bucket",
                            "arn:aws:s3:::wrong-bucket/*",
                        ],
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        # Mock IAM simulation that will be called for resource-specific permissions
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [{"EvalDecision": "denied"}]
        }

        errors = []
        # Test both JSON parsing (string) and IAM simulation (tuple) permissions
        required_permissions = [
            "iam:PassRole",  # Will use JSON parsing and should fail
            (
                "s3:GetObject",
                "arn:aws:s3:::required-bucket/*",
            ),  # Will use IAM simulation and should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 2 errors - one from JSON parsing, one from IAM simulation
        self.assertEqual(len(errors), 2)
        self.assertIn(
            "Missing required calling role permission: iam:PassRole", errors[0]
        )
        self.assertIn(
            "Missing required calling role permission: s3:GetObject on arn:aws:s3:::required-bucket/*",
            errors[1],
        )

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_permissions_with_multiple_s3_resources(self, mock_boto3_client):
        """Test that validation generates separate permissions for multiple S3 resources"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()
        mock_sagemaker_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
            "sagemaker": mock_sagemaker_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock successful IAM simulation for all resources
        mock_iam_client.simulate_principal_policy.return_value = {
            "EvaluationResults": [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": "arn:aws:s3:::bucket1/data/*",
                    "EvalDecision": "allowed",
                }
            ]
        }

        errors = []

        # Test SMTJ with multiple buckets
        required_permissions = SMTJRuntimeManager.required_calling_role_permissions(
            "s3://bucket1/data/train.jsonl", "s3://bucket2/output/models/"
        )

        # Should have separate permissions for each bucket and object path
        s3_permissions = [
            p
            for p in required_permissions
            if isinstance(p, tuple) and p[0].startswith("s3:")
        ]

        # Should have 7 S3 permissions: 2 buckets × 2 bucket perms + 1 data path read + 2 output path read/write
        self.assertEqual(len(s3_permissions), 7)

        # Check that we have permissions for both buckets
        bucket_permissions = [
            p for p in s3_permissions if p[0] in ["s3:CreateBucket", "s3:ListBucket"]
        ]
        bucket_arns = [p[1] for p in bucket_permissions]
        self.assertIn("arn:aws:s3:::bucket1", bucket_arns)
        self.assertIn("arn:aws:s3:::bucket2", bucket_arns)

        # Check data_s3_path permissions (read-only)
        input_permissions = [p for p in s3_permissions if "bucket1/data" in p[1]]
        self.assertEqual(len(input_permissions), 1)  # Only GetObject
        self.assertEqual(input_permissions[0][0], "s3:GetObject")
        self.assertEqual(
            input_permissions[0][1], "arn:aws:s3:::bucket1/data/train.jsonl*"
        )

        # Check output_s3_path permissions (read-write)
        output_permissions = [p for p in s3_permissions if "bucket2/output" in p[1]]
        self.assertEqual(len(output_permissions), 2)  # GetObject and PutObject
        output_actions = [p[0] for p in output_permissions]
        self.assertIn("s3:GetObject", output_actions)
        self.assertIn("s3:PutObject", output_actions)

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_permissions_fails_with_missing_specific_bucket_access(
        self, mock_boto3_client
    ):
        """Test that validation fails when role lacks access to specific buckets"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock IAM simulation that denies access to bucket2
        def mock_simulate_policy(PolicySourceArn, ActionNames, ResourceArns):
            results = []
            for action in ActionNames:
                for resource in ResourceArns:
                    if "bucket2" in resource:
                        # Deny access to bucket2
                        results.append(
                            {
                                "EvalActionName": action,
                                "EvalResourceName": resource,
                                "EvalDecision": "implicitDeny",
                            }
                        )
                    else:
                        # Allow access to bucket1
                        results.append(
                            {
                                "EvalActionName": action,
                                "EvalResourceName": resource,
                                "EvalDecision": "allowed",
                            }
                        )
            return {"EvaluationResults": results}

        mock_iam_client.simulate_principal_policy.side_effect = mock_simulate_policy

        errors = []
        required_permissions = [
            ("s3:GetObject", "arn:aws:s3:::bucket1/data/*"),
            ("s3:GetObject", "arn:aws:s3:::bucket2/output/*"),  # This should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 1 error for bucket2 access
        self.assertEqual(len(errors), 1)
        self.assertIn("bucket2", errors[0])
        self.assertIn("s3:GetObject", errors[0])

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_permissions_fails_with_bucket_access_but_missing_object_access(
        self, mock_boto3_client
    ):
        """Test validation fails when we have bucket access but not specific object path access"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        def mock_simulate_policy(PolicySourceArn, ActionNames, ResourceArns, **kwargs):
            results = []
            for action in ActionNames:
                for resource in ResourceArns:
                    # Allow bucket-level operations but deny specific object path access
                    if (
                        "s3:ListBucket" in action
                        and resource == "arn:aws:s3:::test-bucket"
                    ):
                        decision = "allowed"
                    elif "s3:GetObject" in action and "specific/path" in resource:
                        decision = "denied"  # Deny access to specific object path
                    else:
                        decision = "allowed"

                    results.append(
                        {
                            "EvalActionName": action,
                            "EvalResourceName": resource,
                            "EvalDecision": decision,
                        }
                    )
            return {"EvaluationResults": results}

        mock_iam_client.simulate_principal_policy.side_effect = mock_simulate_policy

        errors = []
        required_permissions = [
            ("s3:ListBucket", "arn:aws:s3:::test-bucket"),  # This should pass
            (
                "s3:GetObject",
                "arn:aws:s3:::test-bucket/specific/path/*",
            ),  # This should fail
        ]

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should have 1 error for the specific object path access
        self.assertEqual(len(errors), 1)
        self.assertIn("test-bucket/specific/path", errors[0])
        self.assertIn("s3:GetObject", errors[0])

    def test_validate_permissions_handles_lambda_evaluation_errors(self):
        """Test error handling when resource lambda functions fail during evaluation"""
        mock_infra = MagicMock()

        def failing_lambda(infra):
            raise Exception("Lambda evaluation failed")

        errors = []
        required_permissions = [("sagemaker:DescribeCluster", failing_lambda)]

        # This should handle the lambda failure gracefully
        with patch("boto3.client") as mock_boto3_client:
            mock_iam_client = MagicMock()
            mock_sts_client = MagicMock()

            mock_boto3_client.side_effect = lambda service, **kwargs: {
                "iam": mock_iam_client,
                "sts": mock_sts_client,
            }[service]

            mock_sts_client.get_caller_identity.return_value = {
                "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
                "Account": "123456789012",
            }

            Validator._validate_calling_role_permissions(
                errors, required_permissions, mock_infra, "us-east-1"
            )

            self.assertEqual(len(errors), 1)
            self.assertIn("Failed to evaluate resource lambda", errors[0])

    @patch("sagemaker.get_execution_role")
    @patch("subprocess.run")
    def test_runtime_managers_define_mixed_permission_validation_types(
        self, mock_run, mock_get_execution_role
    ):
        """Test that runtime managers correctly define permissions using multiple validation approaches"""
        from amzn_nova_customization_sdk.manager.runtime_manager import (
            SMHPRuntimeManager,
            SMTJRuntimeManager,
        )

        # Mock subprocess for SMHP
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        # Mock sagemaker execution role for SMTJ
        mock_get_execution_role.return_value = (
            "arn:aws:iam::123456789012:role/test-role"
        )

        # Test SMHP permissions
        smhp_permissions = SMHPRuntimeManager.required_calling_role_permissions()
        self.assertIsInstance(smhp_permissions, list)
        self.assertGreater(len(smhp_permissions), 0)

        # Should have tuple-specified calling role permissions
        has_tuples = any(isinstance(p, tuple) for p in smhp_permissions)
        self.assertTrue(has_tuples, "SMHP should have tuple permissions")

        # Test SMTJ permissions
        smtj_permissions = SMTJRuntimeManager.required_calling_role_permissions()
        self.assertIsInstance(smtj_permissions, list)
        self.assertGreater(len(smtj_permissions), 0)

        # Should have mixed types
        has_tuples = any(isinstance(p, tuple) for p in smtj_permissions)
        has_strings = any(isinstance(p, str) for p in smtj_permissions)
        self.assertTrue(has_tuples, "SMTJ should have tuple permissions")
        self.assertTrue(has_strings, "SMTJ should have string permissions")

    @patch("amzn_nova_customization_sdk.validation.validator.boto3.client")
    def test_validate_calling_role_permissions_with_action_prefix_wildcards(
        self, mock_boto3_client
    ):
        """Test that validation correctly handles action prefix wildcards like iam:Get* in policies"""
        mock_iam_client = Mock()
        mock_sts_client = Mock()

        mock_boto3_client.side_effect = lambda service, **kwargs: {
            "iam": mock_iam_client,
            "sts": mock_sts_client,
        }[service]

        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session",
            "Account": "123456789012",
        }

        # Mock policy with action prefix wildcard
        mock_iam_client.list_role_policies.return_value = {"PolicyNames": ["IAMPolicy"]}
        mock_iam_client.get_role_policy.return_value = {
            "PolicyDocument": {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "iam:Get*",  # Action prefix wildcard
                        "Resource": "*",
                    }
                ]
            }
        }
        mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": []
        }

        errors = []
        required_permissions = ["iam:GetRole"]  # Should be covered by iam:Get*

        Validator._validate_calling_role_permissions(
            errors, required_permissions, None, "us-east-1"
        )

        # Should now support prefix wildcards like iam:Get*
        self.assertEqual(len(errors), 0)

    def test_validate_job_name_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_job_name("bad_job_name")

        self.assertEqual(
            str(context.exception),
            f"Job name must fit pattern ${JOB_NAME_REGEX.pattern}",
        )

    def test_validate_job_name_does_not_raise_exception(self):
        Validator.validate_job_name("good-job-name")

    def test_validate_namespace_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_namespace("!bad_namespace")

        self.assertEqual(
            str(context.exception),
            f"Namespace must fit pattern ${NAMESPACE_REGEX.pattern}",
        )

    def test_validate_namespace_does_not_raise_exception(self):
        Validator.validate_namespace("good-namespace")

    def test_validate_cluster_name_raises_exception(self):
        with self.assertRaises(ValueError) as context:
            Validator.validate_cluster_name("!bad_cluster_name")

        self.assertEqual(
            str(context.exception),
            f"Cluster name must fit pattern ${CLUSTER_NAME_REGEX.pattern}",
        )

    def test_validate_cluster_name_does_not_raise_exception(self):
        Validator.validate_cluster_name("good_cluster-name")


if __name__ == "__main__":
    unittest.main()
