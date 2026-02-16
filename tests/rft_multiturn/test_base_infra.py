"""Tests for base RFT infrastructure components."""

from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_customization_sdk.rft_multiturn import EnvType, StackOutputs, VFEnvId
from amzn_nova_customization_sdk.rft_multiturn.base_infra import (
    ECR_REPO_NAME,
    RFT_EXECUTION_ROLE_NAME,
    RFT_POLICY_NAME,
    STACK_NAME_SUFFIX,
    STARTER_KIT_S3,
    create_rft_execution_role,
)


class TestEnvType:
    """Test EnvType enum."""

    def test_env_type_values(self):
        """Test that EnvType has expected values."""
        assert EnvType.TRAIN.value == "train"
        assert EnvType.EVAL.value == "eval"
        assert EnvType.SAM.value == "sam"

    def test_env_type_is_enum(self):
        """Test that EnvType is an enum."""
        assert isinstance(EnvType.TRAIN, EnvType)
        assert isinstance(EnvType.EVAL, EnvType)
        assert isinstance(EnvType.SAM, EnvType)


class TestVFEnvId:
    """Test VFEnvId enum."""

    def test_vf_env_id_has_values(self):
        """Test that VFEnvId enum has values."""
        assert VFEnvId.WORDLE.value == "wordle"
        assert VFEnvId.TERMINAL_BENCH.value == "terminalbench_env"

    def test_vf_env_id_is_string_enum(self):
        """Test that VFEnvId is a string enum."""
        assert isinstance(VFEnvId.WORDLE, str)
        assert isinstance(VFEnvId.TERMINAL_BENCH, str)


class TestStackOutputs:
    """Test StackOutputs dataclass."""

    def test_stack_outputs_creation(self):
        """Test creating StackOutputs instance."""
        outputs = StackOutputs(
            rollout_request_arn="arn:aws:sqs:us-east-1:123456789012:queue",
            rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/queue",
            rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/queue",
            generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/queue",
            generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/queue",
            proxy_function_url="https://lambda.us-east-1.amazonaws.com/function",
            dynamo_table_name="my-table",
        )
        assert outputs.rollout_request_arn.startswith("arn:aws:sqs")
        assert outputs.dynamo_table_name == "my-table"

    def test_stack_outputs_has_all_required_fields(self):
        """Test that StackOutputs has all required fields."""
        outputs = StackOutputs(
            rollout_request_arn="arn1",
            rollout_response_sqs_url="url1",
            rollout_request_queue_url="url2",
            generate_request_sqs_url="url3",
            generate_response_sqs_url="url4",
            proxy_function_url="url5",
            dynamo_table_name="table",
        )
        assert hasattr(outputs, "rollout_request_arn")
        assert hasattr(outputs, "rollout_response_sqs_url")
        assert hasattr(outputs, "rollout_request_queue_url")
        assert hasattr(outputs, "generate_request_sqs_url")
        assert hasattr(outputs, "generate_response_sqs_url")
        assert hasattr(outputs, "proxy_function_url")
        assert hasattr(outputs, "dynamo_table_name")


class TestConstants:
    """Test module constants."""

    def test_stack_name_suffix_exists(self):
        """Test that STACK_NAME_SUFFIX constant exists."""
        assert STACK_NAME_SUFFIX is not None
        assert isinstance(STACK_NAME_SUFFIX, str)
        assert len(STACK_NAME_SUFFIX) > 0

    def test_rft_execution_role_name_exists(self):
        """Test that RFT_EXECUTION_ROLE_NAME constant exists."""
        assert RFT_EXECUTION_ROLE_NAME is not None
        assert isinstance(RFT_EXECUTION_ROLE_NAME, str)

    def test_rft_policy_name_exists(self):
        """Test that RFT_POLICY_NAME constant exists."""
        assert RFT_POLICY_NAME is not None
        assert isinstance(RFT_POLICY_NAME, str)

    def test_ecr_repo_name_exists(self):
        """Test that ECR_REPO_NAME constant exists."""
        assert ECR_REPO_NAME is not None
        assert isinstance(ECR_REPO_NAME, str)

    def test_starter_kit_s3_exists(self):
        """Test that STARTER_KIT_S3 constant exists."""
        assert STARTER_KIT_S3 is not None
        assert isinstance(STARTER_KIT_S3, str)
        assert STARTER_KIT_S3.startswith("s3://")


class TestCreateRFTExecutionRole:
    """Test create_rft_execution_role function."""

    def test_create_rft_execution_role_exists(self):
        """Test that create_rft_execution_role function exists."""
        assert create_rft_execution_role is not None
        assert callable(create_rft_execution_role)

    def test_create_rft_execution_role_is_function(self):
        """Test that create_rft_execution_role is a function."""
        import inspect

        assert inspect.isfunction(create_rft_execution_role)

    @patch("boto3.client")
    @patch("time.sleep")  # Mock sleep to avoid delays
    def test_create_rft_execution_role_creates_new_role(self, mock_sleep, mock_client):
        """Test creating a new RFT execution role."""
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            elif service == "sts":
                return mock_sts
            return MagicMock()

        mock_client.side_effect = client_factory

        # Mock STS response
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        # Mock IAM responses - role doesn't exist
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/RFTExecutionRoleNovaSDK"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {
                "Arn": "arn:aws:iam::123456789012:policy/RFTExecutionRoleNovaSDKPolicy"
            }
        }
        # First call returns empty (not attached), second call returns attached (after attach)
        mock_iam.list_attached_role_policies.side_effect = [
            {"AttachedPolicies": []},
            {
                "AttachedPolicies": [
                    {
                        "PolicyArn": "arn:aws:iam::123456789012:policy/RFTExecutionRoleNovaSDKPolicy"
                    }
                ]
            },
        ]

        role_arn = create_rft_execution_role(region="us-east-1")

        assert role_arn == "arn:aws:iam::123456789012:role/RFTExecutionRoleNovaSDK"
        mock_iam.create_role.assert_called_once()
        mock_iam.create_policy.assert_called_once()
        mock_iam.attach_role_policy.assert_called_once()

    @patch("boto3.client")
    @patch("time.sleep")  # Mock sleep to avoid delays
    def test_create_rft_execution_role_uses_existing_role(
        self, mock_sleep, mock_client
    ):
        """Test using an existing RFT execution role."""
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            elif service == "sts":
                return mock_sts
            return MagicMock()

        mock_client.side_effect = client_factory

        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        # Mock IAM responses - role exists
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/RFTExecutionRoleNovaSDK"}
        }
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (Exception,), {}
        )
        mock_iam.create_policy.side_effect = (
            mock_iam.exceptions.EntityAlreadyExistsException()
        )
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {
                    "PolicyArn": "arn:aws:iam::123456789012:policy/RFTExecutionRoleNovaSDKPolicy"
                }
            ]
        }

        role_arn = create_rft_execution_role(region="us-west-2")

        assert role_arn == "arn:aws:iam::123456789012:role/RFTExecutionRoleNovaSDK"
        mock_iam.create_role.assert_not_called()

    @patch("boto3.client")
    @patch("time.sleep")  # Mock sleep to avoid delays
    def test_create_rft_execution_role_with_custom_name(self, mock_sleep, mock_client):
        """Test creating role with custom name."""
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            elif service == "sts":
                return mock_sts
            return MagicMock()

        mock_client.side_effect = client_factory

        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (Exception,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CustomRoleName"}
        }
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/CustomRoleNamePolicy"}
        }
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {"PolicyArn": "arn:aws:iam::123456789012:policy/CustomRoleNamePolicy"}
            ]
        }

        role_arn = create_rft_execution_role(
            region="us-east-1", role_name="CustomRoleName"
        )

        assert "CustomRoleName" in role_arn
        call_args = mock_iam.create_role.call_args
        assert call_args[1]["RoleName"] == "CustomRoleName"
