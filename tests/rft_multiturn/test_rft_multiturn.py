"""Tests for RFTMultiturnInfrastructure main class."""

from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_customization_sdk.rft_multiturn import (
    EnvType,
    RFTMultiturnInfrastructure,
    VFEnvId,
    list_rft_stacks,
)
from amzn_nova_customization_sdk.rft_multiturn.custom_environment import (
    CustomEnvironment,
)


class TestRFTMultiturnInfrastructure:
    """Test RFTMultiturnInfrastructure class."""

    def test_rft_multiturn_infrastructure_exists(self):
        """Test that RFTMultiturnInfrastructure class is importable."""
        assert RFTMultiturnInfrastructure is not None

    def test_rft_multiturn_has_required_methods(self):
        """Test that RFTMultiturnInfrastructure has expected methods."""
        assert hasattr(RFTMultiturnInfrastructure, "setup")
        assert hasattr(RFTMultiturnInfrastructure, "cleanup")
        assert hasattr(RFTMultiturnInfrastructure, "start_training_environment")
        assert hasattr(RFTMultiturnInfrastructure, "start_evaluation_environment")
        assert hasattr(RFTMultiturnInfrastructure, "get_logs")
        assert hasattr(RFTMultiturnInfrastructure, "kill_task")
        assert hasattr(RFTMultiturnInfrastructure, "check_all_queues")
        assert hasattr(RFTMultiturnInfrastructure, "flush_all_queues")
        assert hasattr(RFTMultiturnInfrastructure, "get_configuration")
        assert hasattr(RFTMultiturnInfrastructure, "get_recipe_path")
        assert hasattr(RFTMultiturnInfrastructure, "get_recipe_overrides")
        assert hasattr(RFTMultiturnInfrastructure, "detect_platform")

    def test_detect_platform_local(self):
        """Test platform detection returns 'local' for None infrastructure_arn."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    python_venv_name="venv",
                )
                assert rft.platform == "local"

    def test_detect_platform_ec2_instance_id(self):
        """Test platform detection returns 'ec2' for instance ID."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    infrastructure_arn="i-1234567890abcdef0",
                    python_venv_name="venv",
                )
                assert rft.platform == "ec2"

    def test_detect_platform_ec2_arn(self):
        """Test platform detection returns 'ec2' for EC2 ARN."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    infrastructure_arn="arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0",
                    python_venv_name="venv",
                )
                assert rft.platform == "ec2"

    def test_detect_platform_ecs(self):
        """Test platform detection returns 'ecs' for ECS ARN."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
                )
                assert rft.platform == "ecs"

    def test_init_with_vf_env_id_enum(self):
        """Test initialization with VFEnvId enum."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    python_venv_name="venv",
                )
                assert rft.env_id == "wordle"
                assert rft.is_custom_env is False

    def test_init_with_vf_env_id_string(self):
        """Test initialization with VFEnvId as string."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id="terminalbench_env",
                    python_venv_name="venv",
                )
                assert rft.env_id == "terminalbench_env"

    def test_init_with_custom_env(self):
        """Test initialization with custom environment."""
        custom_env = CustomEnvironment(env_id="my-env", local_path="/path/to/env")
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test", custom_env=custom_env, python_venv_name="venv"
                )
                assert rft.env_id == "my-env"
                assert rft.is_custom_env is True

    def test_init_fails_with_both_vf_and_custom(self):
        """Test initialization fails with both vf_env_id and custom_env."""
        custom_env = CustomEnvironment(env_id="my-env", local_path="/path/to/env")
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="Cannot specify both"):
                RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    custom_env=custom_env,
                    python_venv_name="venv",
                )

    def test_init_fails_without_env(self):
        """Test initialization fails without environment."""
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="Please specify one of"):
                RFTMultiturnInfrastructure(stack_name="test", python_venv_name="venv")

    def test_init_fails_with_invalid_vf_env_id(self):
        """Test initialization fails with invalid vf_env_id."""
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="Invalid vf_env_id"):
                RFTMultiturnInfrastructure(
                    stack_name="test", vf_env_id="invalid", python_venv_name="venv"
                )

    def test_local_requires_python_venv_name(self):
        """Test LOCAL platform requires python_venv_name."""
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="python_venv_name is required"):
                RFTMultiturnInfrastructure(stack_name="test", vf_env_id=VFEnvId.WORDLE)

    def test_ec2_requires_python_venv_name(self):
        """Test EC2 platform requires python_venv_name."""
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="python_venv_name is required"):
                RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    infrastructure_arn="i-1234567890abcdef0",
                )

    def test_ecs_defaults_python_venv_name(self):
        """Test ECS platform defaults python_venv_name."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
                )
                assert rft.platform == "ecs"

    def test_custom_env_local_requires_local_path(self):
        """Test custom environment on LOCAL requires local_path."""
        custom_env = CustomEnvironment(env_id="my-env")
        with patch("boto3.client"):
            with pytest.raises(
                ValueError, match="CustomEnvironment.local_path required"
            ):
                RFTMultiturnInfrastructure(
                    stack_name="test", custom_env=custom_env, python_venv_name="venv"
                )

    def test_custom_env_ec2_requires_s3_uri(self):
        """Test custom environment on EC2 requires s3_uri."""
        custom_env = CustomEnvironment(env_id="my-env", local_path="/path")
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="CustomEnvironment.s3_uri required"):
                RFTMultiturnInfrastructure(
                    stack_name="test",
                    custom_env=custom_env,
                    infrastructure_arn="i-1234567890abcdef0",
                    python_venv_name="venv",
                )

    def test_custom_env_ecs_requires_s3_uri(self):
        """Test custom environment on ECS requires s3_uri."""
        custom_env = CustomEnvironment(env_id="my-env", local_path="/path")
        with patch("boto3.client"):
            with pytest.raises(ValueError, match="CustomEnvironment.s3_uri required"):
                RFTMultiturnInfrastructure(
                    stack_name="test",
                    custom_env=custom_env,
                    infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
                )

    def test_stack_name_has_suffix(self):
        """Test that stack_name gets NovaForgeSDK suffix."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ) as mock_infra:
                mock_instance = MagicMock()
                mock_instance.stack_name = "test-NovaForgeSDK"
                mock_infra.return_value = mock_instance

                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    python_venv_name="venv",
                )
                assert rft.stack_name.endswith("NovaForgeSDK")

    def test_region_defaults_to_us_east_1(self):
        """Test that region defaults to us-east-1."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    vf_env_id=VFEnvId.WORDLE,
                    python_venv_name="venv",
                )
                assert rft.region == "us-east-1"

    def test_region_can_be_customized(self):
        """Test that region can be customized."""
        with patch("boto3.client"):
            with patch(
                "amzn_nova_customization_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
            ):
                rft = RFTMultiturnInfrastructure(
                    stack_name="test",
                    region="us-west-2",
                    vf_env_id=VFEnvId.WORDLE,
                    python_venv_name="venv",
                )
                assert rft.region == "us-west-2"


class TestListRFTStacks:
    """Test list_rft_stacks function."""

    def test_list_rft_stacks_exists(self):
        """Test that list_rft_stacks function exists."""
        assert list_rft_stacks is not None
        assert callable(list_rft_stacks)

    @patch("boto3.client")
    def test_list_rft_stacks_filters_by_suffix(self, mock_client):
        """Test that list_rft_stacks filters by NovaForgeSDK suffix."""
        mock_cfn = MagicMock()
        mock_client.return_value = mock_cfn

        mock_paginator = MagicMock()
        mock_cfn.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "StackSummaries": [
                    {"StackName": "test-stack-NovaForgeSDK"},
                    {"StackName": "other-stack"},
                    {"StackName": "another-NovaForgeSDK"},
                ]
            }
        ]

        stacks = list_rft_stacks(region="us-east-1", all_stacks=False)

        assert len(stacks) == 2
        assert "test-stack-NovaForgeSDK" in stacks
        assert "another-NovaForgeSDK" in stacks
        assert "other-stack" not in stacks

    @patch("boto3.client")
    def test_list_rft_stacks_all_stacks(self, mock_client):
        """Test that list_rft_stacks returns all stacks when requested."""
        mock_cfn = MagicMock()
        mock_client.return_value = mock_cfn

        mock_paginator = MagicMock()
        mock_cfn.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "StackSummaries": [
                    {"StackName": "test-stack-NovaForgeSDK"},
                    {"StackName": "other-stack"},
                ]
            }
        ]

        stacks = list_rft_stacks(region="us-west-2", all_stacks=True)

        assert len(stacks) == 2
        assert "test-stack-NovaForgeSDK" in stacks
        assert "other-stack" in stacks

    @patch("boto3.client")
    def test_list_rft_stacks_returns_sorted(self, mock_client):
        """Test that list_rft_stacks returns sorted stack names."""
        mock_cfn = MagicMock()
        mock_client.return_value = mock_cfn

        mock_paginator = MagicMock()
        mock_cfn.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "StackSummaries": [
                    {"StackName": "z-stack-NovaForgeSDK"},
                    {"StackName": "a-stack-NovaForgeSDK"},
                    {"StackName": "m-stack-NovaForgeSDK"},
                ]
            }
        ]

        stacks = list_rft_stacks(region="us-east-1")

        assert stacks == [
            "a-stack-NovaForgeSDK",
            "m-stack-NovaForgeSDK",
            "z-stack-NovaForgeSDK",
        ]
