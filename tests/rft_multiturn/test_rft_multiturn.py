"""Tests for RFTMultiturnInfrastructure main class."""

from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn import (
    EnvType,
    RFTMultiturnInfrastructure,
    VFEnvId,
    list_rft_stacks,
)
from amzn_nova_forge_sdk.rft_multiturn.custom_environment import (
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
        assert hasattr(RFTMultiturnInfrastructure, "start_environment")
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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
                "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
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


class TestRFTMultiturnDumpLoad:
    """Test dump() and load() methods for state serialization."""

    @patch("boto3.client")
    def test_dump_default_parameters(self, mock_boto_client, tmp_path):
        """Test dump with default parameters."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv",
                "starter_kit_path": "/path/to/kit",
                "base_path": "/base/path",
            }
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )
            rft._session_id = "abc123"

            # Dump to tmp directory
            import os

            original_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                result_path = rft.dump()

                # Verify file was created with session_id in name
                assert result_path.exists()
                assert "test-stack-NovaForgeSDK" in result_path.name
                assert "abc123" in result_path.name
                assert result_path.name.startswith("rft_state_")
                assert result_path.name.endswith(".json")
            finally:
                os.chdir(original_cwd)

    @patch("boto3.client")
    def test_dump_custom_file_path(self, mock_boto_client, tmp_path):
        """Test dump with custom file path."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv"
            }
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            result_path = rft.dump(file_path=str(tmp_path))

            assert result_path.parent == tmp_path
            assert result_path.exists()

    @patch("boto3.client")
    def test_dump_custom_file_name(self, mock_boto_client, tmp_path):
        """Test dump with custom file name."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv"
            }
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            result_path = rft.dump(
                file_path=str(tmp_path), file_name="custom_state.json"
            )

            assert result_path.name == "custom_state.json"
            assert result_path.exists()

    @patch("boto3.client")
    def test_dump_without_session_id(self, mock_boto_client, tmp_path):
        """Test dump without session_id in filename."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv"
            }
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )
            rft._session_id = "abc123"

            result_path = rft.dump(file_path=str(tmp_path), include_session_id=False)

            assert result_path.name == "rft_state_test-stack-NovaForgeSDK.json"
            assert "abc123" not in result_path.name

    @patch("boto3.client")
    def test_dump_serializes_all_fields(self, mock_boto_client, tmp_path):
        """Test that dump serializes all required fields."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv",
                "base_path": "/base",
            }
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                region="us-west-2",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            result_path = rft.dump(file_path=str(tmp_path))

            # Read and verify JSON content
            import json

            with open(result_path, "r") as f:
                state = json.load(f)

            assert state["__class_name__"] == "RFTMultiturnInfrastructure"
            assert state["session_id"] == rft._session_id
            assert state["platform"] == "local"
            assert state["region"] == "us-west-2"
            assert state["stack_name"] == "test-stack-NovaForgeSDK"
            assert state["env_id"] == VFEnvId.WORDLE
            assert state["rft_role_name"] == "TestRole"
            assert state["is_custom_env"] is False
            assert state["custom_env"] is None
            assert "infra_state" in state
            assert "dumped_at" in state

    @patch("boto3.client")
    def test_dump_with_custom_environment(self, mock_boto_client, tmp_path):
        """Test dump with custom environment."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv"
            }
            mock_local.return_value = mock_local_instance

            custom_env = CustomEnvironment(
                env_id="custom_env_1",
                local_path="/path/to/env",
                output_dir="/output",
            )

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                custom_env=custom_env,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            result_path = rft.dump(file_path=str(tmp_path))

            import json

            with open(result_path, "r") as f:
                state = json.load(f)

            assert state["is_custom_env"] is True
            assert state["custom_env"] is not None
            assert state["custom_env"]["env_id"] == "custom_env_1"
            assert state["custom_env"]["local_path"] == "/path/to/env"

    @patch("boto3.client")
    def test_load_local_platform(self, mock_boto_client, tmp_path):
        """Test load for LOCAL platform."""
        mock_boto_client.return_value = MagicMock()

        # Create state file
        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "test123",
            "platform": "local",
            "region": "us-east-1",
            "stack_name": "test-stack-NovaForgeSDK",
            "env_id": VFEnvId.WORDLE,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": None,
            "is_custom_env": False,
            "custom_env": None,
            "stack_outputs": None,
            "infra_state": {"python_venv_name": "test_venv", "base_path": "/base"},
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "test_state.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.restore_state = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure.load(str(state_file))

            assert rft._session_id == "test123"
            assert rft.platform == "local"
            assert rft.stack_name == "test-stack-NovaForgeSDK"
            assert rft.region == "us-east-1"
            mock_local_instance.restore_state.assert_called_once()

    @patch("amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure")
    @patch("boto3.client")
    def test_load_ec2_platform(self, mock_boto_client, mock_ec2_class, tmp_path):
        """Test load for EC2 platform."""
        # Import the real class to use for spec
        from amzn_nova_forge_sdk.rft_multiturn.ec2_infra import (
            EC2RFTInfrastructure as RealEC2RFTInfrastructure,
        )

        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}}]}]
        }
        mock_boto_client.return_value = mock_ec2

        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "ec2test",
            "platform": "ec2",
            "region": "us-east-1",
            "stack_name": "test-stack-NovaForgeSDK",
            "env_id": VFEnvId.WORDLE,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": "i-1234567890abcdef0",
            "is_custom_env": False,
            "custom_env": None,
            "stack_outputs": None,
            "infra_state": {
                "instance_id": "i-1234567890abcdef0",
                "python_venv_name": "test_venv",
            },
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "ec2_state.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        # Create a mock instance that will pass isinstance check
        mock_ec2_instance = MagicMock(spec=RealEC2RFTInfrastructure)
        mock_ec2_instance.stack_name = "test-stack-NovaForgeSDK"
        mock_ec2_instance.restore_state = MagicMock()
        mock_ec2_instance.session_id = None

        # Make the mock class itself behave like the real class for isinstance
        mock_ec2_class.return_value = mock_ec2_instance
        mock_ec2_class.__bases__ = RealEC2RFTInfrastructure.__bases__
        mock_ec2_class.__mro__ = (mock_ec2_class,) + RealEC2RFTInfrastructure.__mro__

        rft = RFTMultiturnInfrastructure.load(str(state_file))

        assert rft._session_id == "ec2test"
        assert rft.platform == "ec2"
        assert rft.stack_name == "test-stack-NovaForgeSDK"
        mock_ec2_instance.restore_state.assert_called_once()

    @patch("boto3.client")
    def test_load_ecs_platform(self, mock_boto_client, tmp_path):
        """Test load for ECS platform."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "RUNNING"}]}
        mock_boto_client.return_value = mock_ecs

        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "ecstest",
            "platform": "ecs",
            "region": "us-east-1",
            "stack_name": "test-stack-NovaForgeSDK",
            "env_id": VFEnvId.WORDLE,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": "arn:aws:ecs:us-east-1:123456789012:cluster/test",
            "is_custom_env": False,
            "custom_env": None,
            "stack_outputs": None,
            "infra_state": {
                "cluster_arn": "arn:aws:ecs:us-east-1:123456789012:cluster/test",
                "account_id": "123456789012",
                "python_venv_name": "test_venv",
            },
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "ecs_state.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
        ) as mock_ecs_class:
            mock_ecs_instance = MagicMock()
            mock_ecs_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_ecs_instance.restore_state = MagicMock()
            mock_ecs_class.return_value = mock_ecs_instance

            rft = RFTMultiturnInfrastructure.load(str(state_file))

            assert rft._session_id == "ecstest"
            assert rft.platform == "ecs"
            assert rft.stack_name == "test-stack-NovaForgeSDK"
            mock_ecs_instance.restore_state.assert_called_once()

    @patch("boto3.client")
    def test_load_with_custom_environment(self, mock_boto_client, tmp_path):
        """Test load with custom environment."""
        mock_boto_client.return_value = MagicMock()

        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "custom123",
            "platform": "local",
            "region": "us-east-1",
            "stack_name": "test-stack-NovaForgeSDK",
            "env_id": None,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": None,
            "is_custom_env": True,
            "custom_env": {
                "env_id": "custom_env_1",
                "local_path": "/path/to/env",
                "s3_uri": "s3://bucket/env",
                "output_dir": "/output",
                "env_type": "single_turn",
            },
            "stack_outputs": None,
            "infra_state": {"python_venv_name": "test_venv"},
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "custom_state.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure.load(str(state_file))

            assert rft.is_custom_env is True
            assert rft.custom_env is not None
            assert rft.custom_env.env_id == "custom_env_1"

    def test_load_missing_file(self, tmp_path):
        """Test load with missing file raises error."""
        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            RFTMultiturnInfrastructure.load(str(missing_file))

    def test_load_invalid_json(self, tmp_path):
        """Test load with invalid JSON raises error."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json{")

        import json

        with pytest.raises(json.JSONDecodeError):
            RFTMultiturnInfrastructure.load(str(invalid_file))

    @patch("boto3.client")
    def test_load_unknown_platform(self, mock_boto_client, tmp_path):
        """Test load with unknown platform raises error."""
        mock_boto_client.return_value = MagicMock()

        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "test123",
            "platform": "unknown_platform",
            "region": "us-east-1",
            "stack_name": "test-stack",
            "env_id": VFEnvId.WORDLE,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": None,
            "is_custom_env": False,
            "custom_env": None,
            "stack_outputs": None,
            "infra_state": {},
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "bad_platform.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        with pytest.raises(ValueError, match="Unknown platform"):
            RFTMultiturnInfrastructure.load(str(state_file))

    @patch("boto3.client")
    def test_load_without_auto_reconnect(self, mock_boto_client, tmp_path):
        """Test load with auto_reconnect=False skips restore_state."""
        mock_boto_client.return_value = MagicMock()

        state = {
            "__class_name__": "RFTMultiturnInfrastructure",
            "session_id": "test123",
            "platform": "local",
            "region": "us-east-1",
            "stack_name": "test-stack-NovaForgeSDK",
            "env_id": VFEnvId.WORDLE,
            "workspace_dir": None,
            "rft_role_name": "TestRole",
            "infrastructure_arn": None,
            "is_custom_env": False,
            "custom_env": None,
            "stack_outputs": None,
            "infra_state": {"python_venv_name": "test_venv"},
            "dumped_at": "2024-01-01T00:00:00",
        }

        state_file = tmp_path / "test_state.json"
        import json

        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.restore_state = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure.load(str(state_file), auto_reconnect=False)

            # restore_state should NOT be called
            mock_local_instance.restore_state.assert_not_called()

    @patch("boto3.client")
    def test_dump_load_roundtrip(self, mock_boto_client, tmp_path):
        """Test dump and load roundtrip preserves state."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.stack_name = "test-stack-NovaForgeSDK"
            mock_local_instance.get_state.return_value = {
                "python_venv_name": "test_venv",
                "base_path": "/base",
            }
            mock_local_instance.restore_state = MagicMock()
            mock_local.return_value = mock_local_instance

            # Create and dump
            rft1 = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                region="us-west-2",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )
            original_session_id = rft1._session_id

            state_file = rft1.dump(file_path=str(tmp_path))

            # Load
            rft2 = RFTMultiturnInfrastructure.load(str(state_file))

            # Verify state preserved
            assert rft2._session_id == original_session_id
            assert rft2.stack_name == "test-stack-NovaForgeSDK"
            assert rft2.region == "us-west-2"
            assert rft2.platform == "local"


class TestRFTMultiturnStartEnvironment:
    """Test start_environment method."""

    @patch("boto3.client")
    def test_start_environment_train_without_stack(self, mock_boto_client):
        """Test start_environment raises error when stack not deployed."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local.return_value = MagicMock()

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            with pytest.raises(RuntimeError, match="Stack not deployed"):
                rft.start_environment(env_type=EnvType.TRAIN)

    @patch("boto3.client")
    def test_start_environment_train_local(self, mock_boto_client):
        """Test start_environment for TRAIN on LOCAL platform."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            # Mock stack outputs
            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(
                env_type=EnvType.TRAIN,
                vf_env_args={"use_think": True},
                max_concurrent_rollouts=50,
            )

            # Verify local setup was called
            mock_local_instance.setup_local.assert_called_once()
            mock_local_instance.install_local_environment.assert_called_once_with(
                VFEnvId.WORDLE
            )

            # Verify start_environment was called with correct parameters
            mock_local_instance.start_environment.assert_called_once()
            call_args = mock_local_instance.start_environment.call_args
            assert call_args.kwargs["env_type"] == EnvType.TRAIN
            assert call_args.kwargs["vf_env_args"] == {"use_think": True}
            assert call_args.kwargs["max_concurrent_rollouts"] == 50

    @patch("boto3.client")
    def test_start_environment_eval_local(self, mock_boto_client):
        """Test start_environment for EVAL on LOCAL platform."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(env_type=EnvType.EVAL)

            call_args = mock_local_instance.start_environment.call_args
            assert call_args.kwargs["env_type"] == EnvType.EVAL

    @patch("boto3.client")
    def test_start_environment_with_config_name(self, mock_boto_client):
        """Test start_environment with YAML config."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(
                env_type=EnvType.TRAIN,
                config_name="wordle_training",
                config_path="/custom/configs",
            )

            call_args = mock_local_instance.start_environment.call_args
            assert call_args.kwargs["config_name"] == "wordle_training"
            assert call_args.kwargs["config_path"] == "/custom/configs"

    @patch("boto3.client")
    def test_start_environment_with_custom_queue(self, mock_boto_client):
        """Test start_environment with custom queue URL."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            custom_queue = (
                "https://sqs.us-east-1.amazonaws.com/123456789012/custom-queue"
            )
            rft.start_environment(env_type=EnvType.TRAIN, queue_url=custom_queue)

            # Queue URL is not passed to infra.start_environment, it's used internally
            # Just verify the method was called
            mock_local_instance.start_environment.assert_called_once()

    @patch("boto3.client")
    def test_start_environment_default_vf_env_args(self, mock_boto_client):
        """Test start_environment uses default vf_env_args."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            # Don't provide vf_env_args
            rft.start_environment(env_type=EnvType.TRAIN)

            call_args = mock_local_instance.start_environment.call_args
            # Should default to {"use_think": False}
            assert call_args.kwargs["vf_env_args"] == {"use_think": False}

    @patch("boto3.client")
    def test_start_environment_ec2_no_setup_call(self, mock_boto_client):
        """Test start_environment for EC2 doesn't call setup (handled internally)."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "State": {"Name": "running"},
                            "IamInstanceProfile": {
                                "Arn": "arn:aws:iam::123456789012:instance-profile/test"
                            },
                        }
                    ]
                }
            ]
        }
        mock_boto_client.return_value = mock_ec2

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.EC2RFTInfrastructure"
        ) as mock_ec2_class:
            mock_ec2_instance = MagicMock()
            mock_ec2_instance.start_environment = MagicMock()
            mock_ec2_class.return_value = mock_ec2_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                infrastructure_arn="i-1234567890abcdef0",
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(env_type=EnvType.TRAIN)

            # Verify start_environment was called on EC2 infra
            mock_ec2_instance.start_environment.assert_called_once()

    @patch("boto3.client")
    def test_start_environment_ecs_no_setup_call(self, mock_boto_client):
        """Test start_environment for ECS doesn't call setup (in container)."""
        mock_ecs = MagicMock()
        mock_boto_client.return_value = mock_ecs

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.ECSRFTInfrastructure"
        ) as mock_ecs_class:
            mock_ecs_instance = MagicMock()
            mock_ecs_instance.start_environment = MagicMock()
            mock_ecs_class.return_value = mock_ecs_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                infrastructure_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test",
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(env_type=EnvType.TRAIN)

            # Verify start_environment was called on ECS infra
            mock_ecs_instance.start_environment.assert_called_once()

    @patch("boto3.client")
    def test_start_environment_all_parameters(self, mock_boto_client):
        """Test start_environment with all parameters."""
        mock_boto_client.return_value = MagicMock()

        with patch(
            "amzn_nova_forge_sdk.rft_multiturn.rft_multiturn.LocalRFTInfrastructure"
        ) as mock_local:
            mock_local_instance = MagicMock()
            mock_local_instance.setup_local = MagicMock()
            mock_local_instance.install_local_environment = MagicMock()
            mock_local_instance.start_environment = MagicMock()
            mock_local.return_value = mock_local_instance

            rft = RFTMultiturnInfrastructure(
                stack_name="test-stack",
                vf_env_id=VFEnvId.WORDLE,
                python_venv_name="test_venv",
                rft_role_name="TestRole",
            )

            from amzn_nova_forge_sdk.rft_multiturn.base_infra import (
                StackOutputs,
            )

            rft.stack_outputs = StackOutputs(
                rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/response",
                rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/request",
                generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-req",
                generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/gen-resp",
                proxy_function_url="https://proxy.execute-api.us-east-1.amazonaws.com",
                dynamo_table_name="test-table",
            )

            rft.start_environment(
                env_type=EnvType.TRAIN,
                vf_env_args={"max_examples": 10},
                max_concurrent_rollouts=100,
                max_rollout_timeout=600.0,
                completion_poll_timeout=1200.0,
                completion_poll_interval=1.0,
                rollout_poll_interval=2.0,
                log_output_directory="/logs",
                config_name="test_config",
                config_path="/configs",
                queue_url="https://custom-queue",
            )

            call_args = mock_local_instance.start_environment.call_args
            assert call_args.kwargs["env_type"] == EnvType.TRAIN
            assert call_args.kwargs["vf_env_args"] == {"max_examples": 10}
            assert call_args.kwargs["max_concurrent_rollouts"] == 100
            assert call_args.kwargs["max_rollout_timeout"] == 600.0
            assert call_args.kwargs["completion_poll_timeout"] == 1200.0
            assert call_args.kwargs["completion_poll_interval"] == 1.0
            assert call_args.kwargs["rollout_poll_interval"] == 2.0
            assert call_args.kwargs["log_output_directory"] == "/logs"
            assert call_args.kwargs["config_name"] == "test_config"
            assert call_args.kwargs["config_path"] == "/configs"
