"""Unit tests for EC2RFTInfrastructure."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from amzn_nova_customization_sdk.rft_multiturn.base_infra import EnvType, StackOutputs
from amzn_nova_customization_sdk.rft_multiturn.ec2_infra import EC2RFTInfrastructure


class TestEC2RFTInfrastructure:
    """Test EC2RFTInfrastructure class."""

    def test_ec2_rft_infrastructure_exists(self):
        """Test that EC2RFTInfrastructure class is importable."""
        assert EC2RFTInfrastructure is not None

    def test_ec2_rft_infrastructure_has_required_methods(self):
        """Test that EC2RFTInfrastructure has expected methods."""
        assert hasattr(EC2RFTInfrastructure, "validate_platform")
        assert hasattr(EC2RFTInfrastructure, "deploy_sam_stack")
        assert hasattr(EC2RFTInfrastructure, "start_training_env")
        assert hasattr(EC2RFTInfrastructure, "start_evaluation_env")
        assert hasattr(EC2RFTInfrastructure, "get_logs")
        assert hasattr(EC2RFTInfrastructure, "kill_task")
        assert hasattr(EC2RFTInfrastructure, "cleanup")
        assert hasattr(EC2RFTInfrastructure, "ensure_rft_policy_on_current_role")
        assert hasattr(EC2RFTInfrastructure, "validate_starter_kit_access")
        assert hasattr(EC2RFTInfrastructure, "check_queue_messages")
        assert hasattr(EC2RFTInfrastructure, "flush_queue")

    def test_ec2_rft_infrastructure_is_class(self):
        """Test that EC2RFTInfrastructure is a class."""
        assert isinstance(EC2RFTInfrastructure, type)

    @patch("boto3.client")
    def test_initialization_with_instance_id(self, mock_boto_client):
        """Test EC2 infrastructure initialization with instance ID."""
        mock_boto_client.return_value = MagicMock()

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        assert infra.region == "us-east-1"
        assert infra.instance_id == "i-1234567890abcdef0"
        assert infra.python_venv_name == "test_venv"

    @patch("boto3.client")
    def test_initialization_with_arn(self, mock_boto_client):
        """Test EC2 infrastructure initialization with full ARN."""
        mock_boto_client.return_value = MagicMock()

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        assert infra.instance_id == "i-1234567890abcdef0"

    @patch("boto3.client")
    def test_extract_instance_id_from_short_id(self, mock_boto_client):
        """Test instance ID extraction from short format."""
        mock_boto_client.return_value = MagicMock()

        result = EC2RFTInfrastructure._extract_instance_id("i-1234567890abcdef0")
        assert result == "i-1234567890abcdef0"

    @patch("boto3.client")
    def test_extract_instance_id_from_arn(self, mock_boto_client):
        """Test instance ID extraction from ARN."""
        mock_boto_client.return_value = MagicMock()

        arn = "arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0"
        result = EC2RFTInfrastructure._extract_instance_id(arn)
        assert result == "i-1234567890abcdef0"

    @patch("boto3.client")
    def test_extract_instance_id_invalid_format(self, mock_boto_client):
        """Test instance ID extraction with invalid format."""
        mock_boto_client.return_value = MagicMock()

        with pytest.raises(ValueError, match="Invalid EC2 instance identifier"):
            EC2RFTInfrastructure._extract_instance_id("invalid-id")

    @patch("boto3.client")
    def test_validate_platform_success(self, mock_boto_client):
        """Test successful platform validation."""
        mock_ec2 = MagicMock()
        mock_ssm = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "State": {"Name": "running"},
                            "IamInstanceProfile": {
                                "Arn": "arn:aws:iam::123456789012:instance-profile/test-profile"
                            },
                        }
                    ]
                }
            ]
        }

        mock_ssm.describe_instance_information.return_value = {
            "InstanceInformationList": [{"PingStatus": "Online"}]
        }

        mock_iam.get_instance_profile.return_value = {
            "InstanceProfile": {"Roles": [{"RoleName": "test-role"}]}
        }

        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def mock_client(service, **kwargs):
            return {
                "ec2": mock_ec2,
                "ssm": mock_ssm,
                "iam": mock_iam,
                "sts": mock_sts,
                "cloudformation": MagicMock(),
                "sqs": MagicMock(),
                "logs": MagicMock(),
            }.get(service, MagicMock())

        mock_boto_client.side_effect = mock_client

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Mock the attach_rft_policy_to_role method
        with patch.object(infra, "attach_rft_policy_to_role"):
            infra.validate_platform()

        mock_ec2.describe_instances.assert_called_once()
        mock_ssm.describe_instance_information.assert_called_once()

    @patch("boto3.client")
    def test_validate_platform_instance_not_found(self, mock_boto_client):
        """Test platform validation when instance not found."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}
        mock_boto_client.return_value = mock_ec2

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="not found"):
            infra.validate_platform()

    @patch("boto3.client")
    def test_validate_platform_instance_not_running(self, mock_boto_client):
        """Test platform validation when instance is not running."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "stopped"}}]}]
        }
        mock_boto_client.return_value = mock_ec2

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="not running"):
            infra.validate_platform()

    @patch("boto3.client")
    def test_validate_platform_no_iam_profile(self, mock_boto_client):
        """Test platform validation when instance has no IAM profile."""
        mock_ec2 = MagicMock()
        mock_ssm = MagicMock()

        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}}]}]
        }

        mock_ssm.describe_instance_information.return_value = {
            "InstanceInformationList": [{"PingStatus": "Online"}]
        }

        def mock_client(service, **kwargs):
            return {"ec2": mock_ec2, "ssm": mock_ssm}.get(service, MagicMock())

        mock_boto_client.side_effect = mock_client

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="does not have an IAM instance profile"):
            infra.validate_platform()

    @patch("boto3.client")
    def test_validate_ssm_connectivity_offline(self, mock_boto_client):
        """Test SSM connectivity validation when agent is offline."""
        mock_ssm = MagicMock()
        mock_ssm.describe_instance_information.return_value = {
            "InstanceInformationList": [{"PingStatus": "ConnectionLost"}]
        }
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="SSM agent status"):
            infra._validate_ssm_connectivity()

    # Removed slow test_wait_for_ssm_command_success - takes 10+ seconds

    @patch("boto3.client")
    @patch("time.sleep")  # Mock sleep to avoid delays
    def test_wait_for_ssm_command_failure(self, mock_sleep, mock_boto_client):
        """Test waiting for SSM command that fails."""
        mock_ssm = MagicMock()
        mock_ssm.get_command_invocation.return_value = {"Status": "Failed"}

        # Properly mock the exceptions attribute
        InvocationDoesNotExist = type("InvocationDoesNotExist", (Exception,), {})
        mock_ssm.exceptions.InvocationDoesNotExist = InvocationDoesNotExist

        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(RuntimeError, match="SSM command failed"):
            infra._wait_for_ssm_command("cmd-123", timeout=10)

    @patch("boto3.client")
    @patch("time.sleep")
    def test_deploy_sam_stack(self, mock_sleep, mock_boto_client):
        """Test SAM stack deployment."""
        mock_ssm = MagicMock()
        mock_ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}
        mock_ssm.get_command_invocation.return_value = {"Status": "Success"}
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.deploy_sam_stack()

        mock_ssm.send_command.assert_called_once()
        assert mock_ssm.get_command_invocation.called

    @patch("boto3.client")
    def test_kill_task(self, mock_boto_client):
        """Test task killing."""
        mock_ssm = MagicMock()
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.kill_task(EnvType.TRAIN)

        mock_ssm.send_command.assert_called_once()
        call_args = mock_ssm.send_command.call_args
        assert "train.py" in call_args[1]["Parameters"]["commands"][0]

    @patch("boto3.client")
    def test_cleanup_without_environment(self, mock_boto_client):
        """Test cleanup without environment deletion."""
        mock_ssm = MagicMock()
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.cleanup(cleanup_environment=False)

        # Should call send_command twice (train and eval)
        assert mock_ssm.send_command.call_count == 2

    @patch("boto3.client")
    def test_cleanup_with_environment(self, mock_boto_client):
        """Test cleanup with environment deletion."""
        mock_ssm = MagicMock()
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.cleanup(cleanup_environment=True)

        # Should call send_command 3 times (train, eval, and cleanup)
        assert mock_ssm.send_command.call_count == 3

    @patch("boto3.client")
    def test_get_logs(self, mock_boto_client):
        """Test log retrieval."""
        mock_ssm = MagicMock()
        mock_ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}
        mock_ssm.get_command_invocation.return_value = {
            "StandardOutputContent": "log line 1\nlog line 2\nlog line 3"
        }
        mock_boto_client.return_value = mock_ssm

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        logs = infra.get_logs(
            env_type=EnvType.TRAIN,
            limit=100,
            start_from_head=False,
            log_stream_name=None,
        )

        assert len(logs) == 3
        assert logs[0] == "log line 1"
