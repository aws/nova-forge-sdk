"""Unit tests for EC2RFTInfrastructure."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.base_infra import EnvType, StackOutputs
from amzn_nova_forge_sdk.rft_multiturn.ec2_infra import EC2RFTInfrastructure


def setup_ami_validation_mocks(mock_ec2):
    """
    Helper function to setup AMI validation mocks for EC2 tests.
    This ensures all tests have consistent AMI data for the validation that happens in __init__.
    """
    # Setup describe_instances to return Amazon Linux AMI
    if not mock_ec2.describe_instances.return_value:
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "State": {"Name": "running"},
                            "ImageId": "ami-12345678",
                        }
                    ]
                }
            ]
        }

    # Setup describe_images to return Amazon Linux AMI details
    mock_ec2.describe_images.return_value = {
        "Images": [
            {
                "ImageId": "ami-12345678",
                "Name": "amzn2-ami-hvm-2.0.20230101-x86_64-gp2",
                "Description": "Amazon Linux 2 AMI",
            }
        ]
    }


def create_mock_boto_client_with_ami_validation(additional_mocks=None):
    """
    Create a mock boto3.client that returns properly configured mocks for EC2 and other services.

    Args:
        additional_mocks: Dict mapping service names to mock objects

    Returns:
        A side_effect function for boto3.client mock
    """
    mock_ec2 = MagicMock()
    setup_ami_validation_mocks(mock_ec2)

    service_mocks = {"ec2": mock_ec2}
    if additional_mocks:
        service_mocks.update(additional_mocks)

    def get_client(service_name, **kwargs):
        return service_mocks.get(service_name, MagicMock())

    return get_client, service_mocks


class TestEC2RFTInfrastructure:
    """Test EC2RFTInfrastructure class."""

    @pytest.fixture
    def mock_ec2_client(self):
        """Fixture to provide a properly mocked EC2 client with AMI validation."""
        mock_ec2 = MagicMock()
        setup_ami_validation_mocks(mock_ec2)
        return mock_ec2

    def test_ec2_rft_infrastructure_exists(self):
        """Test that EC2RFTInfrastructure class is importable."""
        assert EC2RFTInfrastructure is not None

    def test_ec2_rft_infrastructure_has_required_methods(self):
        """Test that EC2RFTInfrastructure has expected methods."""
        assert hasattr(EC2RFTInfrastructure, "validate_platform")
        assert hasattr(EC2RFTInfrastructure, "deploy_sam_stack")
        assert hasattr(EC2RFTInfrastructure, "start_environment")
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
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Setup AMI validation mocks
        setup_ami_validation_mocks(mock_ec2)

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
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Setup AMI validation mocks
        setup_ami_validation_mocks(mock_ec2)

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
                            "ImageId": "ami-12345678",
                            "IamInstanceProfile": {
                                "Arn": "arn:aws:iam::123456789012:instance-profile/test-profile"
                            },
                        }
                    ]
                }
            ]
        }

        mock_ec2.describe_images.return_value = {
            "Images": [
                {
                    "ImageId": "ami-12345678",
                    "Name": "amzn2-ami-hvm-2.0.20230101-x86_64-gp2",
                    "Description": "Amazon Linux 2 AMI",
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

        # describe_instances is called twice: once in __init__ for AMI validation, once in validate_platform
        assert mock_ec2.describe_instances.call_count == 2
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
        mock_ec2 = MagicMock()
        mock_ssm = MagicMock()

        # Setup AMI validation mocks for EC2
        setup_ami_validation_mocks(mock_ec2)

        # Setup SSM mock
        mock_ssm.describe_instance_information.return_value = {
            "InstanceInformationList": [{"PingStatus": "ConnectionLost"}]
        }

        # Return appropriate client based on service name
        def get_client(service_name, **kwargs):
            if service_name == "ec2":
                return mock_ec2
            elif service_name == "ssm":
                return mock_ssm
            return MagicMock()

        mock_boto_client.side_effect = get_client

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
        mock_ec2 = MagicMock()
        mock_ssm = MagicMock()

        # Setup AMI validation mocks
        setup_ami_validation_mocks(mock_ec2)

        # Setup SSM mock
        mock_ssm.get_command_invocation.return_value = {"Status": "Failed"}

        # Properly mock the exceptions attribute
        InvocationDoesNotExist = type("InvocationDoesNotExist", (Exception,), {})
        mock_ssm.exceptions.InvocationDoesNotExist = InvocationDoesNotExist

        # Return appropriate client based on service name
        def get_client(service_name, **kwargs):
            if service_name == "ec2":
                return mock_ec2
            elif service_name == "ssm":
                return mock_ssm
            return MagicMock()

        mock_boto_client.side_effect = get_client

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

        get_client, mocks = create_mock_boto_client_with_ami_validation(
            {"ssm": mock_ssm}
        )
        mock_boto_client.side_effect = get_client

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

        get_client, mocks = create_mock_boto_client_with_ami_validation(
            {"ssm": mock_ssm}
        )
        mock_boto_client.side_effect = get_client

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            session_id="test1234",
        )

        infra.kill_task(EnvType.TRAIN)

        mock_ssm.send_command.assert_called_once()
        call_args = mock_ssm.send_command.call_args
        # Check for session-based script name (new pattern: {session_id}_{env_type}.sh)
        assert "test1234_train.sh" in call_args[1]["Parameters"]["commands"][0]

    @patch("boto3.client")
    def test_cleanup_without_environment(self, mock_boto_client):
        """Test cleanup without environment deletion."""
        mock_ssm = MagicMock()

        get_client, mocks = create_mock_boto_client_with_ami_validation(
            {"ssm": mock_ssm}
        )
        mock_boto_client.side_effect = get_client

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

        get_client, mocks = create_mock_boto_client_with_ami_validation(
            {"ssm": mock_ssm}
        )
        mock_boto_client.side_effect = get_client

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

    @patch("time.sleep")  # Mock sleep to speed up test
    @patch("boto3.client")
    def test_get_logs(self, mock_boto_client, mock_sleep):
        """Test log retrieval."""
        mock_ssm = MagicMock()

        # Mock the exceptions attribute
        mock_invocation_exception = type("InvocationDoesNotExist", (Exception,), {})
        mock_ssm.exceptions.InvocationDoesNotExist = mock_invocation_exception

        mock_ssm.send_command.return_value = {"Command": {"CommandId": "cmd-123"}}
        mock_ssm.get_command_invocation.return_value = {
            "Status": "Success",
            "StandardOutputContent": "log line 1\nlog line 2\nlog line 3",
        }

        get_client, mocks = create_mock_boto_client_with_ami_validation(
            {"ssm": mock_ssm}
        )
        mock_boto_client.side_effect = get_client

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            session_id="test1234",
        )

        logs = infra.get_logs(
            env_type=EnvType.TRAIN,
            limit=100,
            start_from_head=False,
            log_stream_name=None,
        )

        assert len(logs) == 3
        assert logs[0] == "log line 1"

    @patch("boto3.client")
    def test_get_state(self, mock_boto_client):
        """Test get_state captures EC2 platform state."""
        get_client, mocks = create_mock_boto_client_with_ami_validation()
        mock_boto_client.side_effect = get_client

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra.base_path = "/home/ec2-user/v1"
        infra.starter_kit_s3 = "s3://bucket/starter-kit.tar.gz"

        state = infra.get_state()

        assert state["instance_id"] == "i-1234567890abcdef0"
        assert state["python_venv_name"] == "test_venv"
        assert state["base_path"] == "/home/ec2-user/v1"
        assert state["starter_kit_s3"] == "s3://bucket/starter-kit.tar.gz"

    @patch("boto3.client")
    def test_restore_state_success(self, mock_boto_client):
        """Test restore_state with running instance."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "State": {"Name": "running"},
                            "ImageId": "ami-12345678",
                        }
                    ]
                }
            ]
        }
        mock_ec2.describe_images.return_value = {
            "Images": [
                {
                    "ImageId": "ami-12345678",
                    "Name": "amzn2-ami-hvm-2.0.20230101-x86_64-gp2",
                    "Description": "Amazon Linux 2 AMI",
                }
            ]
        }
        mock_boto_client.return_value = mock_ec2

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "instance_id": "i-1234567890abcdef0",
            "python_venv_name": "test_venv",
            "base_path": "/home/ec2-user/v1",
            "starter_kit_s3": "s3://bucket/custom-kit.tar.gz",
        }

        infra.restore_state(state)

        assert infra.starter_kit_s3 == "s3://bucket/custom-kit.tar.gz"
        # describe_instances is called twice: once in __init__ for AMI validation, once in restore_state
        assert mock_ec2.describe_instances.call_count == 2

    @patch("boto3.client")
    def test_restore_state_stopped_instance(self, mock_boto_client):
        """Test restore_state with stopped instance."""
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

        state = {
            "instance_id": "i-1234567890abcdef0",
            "starter_kit_s3": "s3://bucket/kit.tar.gz",
        }

        # Should not raise, just log warning
        infra.restore_state(state)

        assert infra.starter_kit_s3 == "s3://bucket/kit.tar.gz"

    @patch("boto3.client")
    def test_restore_state_missing_instance(self, mock_boto_client):
        """Test restore_state with missing instance."""
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

        state = {
            "instance_id": "i-1234567890abcdef0",
            "starter_kit_s3": "s3://bucket/kit.tar.gz",
        }

        # Should not raise, just log warning
        infra.restore_state(state)

    @patch("boto3.client")
    def test_restore_state_api_error(self, mock_boto_client):
        """Test restore_state handles API errors gracefully."""
        mock_ec2 = MagicMock()
        setup_ami_validation_mocks(mock_ec2)

        # After initialization, set up the API error for restore_state
        def describe_instances_side_effect(*args, **kwargs):
            # First call during __init__ succeeds
            if not hasattr(describe_instances_side_effect, "called"):
                describe_instances_side_effect.called = True
                return mock_ec2.describe_instances.return_value
            # Subsequent calls fail
            raise Exception("API Error")

        mock_ec2.describe_instances.side_effect = describe_instances_side_effect
        mock_boto_client.return_value = mock_ec2

        infra = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {"instance_id": "i-1234567890abcdef0"}

        # Should not raise, just log warning
        infra.restore_state(state)

    @patch("boto3.client")
    def test_get_state_restore_state_roundtrip(self, mock_boto_client):
        """Test get_state and restore_state roundtrip."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}}]}]
        }
        mock_boto_client.return_value = mock_ec2

        infra1 = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra1.starter_kit_s3 = "s3://bucket/kit.tar.gz"

        state = infra1.get_state()

        infra2 = EC2RFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            instance_arn="i-1234567890abcdef0",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra2.restore_state(state)

        assert infra2.starter_kit_s3 == infra1.starter_kit_s3
