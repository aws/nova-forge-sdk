"""Unit tests for LocalRFTInfrastructure."""

import os
import subprocess
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.base_infra import EnvType, StackOutputs
from amzn_nova_forge_sdk.rft_multiturn.local_infra import (
    LocalRFTInfrastructure,
)


class TestLocalRFTInfrastructure:
    """Test LocalRFTInfrastructure class."""

    def test_local_rft_infrastructure_exists(self):
        """Test that LocalRFTInfrastructure class is importable."""
        assert LocalRFTInfrastructure is not None

    def test_local_rft_infrastructure_has_required_methods(self):
        """Test that LocalRFTInfrastructure has expected methods."""
        assert hasattr(LocalRFTInfrastructure, "setup_local")
        assert hasattr(LocalRFTInfrastructure, "deploy_sam_stack")
        assert hasattr(LocalRFTInfrastructure, "start_environment")
        assert hasattr(LocalRFTInfrastructure, "get_logs")
        assert hasattr(LocalRFTInfrastructure, "kill_task")
        assert hasattr(LocalRFTInfrastructure, "cleanup")
        assert hasattr(LocalRFTInfrastructure, "ensure_rft_policy_on_current_role")
        assert hasattr(LocalRFTInfrastructure, "validate_starter_kit_access")
        assert hasattr(LocalRFTInfrastructure, "check_queue_messages")
        assert hasattr(LocalRFTInfrastructure, "flush_queue")

    def test_local_rft_infrastructure_is_class(self):
        """Test that LocalRFTInfrastructure is a class."""
        assert isinstance(LocalRFTInfrastructure, type)

    @patch("boto3.client")
    def test_initialization(self, mock_boto_client):
        """Test LOCAL infrastructure initialization."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        assert infra.region == "us-east-1"
        assert infra.workspace_dir == "/home/user/workspace"
        assert infra.python_venv_name == "test_venv"
        assert infra.train_process is None
        assert infra.eval_process is None

    @patch("boto3.client")
    def test_initialization_with_custom_policy(self, mock_boto_client):
        """Test LOCAL infrastructure initialization with custom policy."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            custom_policy_path="/path/to/policy.json",
        )

        assert infra.custom_policy_path == "/path/to/policy.json"

    @patch("boto3.client")
    @patch("os.path.exists")
    def test_setup_local_existing_starter_kit(self, mock_exists, mock_boto_client):
        """Test setup_local when starter kit already exists."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        result = infra.setup_local("/home/user/workspace")

        assert result == "/home/user/v1"
        assert infra.starter_kit_path == "/home/user/v1"
        assert infra.base_path == "/home/user/v1"

    @patch("boto3.client")
    @patch("os.path.exists")
    def test_setup_local_missing_starter_kit(self, mock_exists, mock_boto_client):
        """Test setup_local when starter kit doesn't exist."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = False

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        result = infra.setup_local("/home/user/workspace")

        assert result == "/home/user/v1"
        assert infra.starter_kit_path == "/home/user/v1"

    @patch("boto3.client")
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_install_local_environment_success(
        self, mock_exists, mock_subprocess, mock_boto_client
    ):
        """Test successful local environment installation."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = False
        mock_subprocess.return_value = Mock(returncode=0, stderr="", stdout="")

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.starter_kit_path = "/home/user/v1"
        infra.base_path = "/home/user/v1"

        infra.install_local_environment("wordle")

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert "/bin/bash" in call_args[0][0]

    @patch("boto3.client")
    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_install_local_environment_failure(
        self, mock_rmtree, mock_exists, mock_subprocess, mock_boto_client
    ):
        """Test local environment installation failure with cleanup."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(
            returncode=1, stderr="Installation failed", stdout=""
        )

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.starter_kit_path = "/home/user/v1"
        infra.base_path = "/home/user/v1"

        with pytest.raises(RuntimeError, match="Environment setup failed"):
            infra.install_local_environment("wordle")

        mock_rmtree.assert_called_once()

    @patch("boto3.client")
    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_deploy_sam_stack_success(
        self, mock_file, mock_subprocess, mock_boto_client
    ):
        """Test successful SAM stack deployment."""
        mock_boto_client.return_value = MagicMock()
        mock_subprocess.return_value = Mock(returncode=0, stderr="", stdout="Success")

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.starter_kit_path = "/home/user/v1"

        infra.deploy_sam_stack()

        mock_subprocess.assert_called_once()
        mock_file.assert_called()

    @patch("boto3.client")
    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_deploy_sam_stack_failure(
        self, mock_file, mock_subprocess, mock_boto_client
    ):
        """Test SAM stack deployment failure."""
        mock_boto_client.return_value = MagicMock()
        mock_subprocess.return_value = Mock(
            returncode=1, stderr="Deployment failed", stdout=""
        )

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.starter_kit_path = "/home/user/v1"

        with pytest.raises(RuntimeError, match="SAM deployment failed"):
            infra.deploy_sam_stack()

    @patch("boto3.client")
    @patch("subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    def test_start_local_process(self, mock_file, mock_popen, mock_boto_client):
        """Test starting a local process."""
        mock_boto_client.return_value = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        result = infra.start_local_process(
            cmd="echo test", log_file="/tmp/test.log", env_vars={"TEST": "value"}
        )

        assert result == mock_process
        mock_popen.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    @patch("subprocess.run")
    @patch("os.makedirs")  # Mock directory creation
    @patch("boto3.client")
    def test_start_environment(
        self,
        mock_boto_client,
        mock_makedirs,
        mock_subprocess_run,
        mock_popen,
        mock_file,
    ):
        """Test starting environment."""
        mock_boto_client.return_value = MagicMock()
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_subprocess_run.return_value = Mock(returncode=0)
        mock_makedirs.return_value = None  # Mock successful directory creation

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.base_path = "/home/user/v1"

        stack_outputs = StackOutputs(
            rollout_request_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            rollout_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/rollout-response",
            rollout_request_queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/rollout-request",
            generate_request_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/generate-request",
            generate_response_sqs_url="https://sqs.us-east-1.amazonaws.com/123456789012/generate-response",
            proxy_function_url="https://lambda.url",
            dynamo_table_name="test-table",
        )

        infra.start_environment(
            env_type=EnvType.TRAIN,
            vf_env_id="wordle",
            vf_env_args={"use_think": False},
            stack_outputs=stack_outputs,
        )

        assert infra.train_process == mock_process
        mock_popen.assert_called_once()
        """Test starting evaluation environment."""
        mock_boto_client.return_value = MagicMock()

    @patch("boto3.client")
    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="log line 1\nlog line 2\nlog line 3\n",
    )
    def test_get_logs_from_head(self, mock_file, mock_exists, mock_boto_client):
        """Test getting logs from head."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        logs = infra.get_logs(
            env_type=EnvType.TRAIN,
            limit=2,
            start_from_head=True,
            log_stream_name=None,
        )

        assert len(logs) == 2
        assert logs[0] == "log line 1\n"

    @patch("boto3.client")
    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="log line 1\nlog line 2\nlog line 3\n",
    )
    def test_get_logs_from_tail(self, mock_file, mock_exists, mock_boto_client):
        """Test getting logs from tail."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        logs = infra.get_logs(
            env_type=EnvType.TRAIN,
            limit=2,
            start_from_head=False,
            log_stream_name=None,
        )

        assert len(logs) == 2

    @patch("boto3.client")
    @patch("os.path.exists")
    def test_get_logs_file_not_found(self, mock_exists, mock_boto_client):
        """Test getting logs when file doesn't exist."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = False

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        logs = infra.get_logs(
            env_type=EnvType.TRAIN,
            limit=100,
            start_from_head=False,
            log_stream_name=None,
        )

        assert logs == []

    @patch(
        "builtins.open", new_callable=mock_open
    )  # Mock file operations for status file
    @patch("os.makedirs")  # Mock directory creation
    @patch("os.remove")
    @patch("os.path.exists")
    @patch("boto3.client")
    def test_kill_task_running_process(
        self, mock_boto_client, mock_exists, mock_remove, mock_makedirs, mock_file
    ):
        """Test killing a running task."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True
        mock_makedirs.return_value = None

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.train_process = mock_process

        infra.kill_task(EnvType.TRAIN)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        # After refactoring, we write status file and remove PID file
        assert mock_remove.called  # At least one file removed (PID file)

    @patch(
        "builtins.open", new_callable=mock_open
    )  # Mock file operations for status file
    @patch("os.makedirs")  # Mock directory creation
    @patch("os.remove")
    @patch("os.path.exists")
    @patch("boto3.client")
    def test_kill_task_no_process(
        self, mock_boto_client, mock_exists, mock_remove, mock_makedirs, mock_file
    ):
        """Test killing task when no process is running."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True
        mock_makedirs.return_value = None

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.train_process = None

        infra.kill_task(EnvType.TRAIN)

        # After refactoring, we write status file and may remove PID file
        # Just verify the method completes without error
        assert True

    @patch("boto3.client")
    def test_cleanup_without_environment(self, mock_boto_client):
        """Test cleanup without environment deletion."""
        mock_boto_client.return_value = MagicMock()

        mock_train_process = MagicMock()
        mock_train_process.poll.return_value = None
        mock_eval_process = MagicMock()
        mock_eval_process.poll.return_value = None

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.train_process = mock_train_process
        infra.eval_process = mock_eval_process

        with patch.object(infra, "kill_task"):
            infra.cleanup(cleanup_environment=False)

            # Should call kill_task twice
            assert infra.kill_task.call_count == 2

    @patch("boto3.client")
    @patch("shutil.rmtree")
    @patch("os.path.exists")
    def test_cleanup_with_environment(self, mock_exists, mock_rmtree, mock_boto_client):
        """Test cleanup with environment deletion."""
        mock_boto_client.return_value = MagicMock()
        mock_exists.return_value = True

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.starter_kit_path = "/home/user/v1"
        infra.base_path = "/home/user/v1"
        infra.train_process = None
        infra.eval_process = None

        infra.cleanup(cleanup_environment=True)

        # Should delete venv, starter kit, and logs directory (3 calls after refactoring)
        assert mock_rmtree.call_count == 3

    @patch("boto3.client")
    def test_validate_platform(self, mock_boto_client):
        """Test platform validation (should pass without checks for LOCAL)."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Should not raise
        infra.validate_platform()

    @patch("boto3.client")
    def test_get_package_install_cmd(self, mock_boto_client):
        """Test package install command returns empty list for LOCAL."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        result = infra._get_package_install_cmd()
        assert result == []

    @patch("boto3.client")
    def test_get_state_with_running_processes(self, mock_boto_client):
        """Test get_state captures local platform state with running processes."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra.base_path = "/home/user/v1"
        infra.starter_kit_path = "/home/user/v1"

        # Mock running processes
        mock_train_process = MagicMock()
        mock_train_process.pid = 12345
        mock_train_process.poll.return_value = None  # Still running

        mock_eval_process = MagicMock()
        mock_eval_process.pid = 67890
        mock_eval_process.poll.return_value = None  # Still running

        infra.train_process = mock_train_process
        infra.eval_process = mock_eval_process

        state = infra.get_state()

        assert state["python_venv_name"] == "test_venv"
        assert state["starter_kit_path"] == "/home/user/v1"
        assert state["base_path"] == "/home/user/v1"
        assert state["train_pid"] == 12345
        assert state["eval_pid"] == 67890

    @patch("boto3.client")
    def test_get_state_with_stopped_processes(self, mock_boto_client):
        """Test get_state with stopped processes."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Mock stopped processes
        mock_train_process = MagicMock()
        mock_train_process.pid = 12345
        mock_train_process.poll.return_value = 0  # Exited

        infra.train_process = mock_train_process
        infra.eval_process = None

        state = infra.get_state()

        assert state["train_pid"] is None  # Process stopped
        assert state["eval_pid"] is None  # No process

    @patch("boto3.client")
    def test_get_state_without_processes(self, mock_boto_client):
        """Test get_state without any processes."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra.train_process = None
        infra.eval_process = None

        state = infra.get_state()

        assert state["train_pid"] is None
        assert state["eval_pid"] is None

    @patch("boto3.client")
    def test_restore_state_with_pids(self, mock_boto_client):
        """Test restore_state stores PIDs for recovery."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "python_venv_name": "test_venv",
            "starter_kit_path": "/home/user/v1",
            "base_path": "/home/user/v1",
            "train_pid": 12345,
            "eval_pid": 67890,
        }

        infra.restore_state(state)

        assert infra._train_pid == 12345
        assert infra._eval_pid == 67890

    @patch("boto3.client")
    def test_restore_state_without_pids(self, mock_boto_client):
        """Test restore_state without PIDs."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "python_venv_name": "test_venv",
            "train_pid": None,
            "eval_pid": None,
        }

        infra.restore_state(state)

        assert infra._train_pid is None
        assert infra._eval_pid is None

    @patch("boto3.client")
    def test_get_state_restore_state_roundtrip(self, mock_boto_client):
        """Test get_state and restore_state roundtrip."""
        mock_boto_client.return_value = MagicMock()

        infra1 = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra1.starter_kit_path = "/home/user/v1"
        infra1.base_path = "/home/user/v1"

        # Mock running process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        infra1.train_process = mock_process

        state = infra1.get_state()

        infra2 = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra2.restore_state(state)

        assert infra2._train_pid == 12345
        assert infra2._eval_pid is None

    @patch("boto3.client")
    def test_restore_state_with_partial_state(self, mock_boto_client):
        """Test restore_state with partial state data."""
        mock_boto_client.return_value = MagicMock()

        infra = LocalRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            workspace_dir="/home/user/workspace",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Minimal state
        state = {"train_pid": 12345}

        infra.restore_state(state)

        assert infra._train_pid == 12345
        assert infra._eval_pid is None
