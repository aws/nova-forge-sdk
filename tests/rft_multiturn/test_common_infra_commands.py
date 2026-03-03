"""Unit tests for CommonInfraCommands."""

import json
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.common_infra_commands import (
    BASE_PYTHON_COMMAND,
    CommonInfraCommands,
)
from amzn_nova_forge_sdk.validation.rft_multiturn_validator import (
    validate_env_id,
)


class MockCommonInfraCommands(CommonInfraCommands):
    """Mock implementation for testing abstract methods."""

    def __init__(self):
        self.base_path = "/test/path"
        self.python_venv_name = "test_venv"
        self.starter_kit_s3 = "s3://test-starter-kit"
        self.region = "us-west-2"
        self.custom_env = None
        self.sam_base_dir = "/test/sam"
        self.sam_log_file = "/test/sam.log"
        self.stack_name = "test-stack"

    def _get_package_install_cmd(self):
        return ["apt-get update", "apt-get install -y git"]


class TestCommonInfraCommands:
    """Test CommonInfraCommands class."""

    def test_common_infra_commands_exists(self):
        """Test that CommonInfraCommands class is importable."""
        assert CommonInfraCommands is not None

    def test_validate_env_id_accepts_valid_ids(self):
        """Test that validate_env_id accepts valid environment IDs."""
        valid_ids = [
            "wordle",
            "terminalbench_env",
            "my-custom-env",
            "env123",
            "test_env-123",
        ]
        for env_id in valid_ids:
            # Should not raise
            validate_env_id(env_id)

    def test_validate_env_id_rejects_invalid_ids(self):
        """Test that validate_env_id rejects invalid environment IDs."""
        invalid_ids = [
            "env with spaces",
            "env;rm -rf /",
            "env$(whoami)",
            "env`ls`",
            "env|cat",
            "env&echo",
            "env>file",
            "env<file",
            "env'test",
            'env"test',
            "env\\test",
            "env/test",
        ]
        for env_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid environment ID"):
                validate_env_id(env_id)

    def test_validate_env_id_prevents_shell_injection(self):
        """Test that validate_env_id prevents shell injection attempts."""
        injection_attempts = [
            "; rm -rf /",
            "$(malicious_command)",
            "`malicious_command`",
            "| cat /etc/passwd",
            "&& echo hacked",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_env_id(attempt)

    def test_build_setup_commands_basic(self):
        """Test _build_setup_commands generates correct commands."""
        mock_cmd = MockCommonInfraCommands()
        commands = mock_cmd._build_setup_commands("test_env")

        assert isinstance(commands, list)
        assert len(commands) > 0
        # Verify package install commands are included
        assert "apt-get update" in commands
        # Verify venv creation
        assert any("venv" in cmd for cmd in commands)
        # Verify git clone
        assert any("git clone" in cmd for cmd in commands)

    def test_build_setup_commands_validates_env_id(self):
        """Test _build_setup_commands validates environment ID."""
        mock_cmd = MockCommonInfraCommands()
        with pytest.raises(ValueError, match="Invalid environment ID"):
            mock_cmd._build_setup_commands("invalid;env")

    def test_build_setup_commands_with_custom_base_path(self):
        """Test _build_setup_commands with custom base path."""
        mock_cmd = MockCommonInfraCommands()
        custom_path = "/custom/path"
        commands = mock_cmd._build_setup_commands("test_env", base_path=custom_path)

        # Verify custom path is used
        assert any(custom_path in cmd for cmd in commands)

    def test_build_setup_commands_with_custom_env_s3(self):
        """Test _build_setup_commands with custom environment from S3."""
        mock_cmd = MockCommonInfraCommands()
        mock_env = MagicMock()
        mock_env.s3_uri = "s3://custom-env/env.tar.gz"
        mock_env.local_path = None
        mock_cmd.custom_env = mock_env

        commands = mock_cmd._build_setup_commands("custom_env")

        # Verify S3 download commands
        assert any("aws s3 cp" in cmd for cmd in commands)
        assert any("tar -xzf" in cmd for cmd in commands)

    def test_build_setup_commands_with_custom_env_local(self):
        """Test _build_setup_commands with custom environment from local path."""
        mock_cmd = MockCommonInfraCommands()
        mock_env = MagicMock()
        mock_env.s3_uri = None
        mock_env.local_path = "/local/custom/env"
        mock_cmd.custom_env = mock_env

        commands = mock_cmd._build_setup_commands("custom_env")

        # Verify local path installation
        assert any("/local/custom/env" in cmd for cmd in commands)

    def test_build_sam_deploy_commands(self):
        """Test _build_sam_deploy_commands generates correct commands."""
        mock_cmd = MockCommonInfraCommands()
        commands = mock_cmd._build_sam_deploy_commands()

        assert isinstance(commands, list)
        assert len(commands) > 0
        # Verify package install
        assert "apt-get update" in commands
        # Verify SAM commands
        assert any("sam build" in cmd for cmd in commands)
        assert any("sam deploy" in cmd for cmd in commands)
        # Verify log redirection
        assert any(mock_cmd.sam_log_file in cmd for cmd in commands)

    def test_build_sam_deploy_commands_with_stack_name(self):
        """Test _build_sam_deploy_commands includes stack name."""
        mock_cmd = MockCommonInfraCommands()
        mock_cmd.stack_name = "custom-stack-name"
        commands = mock_cmd._build_sam_deploy_commands()

        # Verify stack name is used
        assert any("custom-stack-name" in cmd for cmd in commands)

    def test_build_sam_deploy_commands_without_log_file(self):
        """Test _build_sam_deploy_commands without log file (ECS mode)."""
        mock_cmd = MockCommonInfraCommands()
        mock_cmd.sam_log_file = None
        commands = mock_cmd._build_sam_deploy_commands()

        # Verify no log redirection when log_file is None
        assert not any(">>" in cmd or "> " in cmd for cmd in commands if "sam" in cmd)

    def test_get_package_install_cmd_not_implemented(self):
        """Test _get_package_install_cmd raises NotImplementedError."""
        cmd = CommonInfraCommands()
        with pytest.raises(NotImplementedError):
            cmd._get_package_install_cmd()

    def test_base_python_command_constant(self):
        """Test BASE_PYTHON_COMMAND constant is defined."""
        assert BASE_PYTHON_COMMAND == "python3.12"
