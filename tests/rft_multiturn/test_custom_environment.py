"""Tests for custom environment functionality."""

from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.custom_environment import (
    MULTI_TURN_TEMPLATE,
    PYPROJECT_TEMPLATE,
    README_TEMPLATE,
    SINGLE_TURN_TEMPLATE,
    CustomEnvironment,
)


class TestCustomEnvironmentTemplates:
    """Test custom environment templates."""

    def test_single_turn_template_structure(self):
        """Test single turn template has required content."""
        assert "def load_environment" in SINGLE_TURN_TEMPLATE
        assert "vf.Environment" in SINGLE_TURN_TEMPLATE

    def test_multi_turn_template_structure(self):
        """Test multi turn template has required content."""
        assert "def load_environment" in MULTI_TURN_TEMPLATE
        assert "vf.Environment" in MULTI_TURN_TEMPLATE

    def test_pyproject_template_structure(self):
        """Test pyproject template has required content."""
        assert "[project]" in PYPROJECT_TEMPLATE

    def test_readme_template_exists(self):
        """Test README template is not empty."""
        assert len(README_TEMPLATE) > 0


class TestCustomEnvironmentClass:
    """Test CustomEnvironment class."""

    def test_custom_environment_has_create_method(self):
        """Test that CustomEnvironment has create method."""
        assert hasattr(CustomEnvironment, "create")
        assert callable(getattr(CustomEnvironment, "create"))

    def test_custom_environment_initialization(self):
        """Test CustomEnvironment initialization."""
        env = CustomEnvironment(
            env_id="test_env",
            local_path="/path/to/env",
            s3_uri="s3://bucket/env.tar.gz",
            output_dir="~/custom_envs",
            env_type="single_turn",
        )
        assert env.env_id == "test_env"
        assert env.local_path == "/path/to/env"
        assert env.s3_uri == "s3://bucket/env.tar.gz"
        assert env.output_dir == "~/custom_envs"
        assert env.env_type == "single_turn"

    def test_custom_environment_defaults(self):
        """Test CustomEnvironment default values."""
        env = CustomEnvironment(env_id="test_env")
        assert env.local_path is None
        assert env.s3_uri is None
        assert env.output_dir is None
        assert env.env_type == "single_turn"

    def test_create_fails_without_output_dir(self):
        """Test create fails when output_dir is not provided."""
        env = CustomEnvironment(env_id="test_env")
        with pytest.raises(ValueError, match="output_dir is required"):
            env.create()

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_create_single_turn_environment(
        self, mock_open, mock_exists, mock_makedirs
    ):
        """Test creating a single turn environment."""
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        env = CustomEnvironment(
            env_id="test_env", env_type="single_turn", output_dir="~/custom_envs"
        )
        result = env.create()

        assert result == env
        assert mock_makedirs.called
        assert (
            mock_open.call_count >= 3
        )  # __init__.py, env file, pyproject.toml, README

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_create_multi_turn_environment(self, mock_open, mock_exists, mock_makedirs):
        """Test creating a multi turn environment."""
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        env = CustomEnvironment(
            env_id="test_env", env_type="multi_turn", output_dir="~/custom_envs"
        )
        result = env.create()

        assert result == env
        assert mock_makedirs.called

    @patch("os.path.exists")
    def test_create_fails_if_exists_without_overwrite(self, mock_exists):
        """Test create fails if directory exists and overwrite=False."""
        mock_exists.return_value = True

        env = CustomEnvironment(env_id="test_env", output_dir="~/custom_envs")
        with pytest.raises(FileExistsError, match="already exists"):
            env.create(overwrite=False)

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_create_overwrites_if_requested(
        self, mock_open, mock_exists, mock_makedirs
    ):
        """Test create overwrites existing directory if overwrite=True."""
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        env = CustomEnvironment(env_id="test_env", output_dir="~/custom_envs")
        result = env.create(overwrite=True)

        assert result == env
        assert mock_makedirs.called

    def test_create_fails_with_invalid_env_type(self):
        """Test create fails with invalid env_type."""
        env = CustomEnvironment(
            env_id="test_env", env_type="invalid", output_dir="~/custom_envs"
        )
        with pytest.raises(ValueError, match="env_type must be"):
            env.create()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_load_existing_environment(self, mock_isdir, mock_exists):
        """Test loading an existing environment."""
        mock_exists.return_value = True
        mock_isdir.return_value = True

        env = CustomEnvironment(env_id="test-env", local_path="/base/path")
        result = env.load()

        assert result == env
        assert env.local_path is not None

    def test_load_fails_without_local_path(self):
        """Test load fails without local_path."""
        env = CustomEnvironment(env_id="test_env")
        with pytest.raises(ValueError, match="local_path required"):
            env.load()

    @patch("os.path.exists")
    def test_load_fails_if_not_exists(self, mock_exists):
        """Test load fails if environment doesn't exist."""
        mock_exists.return_value = False

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="does not exist"):
            env.load()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_load_fails_if_not_directory(self, mock_isdir, mock_exists):
        """Test load fails if path is not a directory."""
        mock_exists.return_value = True
        mock_isdir.return_value = False

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="not a directory"):
            env.load()

    def test_validate_fails_without_local_path(self):
        """Test validate fails without local_path."""
        env = CustomEnvironment(env_id="test_env")
        with pytest.raises(ValueError, match="local_path required"):
            env.validate()

    @patch("os.path.exists")
    def test_validate_fails_if_not_exists(self, mock_exists):
        """Test validate fails if path doesn't exist."""
        mock_exists.return_value = False

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="does not exist"):
            env.validate()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_validate_fails_if_not_directory(self, mock_isdir, mock_exists):
        """Test validate fails if path is not a directory."""
        mock_exists.side_effect = [True, False]  # First call True, second False
        mock_isdir.return_value = False

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="not a directory"):
            env.validate()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.listdir")
    def test_validate_fails_without_pyproject(
        self, mock_listdir, mock_isdir, mock_exists
    ):
        """Test validate fails without pyproject.toml."""

        # First two calls for path validation, third for pyproject check
        def exists_side_effect(path):
            if "pyproject.toml" in path:
                return False
            return True

        mock_exists.side_effect = exists_side_effect
        mock_isdir.return_value = True

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="Missing pyproject.toml"):
            env.validate()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("builtins.open", create=True)
    def test_validate_fails_without_load_environment(
        self, mock_open, mock_listdir, mock_isdir, mock_exists
    ):
        """Test validate fails without load_environment function."""
        mock_exists.side_effect = lambda p: "pyproject.toml" in p or "/path" in p
        mock_isdir.return_value = True
        # Return empty list for all listdir calls to simulate no Python files found
        mock_listdir.return_value = []

        mock_file = MagicMock()
        mock_file.read.return_value = "# No load_environment here"
        mock_open.return_value.__enter__.return_value = mock_file

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="No load_environment"):
            env.validate()

    @patch("os.path.exists")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("builtins.open", create=True)
    def test_validate_succeeds_with_valid_structure(
        self, mock_open, mock_listdir, mock_isdir, mock_exists
    ):
        """Test validate succeeds with valid environment structure."""
        mock_exists.side_effect = lambda p: "pyproject.toml" in p or "/path" in p
        mock_isdir.return_value = True
        mock_listdir.side_effect = [["subdir"], ["test.py"]]

        mock_file = MagicMock()
        mock_file.read.return_value = "def load_environment():\n    pass"
        mock_open.return_value.__enter__.return_value = mock_file

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        result = env.validate()

        assert result is True

    @patch("boto3.client")
    @patch("boto3.Session")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch.object(CustomEnvironment, "validate")
    def test_package_and_upload_with_default_bucket(
        self, mock_validate, mock_remove, mock_tarfile, mock_session, mock_client
    ):
        """Test packaging and uploading to default SageMaker bucket."""
        mock_validate.return_value = True

        mock_sagemaker_session = MagicMock()
        mock_sagemaker_session.default_bucket.return_value = (
            "sagemaker-us-east-1-123456789012"
        )

        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with patch("sagemaker.Session", return_value=mock_sagemaker_session):
            env = CustomEnvironment(env_id="test_env", local_path="/path")
            s3_uri = env.package_and_upload(region="us-east-1")

        assert s3_uri.startswith("s3://sagemaker-us-east-1-123456789012")
        assert env.s3_uri == s3_uri
        mock_s3.upload_file.assert_called_once()

    @patch("boto3.client")
    @patch("tarfile.open")
    @patch("os.remove")
    @patch.object(CustomEnvironment, "validate")
    def test_package_and_upload_with_custom_bucket(
        self, mock_validate, mock_remove, mock_tarfile, mock_client
    ):
        """Test packaging and uploading to custom bucket."""
        mock_validate.return_value = True

        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        s3_uri = env.package_and_upload(
            s3_bucket="my-bucket", s3_prefix="envs", region="us-west-2"
        )

        assert s3_uri == "s3://my-bucket/envs/test_env.tar.gz"
        assert env.s3_uri == s3_uri

    def test_package_and_upload_fails_without_local_path(self):
        """Test package_and_upload fails without local_path."""
        env = CustomEnvironment(env_id="test_env")
        with pytest.raises(ValueError, match="local_path required"):
            env.package_and_upload()

    @patch.object(CustomEnvironment, "validate")
    def test_package_and_upload_fails_with_s3_uri_as_bucket(self, mock_validate):
        """Test package_and_upload fails if s3_bucket starts with s3://."""
        mock_validate.return_value = True

        env = CustomEnvironment(env_id="test_env", local_path="/path")
        with pytest.raises(ValueError, match="bucket name only"):
            env.package_and_upload(s3_bucket="s3://my-bucket")
