"""Unit tests for RFT Multiturn utils module."""

import os
import tempfile
from unittest.mock import patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.utils import (
    build_duplicate_job_error_message,
    validate_starter_kit_path,
)


class TestBuildDuplicateJobErrorMessage:
    """Test build_duplicate_job_error_message function."""

    def test_train_jobs_only(self):
        """Test error message with only train jobs."""
        result = build_duplicate_job_error_message(
            stack_name="test-stack",
            train_jobs=["PID: 1234", "PID: 5678"],
            eval_jobs=[],
        )

        assert "test-stack" in result
        assert "2 TRAIN job(s)" in result
        assert "PID: 1234" in result
        assert "PID: 5678" in result
        assert "kill_task(env_type=EnvType.TRAIN" in result
        assert "EVAL" not in result

    def test_eval_jobs_only(self):
        """Test error message with only eval jobs."""
        result = build_duplicate_job_error_message(
            stack_name="test-stack",
            train_jobs=[],
            eval_jobs=["task: abc123"],
        )

        assert "test-stack" in result
        assert "1 EVAL job(s)" in result
        assert "task: abc123" in result
        assert "kill_task(env_type=EnvType.EVAL" in result
        assert "TRAIN" not in result

    def test_both_train_and_eval_jobs(self):
        """Test error message with both train and eval jobs."""
        result = build_duplicate_job_error_message(
            stack_name="test-stack",
            train_jobs=["PID: 1234"],
            eval_jobs=["PID: 5678", "PID: 9012"],
        )

        assert "test-stack" in result
        assert "1 TRAIN job(s)" in result
        assert "2 EVAL job(s)" in result
        assert "PID: 1234" in result
        assert "PID: 5678" in result
        assert "PID: 9012" in result
        assert "kill_task(env_type=EnvType.TRAIN" in result
        assert "kill_task(env_type=EnvType.EVAL" in result

    def test_with_platform_info(self):
        """Test error message with platform info."""
        result = build_duplicate_job_error_message(
            stack_name="test-stack",
            train_jobs=["PID: 1234"],
            eval_jobs=[],
            platform_info="on EC2 instance i-123456",
        )

        assert "test-stack" in result
        assert "on EC2 instance i-123456" in result

    def test_without_platform_info(self):
        """Test error message without platform info."""
        result = build_duplicate_job_error_message(
            stack_name="test-stack",
            train_jobs=["PID: 1234"],
            eval_jobs=[],
            platform_info="",
        )

        assert "test-stack" in result
        # Should not have extra spaces or platform info
        assert "on EC2" not in result
        assert "on ECS" not in result


class TestValidateStarterKitPath:
    """Test validate_starter_kit_path function."""

    def test_valid_starter_kit_path(self):
        """Test validation with valid starter kit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create lambda_proxy subdirectory
            lambda_proxy_dir = os.path.join(tmpdir, "lambda_proxy")
            os.makedirs(lambda_proxy_dir)

            # Should return True and not raise
            result = validate_starter_kit_path(tmpdir)
            assert result is True

    def test_invalid_starter_kit_path_raises(self):
        """Test validation with invalid path raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create lambda_proxy subdirectory

            with pytest.raises(ValueError) as exc_info:
                validate_starter_kit_path(tmpdir, raise_on_invalid=True)

            assert "Invalid starter kit directory" in str(exc_info.value)
            assert "missing lambda_proxy" in str(exc_info.value)
            assert tmpdir in str(exc_info.value)

    def test_invalid_starter_kit_path_returns_false(self):
        """Test validation with invalid path returns False when raise_on_invalid=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create lambda_proxy subdirectory

            result = validate_starter_kit_path(tmpdir, raise_on_invalid=False)
            assert result is False

    def test_nonexistent_path_raises(self):
        """Test validation with nonexistent path."""
        nonexistent_path = "/path/that/does/not/exist/12345"

        with pytest.raises(ValueError) as exc_info:
            validate_starter_kit_path(nonexistent_path, raise_on_invalid=True)

        assert "Invalid starter kit directory" in str(exc_info.value)

    def test_nonexistent_path_returns_false(self):
        """Test validation with nonexistent path returns False."""
        nonexistent_path = "/path/that/does/not/exist/12345"

        result = validate_starter_kit_path(nonexistent_path, raise_on_invalid=False)
        assert result is False

    def test_path_with_other_subdirectories(self):
        """Test that only lambda_proxy subdirectory is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create lambda_proxy and other directories
            os.makedirs(os.path.join(tmpdir, "lambda_proxy"))
            os.makedirs(os.path.join(tmpdir, "other_dir"))
            os.makedirs(os.path.join(tmpdir, "another_dir"))

            # Should still be valid
            result = validate_starter_kit_path(tmpdir)
            assert result is True

    def test_empty_directory_invalid(self):
        """Test that empty directory is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory, no lambda_proxy

            with pytest.raises(ValueError):
                validate_starter_kit_path(tmpdir, raise_on_invalid=True)
