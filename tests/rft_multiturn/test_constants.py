"""Unit tests for RFT Multiturn constants module."""

import pytest

from amzn_nova_forge_sdk.rft_multiturn.constants import (
    ECR_REPO_NAME,
    IAM_PROPAGATION_WAIT_TIME,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_KILLED,
    JOB_STATUS_RUNNING,
    RFT_EVAL_LOG,
    RFT_EXECUTION_ROLE_NAME,
    RFT_POLICY_NAME,
    RFT_SAM_LOG,
    RFT_TRAIN_LOG,
    SAM_WAIT_TIME,
    SDK_RFT_LOGS_DIR,
    SDK_RFT_SCRIPTS_DIR,
    SSM_COMMAND_MAX_POLL_ATTEMPTS,
    SSM_COMMAND_POLL_INTERVAL,
    STACK_NAME_SUFFIX,
    STARTER_KIT_S3,
)


class TestConstants:
    """Test RFT Multiturn constants."""

    def test_role_and_policy_names(self):
        """Test role and policy name constants."""
        assert RFT_EXECUTION_ROLE_NAME == "RFTExecutionRoleNovaSDK"
        assert RFT_POLICY_NAME == "RFTPolicyNovaSDK"
        assert isinstance(RFT_EXECUTION_ROLE_NAME, str)
        assert isinstance(RFT_POLICY_NAME, str)

    def test_stack_name_suffix(self):
        """Test stack name suffix constant."""
        assert STACK_NAME_SUFFIX == "NovaForgeSDK"
        assert isinstance(STACK_NAME_SUFFIX, str)

    def test_ecr_repo_name(self):
        """Test ECR repository name constant."""
        assert ECR_REPO_NAME == "nova-rft-base"
        assert isinstance(ECR_REPO_NAME, str)

    def test_directory_names(self):
        """Test directory name constants."""
        assert SDK_RFT_LOGS_DIR == "sdk-rft-logs"
        assert SDK_RFT_SCRIPTS_DIR == "sdk-rft-scripts"
        assert isinstance(SDK_RFT_LOGS_DIR, str)
        assert isinstance(SDK_RFT_SCRIPTS_DIR, str)

    def test_log_file_names(self):
        """Test log file name constants."""
        assert RFT_TRAIN_LOG == "rft_train.log"
        assert RFT_EVAL_LOG == "rft_eval.log"
        assert RFT_SAM_LOG == "rft_sam.log"
        assert isinstance(RFT_TRAIN_LOG, str)
        assert isinstance(RFT_EVAL_LOG, str)
        assert isinstance(RFT_SAM_LOG, str)

    def test_job_status_values(self):
        """Test job status constants."""
        assert JOB_STATUS_RUNNING == "running"
        assert JOB_STATUS_COMPLETED == "completed"
        assert JOB_STATUS_KILLED == "killed"
        assert JOB_STATUS_FAILED == "failed"
        assert isinstance(JOB_STATUS_RUNNING, str)
        assert isinstance(JOB_STATUS_COMPLETED, str)
        assert isinstance(JOB_STATUS_KILLED, str)
        assert isinstance(JOB_STATUS_FAILED, str)

    def test_wait_times(self):
        """Test wait time constants."""
        assert IAM_PROPAGATION_WAIT_TIME == 15
        assert SAM_WAIT_TIME == 600
        assert isinstance(IAM_PROPAGATION_WAIT_TIME, int)
        assert isinstance(SAM_WAIT_TIME, int)
        assert IAM_PROPAGATION_WAIT_TIME > 0
        assert SAM_WAIT_TIME > 0

    def test_starter_kit_s3(self):
        """Test starter kit S3 location constant."""
        assert STARTER_KIT_S3.startswith("s3://")
        assert "nova-rft-starter-kit" in STARTER_KIT_S3
        assert isinstance(STARTER_KIT_S3, str)

    def test_ssm_command_constants(self):
        """Test SSM command polling constants."""
        assert SSM_COMMAND_POLL_INTERVAL == 0.5
        assert SSM_COMMAND_MAX_POLL_ATTEMPTS == 10
        assert isinstance(SSM_COMMAND_POLL_INTERVAL, (int, float))
        assert isinstance(SSM_COMMAND_MAX_POLL_ATTEMPTS, int)
        assert SSM_COMMAND_POLL_INTERVAL > 0
        assert SSM_COMMAND_MAX_POLL_ATTEMPTS > 0

    def test_all_constants_are_immutable_types(self):
        """Test that all constants are immutable types (str, int, float)."""
        constants = [
            RFT_EXECUTION_ROLE_NAME,
            RFT_POLICY_NAME,
            STACK_NAME_SUFFIX,
            ECR_REPO_NAME,
            SDK_RFT_LOGS_DIR,
            SDK_RFT_SCRIPTS_DIR,
            RFT_TRAIN_LOG,
            RFT_EVAL_LOG,
            RFT_SAM_LOG,
            JOB_STATUS_RUNNING,
            JOB_STATUS_COMPLETED,
            JOB_STATUS_KILLED,
            JOB_STATUS_FAILED,
            IAM_PROPAGATION_WAIT_TIME,
            SAM_WAIT_TIME,
            STARTER_KIT_S3,
            SSM_COMMAND_POLL_INTERVAL,
            SSM_COMMAND_MAX_POLL_ATTEMPTS,
        ]

        for constant in constants:
            assert isinstance(constant, (str, int, float)), (
                f"Constant {constant} is not an immutable type"
            )

    def test_constants_are_not_empty(self):
        """Test that string constants are not empty."""
        string_constants = [
            RFT_EXECUTION_ROLE_NAME,
            RFT_POLICY_NAME,
            STACK_NAME_SUFFIX,
            ECR_REPO_NAME,
            SDK_RFT_LOGS_DIR,
            SDK_RFT_SCRIPTS_DIR,
            RFT_TRAIN_LOG,
            RFT_EVAL_LOG,
            RFT_SAM_LOG,
            JOB_STATUS_RUNNING,
            JOB_STATUS_COMPLETED,
            JOB_STATUS_KILLED,
            JOB_STATUS_FAILED,
            STARTER_KIT_S3,
        ]

        for constant in string_constants:
            assert len(constant) > 0, f"String constant {constant} is empty"
