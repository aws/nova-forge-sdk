"""Unit tests for ECSRFTInfrastructure."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from amzn_nova_forge_sdk.rft_multiturn.base_infra import EnvType, StackOutputs
from amzn_nova_forge_sdk.rft_multiturn.ecs_infra import ECSRFTInfrastructure


class TestECSRFTInfrastructure:
    """Test ECSRFTInfrastructure class."""

    def test_ecs_rft_infrastructure_exists(self):
        """Test that ECSRFTInfrastructure class is importable."""
        assert ECSRFTInfrastructure is not None

    def test_ecs_rft_infrastructure_has_required_methods(self):
        """Test that ECSRFTInfrastructure has expected methods."""
        assert hasattr(ECSRFTInfrastructure, "validate_platform")
        assert hasattr(ECSRFTInfrastructure, "deploy_sam_stack")
        assert hasattr(ECSRFTInfrastructure, "start_environment")
        assert hasattr(ECSRFTInfrastructure, "get_logs")
        assert hasattr(ECSRFTInfrastructure, "kill_task")
        assert hasattr(ECSRFTInfrastructure, "cleanup")
        assert hasattr(ECSRFTInfrastructure, "ensure_rft_policy_on_current_role")
        assert hasattr(ECSRFTInfrastructure, "validate_starter_kit_access")
        assert hasattr(ECSRFTInfrastructure, "check_queue_messages")
        assert hasattr(ECSRFTInfrastructure, "flush_queue")

    def test_ecs_rft_infrastructure_is_class(self):
        """Test that ECSRFTInfrastructure is a class."""
        assert isinstance(ECSRFTInfrastructure, type)

    @patch("boto3.client")
    def test_initialization(self, mock_boto_client):
        """Test ECS infrastructure initialization."""
        mock_cfn = MagicMock()
        mock_ecs = MagicMock()
        mock_boto_client.side_effect = lambda service, **kwargs: {
            "cloudformation": mock_cfn,
            "ecs": mock_ecs,
            "ecr": MagicMock(),
            "ec2": MagicMock(),
            "sqs": MagicMock(),
            "logs": MagicMock(),
            "iam": MagicMock(),
            "sts": MagicMock(),
        }.get(service, MagicMock())

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        assert infra.region == "us-east-1"
        assert (
            infra.cluster_arn
            == "arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster"
        )
        assert infra.account_id == "123456789012"
        assert infra.python_venv_name == "test_venv"

    @patch("boto3.client")
    def test_initialization_with_vpc_config(self, mock_boto_client):
        """Test ECS infrastructure initialization with VPC config."""
        mock_boto_client.return_value = MagicMock()

        vpc_config = {
            "subnets": ["subnet-123", "subnet-456"],
            "security_groups": ["sg-789"],
        }

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            vpc_config=vpc_config,
        )

        assert infra.user_vpc_config == vpc_config

    @patch("boto3.client")
    def test_initialization_with_cpu_memory(self, mock_boto_client):
        """Test ECS infrastructure initialization with custom CPU and memory."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            cpu="4096",
            memory="8192",
        )

        assert infra.user_cpu == "4096"
        assert infra.user_memory == "8192"

    @patch("boto3.client")
    def test_validate_platform_success(self, mock_boto_client):
        """Test successful platform validation."""
        mock_ecs = MagicMock()
        mock_ecs.describe_clusters.return_value = {
            "clusters": [{"status": "ACTIVE", "clusterName": "test-cluster"}]
        }
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Should not raise
        infra.validate_platform()
        mock_ecs.describe_clusters.assert_called_once()

    @patch("boto3.client")
    def test_validate_platform_cluster_not_found(self, mock_boto_client):
        """Test platform validation when cluster not found."""
        mock_ecs = MagicMock()
        mock_ecs.describe_clusters.return_value = {"clusters": []}
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="not found"):
            infra.validate_platform()

    @patch("boto3.client")
    def test_validate_platform_cluster_not_active(self, mock_boto_client):
        """Test platform validation when cluster is not active."""
        mock_ecs = MagicMock()
        mock_ecs.describe_clusters.return_value = {
            "clusters": [{"status": "INACTIVE", "clusterName": "test-cluster"}]
        }
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        with pytest.raises(ValueError, match="not active"):
            infra.validate_platform()

    @patch("boto3.client")
    def test_get_cluster_vpc_config_user_provided(self, mock_boto_client):
        """Test VPC config retrieval when user provides config."""
        mock_boto_client.return_value = MagicMock()

        vpc_config = {
            "subnets": ["subnet-123", "subnet-456"],
            "security_groups": ["sg-789"],
        }

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
            vpc_config=vpc_config,
        )

        result = infra.get_cluster_vpc_config()
        assert result == vpc_config

    @patch("boto3.client")
    def test_get_cluster_vpc_config_from_service(self, mock_boto_client):
        """Test VPC config discovery from ECS service."""
        mock_ecs = MagicMock()
        mock_ecs.list_services.return_value = {
            "serviceArns": ["arn:aws:ecs:us-east-1:123456789012:service/test-service"]
        }
        mock_ecs.describe_services.return_value = {
            "services": [
                {
                    "networkConfiguration": {
                        "awsvpcConfiguration": {
                            "subnets": ["subnet-abc"],
                            "securityGroups": ["sg-xyz"],
                        }
                    }
                }
            ]
        }
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        result = infra.get_cluster_vpc_config()
        assert result["subnets"] == ["subnet-abc"]
        assert result["security_groups"] == ["sg-xyz"]

    @patch("boto3.client")
    def test_build_container_command(self, mock_boto_client):
        """Test container command building."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        command = infra._build_container_command(
            vf_env_id="wordle",
            vf_env_args={"use_think": False},
            lambda_url="https://lambda.url",
            queue_url="https://queue.url",
        )

        assert isinstance(command, list)
        assert command[0] == "/bin/bash"
        assert command[1] == "-c"
        assert "wordle" in command[2]

    @patch("boto3.client")
    def test_extract_s3_uri_from_command(self, mock_boto_client):
        """Test S3 URI extraction from command."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        command = [
            "/bin/bash",
            "-c",
            "aws s3 cp s3://bucket/custom-env.tar.gz /tmp/env.tar.gz",
        ]

        s3_uri = infra._extract_s3_uri_from_command(command)
        assert s3_uri == "s3://bucket/custom-env.tar.gz"

    @patch("boto3.client")
    def test_extract_params_from_command(self, mock_boto_client):
        """Test parameter extraction from command."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        # Test with unified client parameters
        command = [
            "/bin/bash",
            "-c",
            "python environment_client.py --max-concurrent-rollouts=40 --max-rollout-timeout=300 --completion-poll-timeout=600",
        ]

        params = infra._extract_params_from_command(command)
        assert params["max-concurrent-rollouts"] == "40"
        assert params["max-rollout-timeout"] == "300"
        assert params["completion-poll-timeout"] == "600"

    @patch("boto3.client")
    def test_kill_task(self, mock_boto_client):
        """Test task killing."""
        mock_ecs = MagicMock()
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.latest_train_task_id = "task-123"
        infra.kill_task(EnvType.TRAIN)

        mock_ecs.stop_task.assert_called_once()

    @patch("boto3.client")
    def test_kill_task_with_deregister(self, mock_boto_client):
        """Test task killing with task definition deregistration."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {
            "tasks": [
                {
                    "taskDefinitionArn": "arn:aws:ecs:us-east-1:123456789012:task-definition/test:1"
                }
            ]
        }
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.latest_train_task_id = "task-123"
        infra.kill_task(EnvType.TRAIN, deregister_task_def=True)

        mock_ecs.stop_task.assert_called_once()
        mock_ecs.deregister_task_definition.assert_called_once()

    @patch("boto3.client")
    def test_cleanup(self, mock_boto_client):
        """Test cleanup without environment deletion."""
        mock_ecs = MagicMock()
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.latest_train_task_id = "task-123"
        infra.latest_eval_task_id = "task-456"

        infra.cleanup(cleanup_environment=False)

        # Should stop both tasks but not deregister
        assert mock_ecs.stop_task.call_count == 2
        mock_ecs.deregister_task_definition.assert_not_called()

    @patch("boto3.client")
    def test_cleanup_with_environment(self, mock_boto_client):
        """Test cleanup with environment deletion."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {
            "tasks": [
                {
                    "taskDefinitionArn": "arn:aws:ecs:us-east-1:123456789012:task-definition/test:1"
                }
            ]
        }
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        infra.latest_train_task_id = "task-123"
        infra.latest_eval_task_id = "task-456"

        infra.cleanup(cleanup_environment=True)

        # Should stop and deregister both tasks
        assert mock_ecs.stop_task.call_count == 2
        assert mock_ecs.deregister_task_definition.call_count == 2

    @patch("boto3.client")
    def test_get_state(self, mock_boto_client):
        """Test get_state captures ECS platform state."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra.latest_train_task_id = "task-train-123"
        infra.latest_eval_task_id = "task-eval-456"
        infra.latest_sam_task_id = "task-sam-789"
        infra.starter_kit_s3 = "s3://bucket/kit.tar.gz"

        state = infra.get_state()

        assert (
            state["cluster_arn"]
            == "arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster"
        )
        assert state["account_id"] == "123456789012"
        assert state["python_venv_name"] == "test_venv"
        assert state["latest_train_task_id"] == "task-train-123"
        assert state["latest_eval_task_id"] == "task-eval-456"
        assert state["latest_sam_task_id"] == "task-sam-789"
        assert state["starter_kit_s3"] == "s3://bucket/kit.tar.gz"

    @patch("boto3.client")
    def test_restore_state_with_running_tasks(self, mock_boto_client):
        """Test restore_state with running tasks."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "RUNNING"}]}
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "cluster_arn": "arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            "account_id": "123456789012",
            "python_venv_name": "test_venv",
            "latest_train_task_id": "task-train-123",
            "latest_eval_task_id": "task-eval-456",
            "latest_sam_task_id": None,
            "starter_kit_s3": "s3://bucket/custom-kit.tar.gz",
        }

        infra.restore_state(state)

        assert infra.latest_train_task_id == "task-train-123"
        assert infra.latest_eval_task_id == "task-eval-456"
        assert infra.latest_sam_task_id is None
        assert infra.starter_kit_s3 == "s3://bucket/custom-kit.tar.gz"
        assert mock_ecs.describe_tasks.call_count == 2  # train and eval

    @patch("boto3.client")
    def test_restore_state_with_stopped_tasks(self, mock_boto_client):
        """Test restore_state with stopped tasks."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "STOPPED"}]}
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "latest_train_task_id": "task-train-123",
            "starter_kit_s3": "s3://bucket/kit.tar.gz",
        }

        # Should not raise, just log warning
        infra.restore_state(state)

        assert infra.latest_train_task_id == "task-train-123"

    @patch("boto3.client")
    def test_restore_state_with_missing_tasks(self, mock_boto_client):
        """Test restore_state with missing tasks."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {"tasks": []}
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {
            "latest_train_task_id": "task-train-123",
            "latest_eval_task_id": "task-eval-456",
        }

        # Should not raise, just log warning
        infra.restore_state(state)

    @patch("boto3.client")
    def test_restore_state_api_error(self, mock_boto_client):
        """Test restore_state handles API errors gracefully."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.side_effect = Exception("API Error")
        mock_boto_client.return_value = mock_ecs

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )

        state = {"latest_train_task_id": "task-train-123"}

        # Should not raise, just log warning
        infra.restore_state(state)

    @patch("boto3.client")
    def test_get_state_restore_state_roundtrip(self, mock_boto_client):
        """Test get_state and restore_state roundtrip."""
        mock_ecs = MagicMock()
        mock_ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "RUNNING"}]}
        mock_boto_client.return_value = mock_ecs

        infra1 = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra1.latest_train_task_id = "task-train-123"
        infra1.latest_eval_task_id = "task-eval-456"
        infra1.starter_kit_s3 = "s3://bucket/kit.tar.gz"

        state = infra1.get_state()

        infra2 = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        infra2.restore_state(state)

        assert infra2.latest_train_task_id == infra1.latest_train_task_id
        assert infra2.latest_eval_task_id == infra1.latest_eval_task_id
        assert infra2.starter_kit_s3 == infra1.starter_kit_s3

    @patch("boto3.client")
    def test_restore_state_without_starter_kit(self, mock_boto_client):
        """Test restore_state without custom starter kit."""
        mock_boto_client.return_value = MagicMock()

        infra = ECSRFTInfrastructure(
            region="us-east-1",
            stack_name="test-stack",
            cluster_arn="arn:aws:ecs:us-east-1:123456789012:cluster/test-cluster",
            account_id="123456789012",
            python_venv_name="test_venv",
            rft_role_name="TestRole",
        )
        original_starter_kit = infra.starter_kit_s3

        state = {
            "latest_train_task_id": "task-train-123",
            # No starter_kit_s3 in state
        }

        infra.restore_state(state)

        # Should keep original value
        assert infra.starter_kit_s3 == original_starter_kit
