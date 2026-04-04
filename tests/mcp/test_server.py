import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_forge.mcp.server import (
    DEPLOY_PLATFORMS,
    EVAL_TASKS,
    MODELS,
    PLATFORMS,
    TRAINING_METHODS,
    _build_infra,
    _format_job_result,
    _resolve,
    deploy,
    evaluate,
    get_job_status,
    get_logs,
    list_options,
    mcp,
    train,
    validate_dataset,
)
from amzn_nova_forge.model.model_enums import (
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.recipe.recipe_config import EvaluationTask


class TestEnumLookups(unittest.TestCase):
    def test_models_contains_all_model_members(self):
        for m in Model:
            self.assertIn(m.name, MODELS)
            self.assertEqual(MODELS[m.name], m)

    def test_training_methods_contains_all_members(self):
        for m in TrainingMethod:
            self.assertIn(m.name, TRAINING_METHODS)
            self.assertEqual(TRAINING_METHODS[m.name], m)

    def test_platforms_contains_all_members(self):
        for p in Platform:
            self.assertIn(p.name, PLATFORMS)
            self.assertEqual(PLATFORMS[p.name], p)

    def test_deploy_platforms_contains_all_members(self):
        for p in DeployPlatform:
            self.assertIn(p.name, DEPLOY_PLATFORMS)
            self.assertEqual(DEPLOY_PLATFORMS[p.name], p)

    def test_eval_tasks_contains_all_members(self):
        for t in EvaluationTask:
            self.assertIn(t.name, EVAL_TASKS)
            self.assertEqual(EVAL_TASKS[t.name], t)


class TestResolve(unittest.TestCase):
    def test_resolve_valid_key(self):
        self.assertEqual(_resolve(MODELS, "NOVA_PRO", "model"), Model.NOVA_PRO)

    def test_resolve_invalid_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve(MODELS, "NONEXISTENT", "model")
        self.assertIn("Unknown model", str(ctx.exception))
        self.assertIn("NOVA_PRO", str(ctx.exception))

    def test_resolve_lists_valid_options(self):
        with self.assertRaises(ValueError) as ctx:
            _resolve(TRAINING_METHODS, "BAD", "training_method")
        self.assertIn("SFT_LORA", str(ctx.exception))


class TestListOptions(unittest.TestCase):
    def test_list_options_contains_all_sections(self):
        result = list_options()
        self.assertIn("Models:", result)
        self.assertIn("Training Methods:", result)
        self.assertIn("Platforms:", result)
        self.assertIn("Deploy Platforms:", result)
        self.assertIn("Evaluation Tasks:", result)

    def test_list_options_contains_specific_values(self):
        result = list_options()
        self.assertIn("NOVA_PRO", result)
        self.assertIn("SFT_LORA", result)
        self.assertIn("SMTJ", result)
        self.assertIn("BEDROCK_OD", result)
        self.assertIn("MMLU", result)


class TestBuildInfra(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server.SMTJRuntimeManager")
    def test_build_smtj(self, mock_smtj):
        _build_infra(
            platform="SMTJ",
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            execution_role="arn:aws:iam::123:role/test",
            kms_key_id="key-123",
        )
        mock_smtj.assert_called_once_with(
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            execution_role="arn:aws:iam::123:role/test",
            kms_key_id="key-123",
        )

    @patch("amzn_nova_forge.mcp.server.SMHPRuntimeManager")
    def test_build_smhp(self, mock_smhp):
        _build_infra(
            platform="SMHP",
            instance_type="ml.p4d.24xlarge",
            instance_count=4,
            cluster_name="my-cluster",
            namespace="default",
            kms_key_id=None,
        )
        mock_smhp.assert_called_once_with(
            instance_type="ml.p4d.24xlarge",
            instance_count=4,
            cluster_name="my-cluster",
            namespace="default",
            kms_key_id=None,
        )

    @patch("amzn_nova_forge.mcp.server.SMHPRuntimeManager")
    def test_build_smhp_default_namespace(self, mock_smhp):
        _build_infra(
            platform="SMHP",
            instance_type="ml.p4d.24xlarge",
            instance_count=4,
            cluster_name="my-cluster",
        )
        mock_smhp.assert_called_once_with(
            instance_type="ml.p4d.24xlarge",
            instance_count=4,
            cluster_name="my-cluster",
            namespace="kubeflow",
            kms_key_id=None,
        )

    @patch("amzn_nova_forge.mcp.server.BedrockRuntimeManager")
    def test_build_bedrock(self, mock_bedrock):
        _build_infra(
            platform="BEDROCK",
            execution_role="arn:aws:iam::123:role/bedrock-role",
            kms_key_id="key-456",
        )
        mock_bedrock.assert_called_once_with(
            execution_role="arn:aws:iam::123:role/bedrock-role",
            kms_key_id="key-456",
        )

    def test_build_smtj_missing_instance_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _build_infra(platform="SMTJ", instance_count=2)
        self.assertIn("instance_type", str(ctx.exception))

    def test_build_smtj_missing_instance_count_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _build_infra(platform="SMTJ", instance_type="ml.p5.48xlarge")
        self.assertIn("instance_count", str(ctx.exception))

    def test_build_smhp_missing_cluster_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _build_infra(
                platform="SMHP",
                instance_type="ml.p4d.24xlarge",
                instance_count=4,
            )
        self.assertIn("cluster_name", str(ctx.exception))

    def test_build_bedrock_missing_execution_role_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _build_infra(platform="BEDROCK")
        self.assertIn("execution_role", str(ctx.exception))

    def test_build_unknown_platform_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _build_infra(platform="UNKNOWN")
        self.assertIn("Unknown platform", str(ctx.exception))


class TestFormatJobResult(unittest.TestCase):
    def test_format_none_returns_dry_run_message(self):
        result = _format_job_result(None)
        self.assertIn("Dry run", result)

    def test_format_training_result(self):
        mock_result = Mock()
        mock_result.job_id = "train-job-123"
        mock_result.started_time = datetime(2026, 3, 31, 12, 0, 0)
        mock_result.get_job_status.return_value = ("InProgress", "InProgress")
        mock_result.method = TrainingMethod.SFT_LORA
        mock_result.model_artifacts = "s3://bucket/output"

        result = _format_job_result(mock_result)
        self.assertIn("train-job-123", result)
        self.assertIn("InProgress", result)
        self.assertIn("sft_lora", result)

    def test_format_eval_result(self):
        mock_result = Mock()
        mock_result.job_id = "eval-job-456"
        mock_result.started_time = datetime(2026, 3, 31, 12, 0, 0)
        mock_result.get_job_status.return_value = ("Completed", "Completed")
        mock_result.eval_task = EvaluationTask.MMLU
        mock_result.eval_output_path = "s3://bucket/eval-output"
        del mock_result.method
        del mock_result.model_artifacts

        result = _format_job_result(mock_result)
        self.assertIn("eval-job-456", result)
        self.assertIn("Completed", result)
        self.assertIn("mmlu", result)


class TestTrainTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server._build_infra")
    @patch("amzn_nova_forge.mcp.server.NovaModelCustomizer")
    def test_train_launches_job(self, mock_customizer_cls, mock_build_infra):
        mock_infra = Mock()
        mock_build_infra.return_value = mock_infra

        mock_result = Mock()
        mock_result.job_id = "job-abc"
        mock_result.started_time = datetime(2026, 3, 31, 10, 0, 0)
        mock_result.get_job_status.return_value = ("InProgress", "InProgress")
        mock_result.method = TrainingMethod.SFT_LORA
        mock_result.model_artifacts = None

        mock_customizer = Mock()
        mock_customizer.train.return_value = mock_result
        mock_customizer_cls.return_value = mock_customizer

        result = train(
            model="NOVA_PRO",
            training_method="SFT_LORA",
            platform="SMTJ",
            job_name="test-job",
            data_s3_path="s3://bucket/data.jsonl",
            output_s3_path="s3://bucket/output",
            instance_type="ml.p5.48xlarge",
            instance_count=2,
        )

        mock_build_infra.assert_called_once_with(
            platform="SMTJ",
            instance_type="ml.p5.48xlarge",
            instance_count=2,
            execution_role=None,
            kms_key_id=None,
            cluster_name=None,
            namespace=None,
        )
        mock_customizer_cls.assert_called_once_with(
            model=Model.NOVA_PRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path="s3://bucket/data.jsonl",
            output_s3_path="s3://bucket/output",
            model_path=None,
        )
        mock_customizer.train.assert_called_once_with(
            job_name="test-job",
            overrides=None,
            validation_data_s3_path=None,
            dry_run=False,
        )
        self.assertIn("job-abc", result)

    @patch("amzn_nova_forge.mcp.server._build_infra")
    @patch("amzn_nova_forge.mcp.server.NovaModelCustomizer")
    def test_train_dry_run(self, mock_customizer_cls, mock_build_infra):
        mock_build_infra.return_value = Mock()
        mock_customizer = Mock()
        mock_customizer.train.return_value = None
        mock_customizer_cls.return_value = mock_customizer

        result = train(
            model="NOVA_LITE",
            training_method="DPO_LORA",
            platform="SMTJ",
            job_name="dry-run-job",
            data_s3_path="s3://bucket/data.jsonl",
            output_s3_path="s3://bucket/output",
            instance_type="ml.p5.48xlarge",
            instance_count=1,
            dry_run=True,
        )

        mock_customizer.train.assert_called_once_with(
            job_name="dry-run-job",
            overrides=None,
            validation_data_s3_path=None,
            dry_run=True,
        )
        self.assertIn("Dry run", result)

    @patch("amzn_nova_forge.mcp.server._build_infra")
    @patch("amzn_nova_forge.mcp.server.NovaModelCustomizer")
    def test_train_with_overrides(self, mock_customizer_cls, mock_build_infra):
        mock_build_infra.return_value = Mock()
        mock_customizer = Mock()
        mock_result = Mock()
        mock_result.job_id = "job-overrides"
        mock_result.started_time = datetime(2026, 3, 31, 10, 0, 0)
        mock_result.get_job_status.return_value = ("InProgress", "InProgress")
        mock_result.method = TrainingMethod.SFT_LORA
        mock_result.model_artifacts = None
        mock_customizer.train.return_value = mock_result
        mock_customizer_cls.return_value = mock_customizer

        overrides = {"learning_rate": 1e-5, "epochs": 3}
        train(
            model="NOVA_MICRO",
            training_method="SFT_LORA",
            platform="SMTJ",
            job_name="override-job",
            data_s3_path="s3://bucket/data.jsonl",
            output_s3_path="s3://bucket/output",
            instance_type="ml.p5.48xlarge",
            instance_count=1,
            overrides=overrides,
        )

        mock_customizer.train.assert_called_once_with(
            job_name="override-job",
            overrides=overrides,
            validation_data_s3_path=None,
            dry_run=False,
        )


class TestEvaluateTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server._build_infra")
    @patch("amzn_nova_forge.mcp.server.NovaModelCustomizer")
    def test_evaluate_launches_job(self, mock_customizer_cls, mock_build_infra):
        mock_build_infra.return_value = Mock()

        mock_result = Mock()
        mock_result.job_id = "eval-job-789"
        mock_result.started_time = datetime(2026, 3, 31, 10, 0, 0)
        mock_result.get_job_status.return_value = ("InProgress", "InProgress")
        mock_result.eval_task = EvaluationTask.MMLU
        mock_result.eval_output_path = "s3://bucket/eval-output"
        del mock_result.method
        del mock_result.model_artifacts

        mock_customizer = Mock()
        mock_customizer.evaluate.return_value = mock_result
        mock_customizer_cls.return_value = mock_customizer

        result = evaluate(
            model="NOVA_PRO",
            platform="SMTJ",
            job_name="eval-test",
            eval_task="MMLU",
            instance_type="ml.p5.48xlarge",
            instance_count=1,
        )

        mock_customizer_cls.assert_called_once_with(
            model=Model.NOVA_PRO,
            method=TrainingMethod.EVALUATION,
            infra=mock_build_infra.return_value,
            data_s3_path=None,
            output_s3_path=None,
            model_path=None,
        )
        mock_customizer.evaluate.assert_called_once_with(
            job_name="eval-test",
            eval_task=EvaluationTask.MMLU,
            overrides=None,
            dry_run=False,
        )
        self.assertIn("eval-job-789", result)


class TestDeployTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server._build_infra")
    @patch("amzn_nova_forge.mcp.server.NovaModelCustomizer")
    def test_deploy_initiates_deployment(self, mock_customizer_cls, mock_build_infra):
        mock_build_infra.return_value = Mock()

        mock_result = Mock()
        mock_result.endpoint_name = "my-endpoint"
        mock_result.job_id = "deploy-123"

        mock_customizer = Mock()
        mock_customizer.deploy.return_value = mock_result
        mock_customizer_cls.return_value = mock_customizer

        result = deploy(
            model="NOVA_PRO",
            training_method="SFT_LORA",
            platform="SMTJ",
            deploy_platform="SAGEMAKER",
            model_artifact_path="s3://bucket/model",
            instance_type="ml.p5.48xlarge",
            instance_count=1,
        )

        mock_customizer.deploy.assert_called_once_with(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            endpoint_name=None,
            execution_role_name=None,
            sagemaker_instance_type="ml.p5.48xlarge",
        )
        self.assertIn("Deployment initiated", result)
        self.assertIn("my-endpoint", result)


class TestGetJobStatusTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server.boto3")
    def test_get_job_status_smtj(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingStartTime": datetime(2026, 3, 31, 10, 0, 0),
            "TrainingEndTime": datetime(2026, 3, 31, 12, 0, 0),
        }
        mock_boto3.client.return_value = mock_sm_client

        result = get_job_status(job_id="smtj-job-123", platform="SMTJ")

        self.assertIn("smtj-job-123", result)
        self.assertIn("Completed", result)

    @patch("amzn_nova_forge.mcp.server.boto3")
    def test_get_job_status_bedrock(self, mock_boto3):
        mock_br_client = Mock()
        mock_br_client.get_model_customization_job.return_value = {
            "status": "InProgress",
        }
        mock_boto3.client.return_value = mock_br_client

        result = get_job_status(
            job_id="arn:aws:bedrock:us-east-1:123:model-customization-job/test",
            platform="BEDROCK",
        )

        self.assertIn("InProgress", result)

    @patch("amzn_nova_forge.mcp.server.boto3")
    def test_get_job_status_smtj_with_failure(self, mock_boto3):
        mock_sm_client = Mock()
        mock_sm_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "OutOfMemoryError",
        }
        mock_boto3.client.return_value = mock_sm_client

        result = get_job_status(job_id="failed-job", platform="SMTJ")

        self.assertIn("Failed", result)
        self.assertIn("OutOfMemoryError", result)

    def test_get_job_status_smhp_fallback(self):
        result = get_job_status(job_id="smhp-job-456", platform="SMHP")

        self.assertIn("smhp-job-456", result)
        self.assertIn("get_logs", result)

    def test_get_job_status_invalid_platform(self):
        with self.assertRaises(ValueError) as ctx:
            get_job_status(job_id="job-123", platform="INVALID")
        self.assertIn("Unknown platform", str(ctx.exception))


class TestGetLogsTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server.CloudWatchLogMonitor")
    def test_get_logs_returns_formatted_logs(self, mock_monitor_cls):
        mock_monitor = Mock()
        mock_monitor.get_logs.return_value = [
            {"timestamp": "2026-03-31T10:00:00", "message": "Training started"},
            {"timestamp": "2026-03-31T10:01:00", "message": "Epoch 1 complete"},
        ]
        mock_monitor_cls.from_job_id.return_value = mock_monitor

        result = get_logs(job_id="job-123", platform="SMTJ", limit=50)

        mock_monitor_cls.from_job_id.assert_called_once_with(
            job_id="job-123",
            platform=Platform.SMTJ,
        )
        mock_monitor.get_logs.assert_called_once_with(limit=50)
        self.assertIn("Training started", result)
        self.assertIn("Epoch 1 complete", result)

    @patch("amzn_nova_forge.mcp.server.CloudWatchLogMonitor")
    def test_get_logs_empty(self, mock_monitor_cls):
        mock_monitor = Mock()
        mock_monitor.get_logs.return_value = []
        mock_monitor_cls.from_job_id.return_value = mock_monitor

        result = get_logs(job_id="job-empty", platform="SMTJ")

        self.assertIn("No logs found", result)

    @patch("amzn_nova_forge.mcp.server.CloudWatchLogMonitor")
    def test_get_logs_none_returns_no_logs(self, mock_monitor_cls):
        mock_monitor = Mock()
        mock_monitor.get_logs.return_value = None
        mock_monitor_cls.from_job_id.return_value = mock_monitor

        result = get_logs(job_id="job-none", platform="BEDROCK")

        self.assertIn("No logs found", result)


class TestValidateDatasetTool(unittest.TestCase):
    @patch("amzn_nova_forge.mcp.server.JSONLDatasetLoader")
    def test_validate_valid_dataset(self, mock_loader_cls):
        mock_loader = Mock()
        mock_loader.dataset = [{"input": "a", "output": "b"}] * 100
        mock_loader_cls.return_value = mock_loader

        result = validate_dataset(
            data_path="s3://bucket/data.jsonl",
            model="NOVA_PRO",
        )

        mock_loader.load.assert_called_once_with("s3://bucket/data.jsonl")
        mock_loader.validate.assert_called_once_with(model=Model.NOVA_PRO)
        self.assertIn("valid", result)
        self.assertIn("100", result)

    @patch("amzn_nova_forge.mcp.server.JSONLDatasetLoader")
    def test_validate_without_model(self, mock_loader_cls):
        mock_loader = Mock()
        mock_loader.dataset = [{"input": "a"}] * 50
        mock_loader_cls.return_value = mock_loader

        result = validate_dataset(data_path="data.jsonl")

        mock_loader.validate.assert_called_once_with(model=None)
        self.assertIn("valid", result)

    @patch("amzn_nova_forge.mcp.server.JSONLDatasetLoader")
    def test_validate_invalid_dataset(self, mock_loader_cls):
        mock_loader = Mock()
        mock_loader.validate.side_effect = ValueError("Missing required field: output")
        mock_loader_cls.return_value = mock_loader

        result = validate_dataset(
            data_path="s3://bucket/bad-data.jsonl",
            model="NOVA_LITE",
        )

        self.assertIn("validation failed", result)
        self.assertIn("Missing required field", result)


class TestMcpServerRegistration(unittest.TestCase):
    def test_mcp_server_name(self):
        self.assertEqual(mcp.name, "nova-forge")

    def test_mcp_server_has_instructions(self):
        self.assertIn("Nova Forge", mcp.instructions)


if __name__ == "__main__":
    unittest.main()
