"""Unit tests for NovaModelCustomizer with Bedrock platform.

This module contains tests specific to Bedrock platform integration including:
- model_path warning
- mlflow_monitor warning
- method parameter passing
- Bedrock-specific initialization
- unsupported operations
- result serialization
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from amzn_nova_forge.manager.runtime_manager import BedrockRuntimeManager
from amzn_nova_forge.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.model.nova_model_customizer import (
    NovaModelCustomizer,
)
from amzn_nova_forge.monitor.mlflow_monitor import MLflowMonitor
from amzn_nova_forge.recipe.recipe_config import EvaluationTask


class TestNovaModelCustomizerBedrock(unittest.TestCase):
    """Unit tests for NovaModelCustomizer with Bedrock platform.

    Includes tests for:
    - Bedrock-specific initialization and warnings
    - Unsupported operations (evaluate, batch_inference)
    - Method parameter passing
    - Result serialization (training and evaluation)
    """

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.model.nova_model_customizer.set_output_s3_path")
    @patch("boto3.session.Session")
    def test_model_path_with_bedrock_creates_customizer(
        self, mock_session, mock_set_output, mock_setup
    ):
        """Test that passing model_path with Bedrock still creates customizer successfully.

        The warning is logged but doesn't prevent initialization.

        Validates: Bedrock customizer can be created with model_path (though it's ignored)
        """
        # Mock session
        mock_session_instance = MagicMock()
        mock_session_instance.region_name = "us-east-1"
        mock_session.return_value = mock_session_instance

        # Mock output S3 path
        mock_set_output.return_value = "s3://bucket/output/"

        # Create Bedrock runtime manager
        bedrock_infra = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create customizer with model_path (should work but log warning)
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=bedrock_infra,
            model_path="s3://bucket/my-model/",  # This triggers warning but doesn't fail
        )

        # Verify customizer was created successfully
        self.assertIsNotNone(customizer)
        self.assertEqual(customizer.model_path, "s3://bucket/my-model/")
        self.assertEqual(customizer.platform, Platform.BEDROCK)

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.model.nova_model_customizer.set_output_s3_path")
    @patch("amzn_nova_forge.monitor.mlflow_monitor.validate_mlflow_overrides")
    @patch("boto3.session.Session")
    def test_mlflow_monitor_with_bedrock_logs_warning(
        self, mock_session, mock_validate_mlflow, mock_set_output, mock_setup
    ):
        """Test that passing mlflow_monitor with Bedrock logs a warning.

        MLflow is not supported on Bedrock platform.

        Validates: Warning is logged when mlflow_monitor is provided with Bedrock
        """
        # Mock session
        mock_session_instance = MagicMock()
        mock_session_instance.region_name = "us-east-1"
        mock_session.return_value = mock_session_instance

        # Mock output S3 path
        mock_set_output.return_value = "s3://bucket/output/"

        # Mock MLflow validation to pass
        mock_validate_mlflow.return_value = []

        # Create Bedrock runtime manager
        bedrock_infra = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create MLflow monitor
        mlflow_monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
            experiment_name="test-experiment",
        )

        # Create customizer with mlflow_monitor (should work but log warning)
        with self.assertLogs("nova_forge_sdk", level="WARNING") as log:
            customizer = NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=bedrock_infra,
                mlflow_monitor=mlflow_monitor,
            )

        # Verify warning was logged
        warning_found = any(
            "MLflow monitoring is not supported on the Bedrock platform" in msg
            for msg in log.output
        )
        self.assertTrue(
            warning_found,
            f"Expected MLflow warning not found in logs: {log.output}",
        )

        # Verify customizer was created successfully
        self.assertIsNotNone(customizer)
        self.assertEqual(customizer.platform, Platform.BEDROCK)
        self.assertEqual(customizer.mlflow_monitor, mlflow_monitor)

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.model.nova_model_customizer.set_output_s3_path")
    @patch("boto3.session.Session")
    def test_evaluate_raises_not_implemented_for_bedrock(
        self, mock_session, mock_set_output, mock_setup
    ):
        """Test that evaluate() raises NotImplementedError for Bedrock platform."""
        # Mock session
        mock_session_instance = MagicMock()
        mock_session_instance.region_name = "us-east-1"
        mock_session.return_value = mock_session_instance

        # Mock output S3 path
        mock_set_output.return_value = "s3://bucket/output/"

        # Create Bedrock runtime manager
        bedrock_infra = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create customizer
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=bedrock_infra,
            data_s3_path="s3://bucket/data/",
        )

        # Attempt to call evaluate - should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            customizer.evaluate(
                job_name="test-eval",
                eval_task=EvaluationTask.MMLU,
            )

        # Verify error message
        self.assertIn(
            "Evaluation is not supported on the Bedrock platform",
            str(context.exception),
        )
        self.assertIn("SageMaker platforms (SMTJ, SMHP)", str(context.exception))

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    @patch("amzn_nova_forge.model.nova_model_customizer.set_output_s3_path")
    @patch("boto3.session.Session")
    def test_batch_inference_raises_not_implemented_for_bedrock(
        self, mock_session, mock_set_output, mock_setup
    ):
        """Test that batch_inference() raises NotImplementedError for Bedrock platform."""
        # Mock session
        mock_session_instance = MagicMock()
        mock_session_instance.region_name = "us-east-1"
        mock_session.return_value = mock_session_instance

        # Mock output S3 path
        mock_set_output.return_value = "s3://bucket/output/"

        # Create Bedrock runtime manager
        bedrock_infra = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create customizer
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=bedrock_infra,
            data_s3_path="s3://bucket/data/",
        )

        # Attempt to call batch_inference - should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            customizer.batch_inference(
                job_name="test-batch",
                input_path="s3://bucket/input/",
                output_s3_path="s3://bucket/output/",
            )

        # Verify error message
        self.assertIn(
            "Batch inference is not supported on Bedrock platform",
            str(context.exception),
        )
        self.assertIn("SageMaker platforms (SMTJ, SMHP)", str(context.exception))

    @patch.object(BedrockRuntimeManager, "setup", return_value=None)
    @patch.object(BedrockRuntimeManager, "execute")
    @patch("amzn_nova_forge.model.nova_model_customizer.set_output_s3_path")
    @patch("amzn_nova_forge.model.nova_model_customizer.RecipeBuilder")
    @patch("boto3.client")
    @patch("boto3.session.Session")
    def test_train_passes_method_to_bedrock_job_config(
        self,
        mock_session,
        mock_boto_client,
        mock_recipe_builder,
        mock_set_output,
        mock_execute,
        mock_setup,
    ):
        """Test that train() passes method parameter in JobConfig for Bedrock."""
        # Mock session
        mock_session_instance = MagicMock()
        mock_session_instance.region_name = "us-east-1"
        mock_session.return_value = mock_session_instance

        # Mock boto3.client to return a mock client
        mock_boto_client.return_value = MagicMock()

        # Mock output S3 path
        mock_set_output.return_value = "s3://bucket/output/"

        # Mock recipe builder
        mock_builder_instance = MagicMock()
        mock_builder_instance.build_and_validate.return_value = (
            "/path/to/recipe.yaml",
            "s3://bucket/output/",
            "s3://bucket/data/",
            "dummy-image-uri",
        )
        mock_recipe_builder.return_value = mock_builder_instance

        # Mock execute to return job ARN
        mock_execute.return_value = (
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job"
        )

        # Create Bedrock runtime manager
        bedrock_infra = BedrockRuntimeManager(
            execution_role="arn:aws:iam::123456789012:role/BedrockRole"
        )

        # Create customizer
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=bedrock_infra,
            data_s3_path="s3://bucket/data/",
        )

        # Call train
        result = customizer.train(job_name="test-job")

        # Verify execute was called
        mock_execute.assert_called_once()

        # Get the JobConfig argument
        call_args = mock_execute.call_args
        job_config = call_args[1]["job_config"]

        # Verify method is in JobConfig
        self.assertTrue(hasattr(job_config, "method"))
        self.assertEqual(job_config.method, TrainingMethod.SFT_LORA)

    @patch("boto3.client")
    def test_training_result_to_dict_serializes_all_fields(self, mock_boto_client):
        """Test that _to_dict() serializes all BedrockTrainingResult fields."""
        from amzn_nova_forge.model.model_config import ModelArtifacts
        from amzn_nova_forge.model.result.training_result import (
            BedrockTrainingResult,
        )

        # Create a BedrockTrainingResult
        started_time = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path=None,  # Bedrock doesn't use checkpoint path
            output_s3_path="s3://bucket/output/",
        )

        result = BedrockTrainingResult(
            job_id="arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
            started_time=started_time,
            method=TrainingMethod.SFT_LORA,
            model_artifacts=model_artifacts,
            model_type=Model.NOVA_MICRO,
        )

        # Serialize to dict
        result_dict = result._to_dict()

        # Verify all fields are present
        self.assertIn("job_id", result_dict)
        self.assertIn("started_time", result_dict)
        self.assertIn("method", result_dict)
        self.assertIn("model_artifacts", result_dict)
        self.assertIn("model_type", result_dict)

        # Verify field values
        self.assertEqual(
            result_dict["job_id"],
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        )
        self.assertEqual(result_dict["started_time"], "2026-03-10T12:00:00+00:00")
        self.assertEqual(result_dict["method"], "sft_lora")
        self.assertEqual(result_dict["model_type"], "NOVA_MICRO")

        # Verify model_artifacts is serialized
        self.assertIsInstance(result_dict["model_artifacts"], dict)
        self.assertIsNone(result_dict["model_artifacts"]["checkpoint_s3_path"])
        self.assertEqual(
            result_dict["model_artifacts"]["output_s3_path"], "s3://bucket/output/"
        )

    @patch("boto3.client")
    def test_training_result_from_dict_deserializes_all_fields(self, mock_boto_client):
        """Test that _from_dict() deserializes all BedrockTrainingResult fields."""
        from amzn_nova_forge.model.model_config import ModelArtifacts
        from amzn_nova_forge.model.result.training_result import (
            BedrockTrainingResult,
        )

        # Create a serialized dict
        result_dict = {
            "job_id": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
            "started_time": "2026-03-10T12:00:00+00:00",
            "method": "sft_lora",
            "model_artifacts": {
                "checkpoint_s3_path": None,
                "output_s3_path": "s3://bucket/output/",
            },
            "model_type": "NOVA_MICRO",
        }

        # Deserialize from dict
        result = BedrockTrainingResult._from_dict(result_dict)

        # Verify all fields are restored
        self.assertEqual(
            result.job_id,
            "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/test-job",
        )
        self.assertEqual(
            result.started_time, datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.model_type, Model.NOVA_MICRO)

        # Verify model_artifacts is restored
        self.assertIsInstance(result.model_artifacts, ModelArtifacts)
        self.assertIsNone(result.model_artifacts.checkpoint_s3_path)
        self.assertEqual(result.model_artifacts.output_s3_path, "s3://bucket/output/")

    @patch("boto3.client")
    def test_training_result_serialization_roundtrip_preserves_data(
        self, mock_boto_client
    ):
        """Test that serialization followed by deserialization preserves all data."""
        from amzn_nova_forge.model.model_config import ModelArtifacts
        from amzn_nova_forge.model.result.training_result import (
            BedrockTrainingResult,
        )

        # Create original result
        started_time = datetime(2026, 3, 10, 15, 30, 45, tzinfo=timezone.utc)
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path=None,
            output_s3_path="s3://my-bucket/my-output/",
        )

        original = BedrockTrainingResult(
            job_id="arn:aws:bedrock:us-west-2:123456789012:model-customization-job/my-job",
            started_time=started_time,
            method=TrainingMethod.RFT_LORA,
            model_artifacts=model_artifacts,
            model_type=Model.NOVA_LITE,
        )

        # Serialize and deserialize
        serialized = original._to_dict()
        restored = BedrockTrainingResult._from_dict(serialized)

        # Verify all data is preserved
        self.assertEqual(restored.job_id, original.job_id)
        self.assertEqual(restored.started_time, original.started_time)
        self.assertEqual(restored.method, original.method)
        self.assertEqual(restored.model_type, original.model_type)
        self.assertEqual(
            restored.model_artifacts.checkpoint_s3_path,
            original.model_artifacts.checkpoint_s3_path,
        )
        self.assertEqual(
            restored.model_artifacts.output_s3_path,
            original.model_artifacts.output_s3_path,
        )


if __name__ == "__main__":
    unittest.main()
