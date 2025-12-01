import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.model_config import ModelArtifacts
from amzn_nova_customization_sdk.model.model_enums import Platform, TrainingMethod
from amzn_nova_customization_sdk.model.result.job_result import (
    JobStatus,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_customization_sdk.model.result.training_result import (
    SMHPTrainingResult,
    SMTJTrainingResult,
)


class TestSMTJTrainResult(unittest.TestCase):
    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/output/",
        )
        self.mock_sagemaker_client = Mock()

    def test_init_with_default_client(self):
        """Test initialization with default SageMaker client"""
        with patch("boto3.client") as mock_boto3:
            result = SMTJTrainingResult(
                job_id="test-job-123",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=TrainingMethod.SFT_LORA,
                model_artifacts=self.model_artifacts,
            )

            self.assertEqual(result.job_id, "test-job-123")
            self.assertEqual(result.method, TrainingMethod.SFT_LORA)
            self.assertEqual(result.model_artifacts, self.model_artifacts)
            self.assertEqual(result.platform, Platform.SMTJ)
            self.assertIsInstance(result.status_manager, SMTJStatusManager)
            mock_boto3.assert_called_once_with("sagemaker")

    def test_init_with_custom_client(self):
        """Test initialization with custom SageMaker client"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_FULLRANK,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        self.assertEqual(result._sagemaker_client, self.mock_sagemaker_client)
        self.assertEqual(result.method, TrainingMethod.SFT_FULLRANK)

    def test_create_status_manager(self):
        """Test status manager creation"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        status_manager = result._create_status_manager()
        self.assertIsInstance(status_manager, SMTJStatusManager)
        self.assertEqual(status_manager._sagemaker_client, self.mock_sagemaker_client)

    def test_to_dict(self):
        """Test dictionary conversion"""
        started_time = datetime(2024, 1, 1, 12, 0, 0)
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=started_time,
            method=TrainingMethod.RFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        result_dict = result._to_dict()
        expected = {
            "job_id": "test-job-123",
            "started_time": started_time.isoformat(),
            "method": TrainingMethod.RFT_LORA.value,
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
        }

        self.assertEqual(result_dict, expected)

    def test_from_dict(self):
        """Test object creation from dictionary"""
        data = {
            "job_id": "test-job-456",
            "started_time": "2024-01-01T12:00:00",
            "method": "sft_lora",
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
        }

        with patch("boto3.client"):
            result = SMTJTrainingResult._from_dict(data)

            self.assertEqual(result.job_id, "test-job-456")
            self.assertEqual(result.started_time, datetime(2024, 1, 1, 12, 0, 0))
            self.assertEqual(result.method, TrainingMethod.SFT_LORA)
            self.assertEqual(
                result.model_artifacts.checkpoint_s3_path, "s3://bucket/checkpoint/"
            )
            self.assertEqual(
                result.model_artifacts.output_s3_path, "s3://bucket/output/"
            )

    def test_get_method(self):
        """Test get method returns dictionary"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        result_data = result.get()
        self.assertIsInstance(result_data, dict)
        self.assertEqual(result_data["job_id"], "test-job-123")

    def test_show_method(self):
        """Test show method prints result"""
        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        with patch("builtins.print") as mock_print:
            result.show()
            mock_print.assert_called_once()
            printed_args = str(mock_print.call_args)
            self.assertIn("test-job-123", printed_args)

    def test_dump_and_load_roundtrip(self):
        """Test dump and load roundtrip"""
        original_result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            sagemaker_client=self.mock_sagemaker_client,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir)
            original_result.dump(str(file_path))

            expected_file = (
                file_path
                / f"{original_result.job_id}_{original_result.platform.value}.json"
            )
            self.assertTrue(expected_file.exists())

            with patch("boto3.client"):
                loaded_result = SMTJTrainingResult.load(str(expected_file))

                self.assertEqual(loaded_result.job_id, original_result.job_id)
                self.assertEqual(loaded_result.method, original_result.method)
                self.assertEqual(
                    loaded_result.model_artifacts.output_s3_path,
                    original_result.model_artifacts.output_s3_path,
                )


class TestSMHPTrainResult(unittest.TestCase):
    def setUp(self):
        self.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/output/",
        )

    def test_init_with_default_namespace(self):
        """Test initialization with default namespace"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
        )

        self.assertEqual(result.job_id, "test-job-123")
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "kubeflow")
        self.assertEqual(result.platform, Platform.SMHP)
        self.assertIsInstance(result.status_manager, SMHPStatusManager)

    def test_init_with_custom_namespace(self):
        """Test initialization with custom namespace"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.RFT,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="custom-namespace",
        )

        self.assertEqual(result.namespace, "custom-namespace")
        self.assertEqual(result.method, TrainingMethod.RFT)

    def test_create_status_manager(self):
        """Test status manager creation"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
        )

        status_manager = result._create_status_manager()
        self.assertIsInstance(status_manager, SMHPStatusManager)
        self.assertEqual(status_manager.cluster_name, "test-cluster")
        self.assertEqual(status_manager.namespace, "test-namespace")

    def test_to_dict(self):
        """Test dictionary conversion"""
        started_time = datetime(2024, 1, 1, 12, 0, 0)
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=started_time,
            method=TrainingMethod.EVALUATION,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
        )

        result_dict = result._to_dict()
        expected = {
            "job_id": "test-job-123",
            "started_time": started_time.isoformat(),
            "method": TrainingMethod.EVALUATION.value,
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
            "cluster_name": "test-cluster",
            "namespace": "test-namespace",
        }

        self.assertEqual(result_dict, expected)

    def test_from_dict(self):
        """Test object creation from dictionary"""
        data = {
            "job_id": "test-job-456",
            "started_time": "2024-01-01T12:00:00",
            "method": "rft_lora",
            "model_artifacts": {
                "checkpoint_s3_path": "s3://bucket/checkpoint/",
                "output_s3_path": "s3://bucket/output/",
            },
            "cluster_name": "test-cluster",
            "namespace": "test-namespace",
        }

        result = SMHPTrainingResult._from_dict(data)

        self.assertEqual(result.job_id, "test-job-456")
        self.assertEqual(result.started_time, datetime(2024, 1, 1, 12, 0, 0))
        self.assertEqual(result.method, TrainingMethod.RFT_LORA)
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "test-namespace")
        self.assertEqual(
            result.model_artifacts.checkpoint_s3_path, "s3://bucket/checkpoint/"
        )

    def test_get_method(self):
        """Test get method returns dictionary"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
        )

        result_data = result.get()
        self.assertIsInstance(result_data, dict)
        self.assertEqual(result_data["job_id"], "test-job-123")
        self.assertEqual(result_data["cluster_name"], "test-cluster")

    def test_show_method(self):
        """Test show method prints result"""
        result = SMHPTrainingResult(
            job_id="test-job-123",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.SFT_LORA,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
        )

        with patch("builtins.print") as mock_print:
            result.show()
            mock_print.assert_called_once()
            printed_args = str(mock_print.call_args)
            self.assertIn("test-job-123", printed_args)

    def test_dump_and_load_roundtrip(self):
        """Test dump and load roundtrip"""
        original_result = SMHPTrainingResult(
            job_id="test-job-456",
            started_time=datetime(2024, 1, 1, 12, 0, 0),
            method=TrainingMethod.RFT,
            model_artifacts=self.model_artifacts,
            cluster_name="test-cluster",
            namespace="test-namespace",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir)
            original_result.dump(str(file_path))

            expected_file = (
                file_path
                / f"{original_result.job_id}_{original_result.platform.value}.json"
            )
            self.assertTrue(expected_file.exists())

            loaded_result = SMHPTrainingResult.load(str(expected_file))

            self.assertEqual(loaded_result.job_id, original_result.job_id)
            self.assertEqual(loaded_result.cluster_name, original_result.cluster_name)
            self.assertEqual(loaded_result.namespace, original_result.namespace)
            self.assertEqual(loaded_result.method, original_result.method)

    def test_all_training_methods(self):
        """Test all training method enums work correctly"""
        methods = [
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULLRANK,
            TrainingMethod.RFT,
            TrainingMethod.RFT_LORA,
            TrainingMethod.EVALUATION,
        ]

        for method in methods:
            result = SMHPTrainingResult(
                job_id=f"test-job-{method.value}",
                started_time=datetime(2024, 1, 1, 12, 0, 0),
                method=method,
                model_artifacts=self.model_artifacts,
                cluster_name="test-cluster",
            )

            self.assertEqual(result.method, method)
            result_dict = result._to_dict()
            self.assertEqual(result_dict["method"], method.value)


if __name__ == "__main__":
    unittest.main()
