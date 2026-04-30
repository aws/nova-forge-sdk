# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_forge.core.result.job_result import (
    JobStatus,
    SMHPStatusManager,
    SMTJStatusManager,
)


class TestBaseJobResultSerialization(unittest.TestCase):
    """Test BaseJobResult dump/load functionality"""

    def test_baseresult_load_preserves_job_cache_hash(self):
        """Test that BaseJobResult.load() preserves job cache hash"""
        from amzn_nova_forge.core.enums import Model, TrainingMethod
        from amzn_nova_forge.core.result import BaseJobResult
        from amzn_nova_forge.core.result.training_result import (
            SMTJTrainingResult,
        )
        from amzn_nova_forge.core.types import ModelArtifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock boto3.client for the entire test
            with patch("boto3.client") as mock_boto:
                mock_client = MagicMock()
                mock_client.describe_training_job.return_value = {
                    "TrainingJobStatus": "Completed",
                    "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
                    "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
                }
                mock_boto.return_value = mock_client

                # Create a real result
                original_result = SMTJTrainingResult(
                    job_id="load-test-123",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_client,
                )

                # Add job cache hash
                test_hash = "test_hash:12345,param:abcde"
                original_result._job_cache_hash = test_hash

                # Save using dump method
                file_path = Path(temp_dir) / "test_result.json"
                original_result.dump(str(temp_dir), "test_result.json")

                # Load it back using BaseJobResult.load()
                loaded_result = BaseJobResult.load(str(file_path))

                # Verify the hash is preserved
                self.assertTrue(
                    hasattr(loaded_result, "_job_cache_hash"),
                    "Loaded result should have job cache hash",
                )
                self.assertEqual(
                    loaded_result._job_cache_hash,
                    test_hash,
                    f"Hash mismatch: expected '{test_hash}', got '{loaded_result._job_cache_hash}'",
                )
                self.assertEqual(
                    loaded_result.job_id,
                    "load-test-123",
                    f"Job ID mismatch: expected 'load-test-123', got '{loaded_result.job_id}'",
                )


class TestJobStatus(unittest.TestCase):
    def test_job_status_values(self):
        self.assertEqual(JobStatus.IN_PROGRESS.value, "InProgress")
        self.assertEqual(JobStatus.COMPLETED.value, "Completed")
        self.assertEqual(JobStatus.FAILED.value, "Failed")

    def test_job_status_aliases(self):
        self.assertEqual(JobStatus("Created"), JobStatus.IN_PROGRESS)
        self.assertEqual(JobStatus("Running"), JobStatus.IN_PROGRESS)
        self.assertEqual(JobStatus("Succeeded"), JobStatus.COMPLETED)

    def test_job_status_unknown_maps_to_failed(self):
        self.assertEqual(JobStatus("Stopped"), JobStatus.FAILED)
        self.assertEqual(JobStatus("Unknown"), JobStatus.FAILED)
        self.assertEqual(JobStatus("Cancelled"), JobStatus.FAILED)


class TestSMTJStatusManager(unittest.TestCase):
    def setUp(self):
        self.mock_sagemaker_client = Mock()
        self.manager = SMTJStatusManager(self.mock_sagemaker_client)

    def test_init_with_client(self):
        manager = SMTJStatusManager(self.mock_sagemaker_client)
        self.assertEqual(manager._sagemaker_client, self.mock_sagemaker_client)
        self.assertEqual(manager._job_status, JobStatus.IN_PROGRESS)
        self.assertEqual(manager._raw_status, JobStatus.IN_PROGRESS.value)

    @patch("boto3.client")
    def test_init_without_client(self, mock_boto3_client):
        manager = SMTJStatusManager()
        mock_boto3_client.assert_called_once_with("sagemaker")

    def test_get_job_status_in_progress(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status, "InProgress")
        self.mock_sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )

    def test_get_job_status_completed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw_status, "Completed")

    def test_get_job_status_failed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(raw_status, "Failed")

    def test_get_job_status_caching_completed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # First call
        status1, raw_status1 = self.manager.get_job_status("test-job")
        self.assertEqual(status1, JobStatus.COMPLETED)

        # Second call should use cache
        status2, raw_status2 = self.manager.get_job_status("test-job")
        self.assertEqual(status2, JobStatus.COMPLETED)

        # Should only call API once
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

    def test_get_job_status_caching_failed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        # First call
        status1, raw_status1 = self.manager.get_job_status("test-job")
        self.assertEqual(status1, JobStatus.FAILED)

        # Second call should use cache
        status2, raw_status2 = self.manager.get_job_status("test-job")
        self.assertEqual(status2, JobStatus.FAILED)

        # Should only call API once
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

    def test_get_job_status_no_caching_in_progress(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        # First call
        status1, raw_status1 = self.manager.get_job_status("test-job")
        self.assertEqual(status1, JobStatus.IN_PROGRESS)

        # Second call should call API again
        status2, raw_status2 = self.manager.get_job_status("test-job")
        self.assertEqual(status2, JobStatus.IN_PROGRESS)

        # Should call API twice
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 2)


class TestSMHPStatusManager(unittest.TestCase):
    def setUp(self):
        self.manager = SMHPStatusManager("test-cluster", "test-namespace")

    def test_init(self):
        self.assertEqual(self.manager.cluster_name, "test-cluster")
        self.assertEqual(self.manager.namespace, "test-namespace")
        self.assertEqual(self.manager._job_status, JobStatus.IN_PROGRESS)

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_connect_cluster_success(self, mock_logger, mock_run):
        mock_result = Mock()
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        self.manager._connect_cluster()

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                "test-cluster",
                "--namespace",
                "test-namespace",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        mock_logger.info.assert_called_once_with(
            "Successfully connected to HyperPod cluster 'test-cluster' in namespace 'test-namespace'."
        )

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_connect_cluster_error(self, mock_logger, mock_run):
        mock_result = Mock()
        mock_result.stderr = "Connection failed"
        mock_run.return_value = mock_result

        with self.assertRaises(Exception):
            self.manager._connect_cluster()

        mock_logger.error.assert_called_once_with(
            "Unable to connect to HyperPod cluster test-cluster: Connection failed"
        )

    @patch("subprocess.run")
    def test_get_job_status_succeeded(self, mock_run):
        # Mock connect-cluster call
        connect_result = Mock()
        connect_result.stderr = ""

        # Mock get-job call
        get_job_result = Mock()
        get_job_result.stdout = json.dumps({"Status": {"conditions": [{"type": "Succeeded"}]}})

        mock_run.side_effect = [connect_result, get_job_result]

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw_status, "Succeeded")
        self.assertEqual(mock_run.call_count, 2)
        mock_run.assert_any_call(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                "test-cluster",
                "--namespace",
                "test-namespace",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        mock_run.assert_any_call(
            ["hyperpod", "get-job", "--job-name", "test-job"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_job_status_failed(self, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps({"Status": {"conditions": [{"type": "Failed"}]}})

        mock_run.side_effect = [connect_result, get_job_result]

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(raw_status, "Failed")

    @patch("subprocess.run")
    def test_get_job_status_pending_null_status(self, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps({"Status": None})

        mock_run.side_effect = [connect_result, get_job_result]

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status, "Pending")

    @patch("subprocess.run")
    def test_get_job_status_pending_no_conditions(self, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps({"Status": {"conditions": []}})

        mock_run.side_effect = [connect_result, get_job_result]

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status, "Pending")

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_get_job_status_connect_cluster_error(self, mock_logger, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "hyperpod")

        with self.assertRaises(RuntimeError) as context:
            self.manager.get_job_status("test-job")

        # Verify error message includes helpful context
        error_msg = str(context.exception)
        self.assertIn("Failed to get job status for test-job", error_msg)
        self.assertIn("insufficient permissions", error_msg.lower())
        self.assertIn("EKS access entry", error_msg)
        mock_logger.error.assert_called_once()
        self.assertIsNotNone(context.exception.__cause__)

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_get_job_status_json_decode_error(self, mock_logger, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = "invalid json"

        mock_run.side_effect = [connect_result, get_job_result]

        with self.assertRaises(RuntimeError) as context:
            self.manager.get_job_status("test-job")

        # Verify error message includes helpful context
        error_msg = str(context.exception)
        self.assertIn("Failed to get job status for test-job", error_msg)
        self.assertIn("not in the expected format", error_msg)
        mock_logger.error.assert_called_once()
        self.assertIsNotNone(context.exception.__cause__)

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_get_job_status_permission_denied_error(self, mock_logger, mock_run):
        # Simulate permission denied error from HPCLI
        error = subprocess.CalledProcessError(
            1,
            "hyperpod",
            stderr="Error: User is not authorized to perform: eks:DescribeCluster",
        )
        mock_run.side_effect = error

        with self.assertRaises(RuntimeError) as context:
            self.manager.get_job_status("test-job")

        error_msg = str(context.exception)

        self.assertIn("Failed to get job status for test-job", error_msg)
        self.assertIn("insufficient permissions", error_msg.lower())
        self.assertIn("EKS access entry", error_msg)
        self.assertIn("Details:", error_msg)
        self.assertIn("User is not authorized to perform: eks:DescribeCluster", error_msg)

        mock_logger.error.assert_called_once()
        self.assertIsNotNone(context.exception.__cause__)

    @patch("subprocess.run")
    @patch("amzn_nova_forge.core.result.job_result.logger")
    def test_get_job_status_json_decode_error_message(self, mock_logger, mock_run):
        """Test that JSON decode errors have appropriate error message (not permission hint)"""
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = "invalid json"

        mock_run.side_effect = [connect_result, get_job_result]

        with self.assertRaises(RuntimeError) as context:
            self.manager.get_job_status("test-job")

        error_msg = str(context.exception)
        self.assertIn("not in the expected format", error_msg)
        self.assertNotIn("permission", error_msg.lower())
        self.assertNotIn("EKS access entry", error_msg)

    @patch("subprocess.run")
    def test_get_job_status_caching(self, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps({"Status": {"conditions": [{"type": "Succeeded"}]}})

        mock_run.side_effect = [connect_result, get_job_result]

        # First call
        status1, raw_status1 = self.manager.get_job_status("test-job")
        self.assertEqual(status1, JobStatus.COMPLETED)

        # Second call should use cache
        status2, raw_status2 = self.manager.get_job_status("test-job")
        self.assertEqual(status2, JobStatus.COMPLETED)

        # Should only call subprocess twice (connect + get-job) for first call, then cache
        self.assertEqual(mock_run.call_count, 2)


class TestResolveStartTime(unittest.TestCase):
    """Test that job results can resolve start_time from the platform when not provided."""

    def test_smtj_resolve_start_time(self):
        """SMTJStatusManager resolves start time from describe_training_job."""
        mock_client = Mock()
        training_start = datetime(2026, 3, 13, 12, 0, 0)
        mock_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingStartTime": training_start,
        }
        manager = SMTJStatusManager(mock_client)
        result = manager.resolve_start_time("test-job")
        self.assertEqual(result, training_start)

    def test_smtj_resolve_start_time_falls_back_to_creation_time(self):
        """Falls back to CreationTime when TrainingStartTime is absent."""
        mock_client = Mock()
        creation_time = datetime(2026, 3, 13, 11, 59, 0)
        mock_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
            "CreationTime": creation_time,
        }
        manager = SMTJStatusManager(mock_client)
        result = manager.resolve_start_time("test-job")
        self.assertEqual(result, creation_time)

    def test_smtj_resolve_start_time_raises_on_missing(self):
        """Raises ValueError when neither timestamp is available."""
        mock_client = Mock()
        mock_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
        }
        manager = SMTJStatusManager(mock_client)
        with self.assertRaises(ValueError):
            manager.resolve_start_time("test-job")

    @patch("subprocess.run")
    def test_smhp_resolve_start_time(self, mock_run):
        """SMHPStatusManager resolves start time from hyperpod get-job --verbose."""
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps(
            {
                "Status": {"startTime": "2026-03-13T12:00:00Z"},
                "CreationTimestamp": "2026-03-13T11:59:00Z",
            }
        )

        mock_run.side_effect = [connect_result, get_job_result]

        manager = SMHPStatusManager("test-cluster", "test-namespace")
        result = manager.resolve_start_time("test-job")

        self.assertEqual(
            result,
            datetime(2026, 3, 13, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc),
        )
        mock_run.assert_any_call(
            ["hyperpod", "get-job", "--job-name", "test-job", "--verbose"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_smhp_resolve_start_time_falls_back_to_creation(self, mock_run):
        """Falls back to CreationTimestamp when Status.startTime is absent."""
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps(
            {
                "Status": {},
                "Metadata": {"CreationTimestamp": "2026-03-13T11:59:00Z"},
            }
        )

        mock_run.side_effect = [connect_result, get_job_result]

        manager = SMHPStatusManager("test-cluster", "test-namespace")
        result = manager.resolve_start_time("test-job")

        self.assertEqual(
            result,
            datetime(2026, 3, 13, 11, 59, 0, tzinfo=__import__("datetime").timezone.utc),
        )

    @patch("subprocess.run")
    def test_smhp_resolve_start_time_raises_on_failure(self, mock_run):
        """Raises ValueError when hyperpod get-job fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "hyperpod")

        manager = SMHPStatusManager("test-cluster", "test-namespace")
        with self.assertRaises(ValueError):
            manager.resolve_start_time("test-job")

    def test_smtj_result_without_started_time(self):
        """SMTJTrainingResult resolves start_time when constructed with only job_id."""
        from amzn_nova_forge.core.enums import Model, TrainingMethod
        from amzn_nova_forge.core.result.training_result import SMTJTrainingResult
        from amzn_nova_forge.core.types import ModelArtifacts

        mock_client = Mock()
        training_start = datetime(2026, 3, 13, 12, 0, 0)
        mock_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingStartTime": training_start,
        }

        result = SMTJTrainingResult(
            job_id="resolve-test-123",
            started_time=None,
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=ModelArtifacts(
                checkpoint_s3_path="s3://test/checkpoint",
                output_s3_path="s3://test/output",
            ),
            sagemaker_client=mock_client,
        )

        self.assertEqual(result.started_time, training_start)
        self.assertEqual(result.job_id, "resolve-test-123")


if __name__ == "__main__":
    unittest.main()
