import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.result.job_result import (
    JobStatus,
    SMHPStatusManager,
    SMTJStatusManager,
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
    @patch("amzn_nova_customization_sdk.model.result.job_result.logger")
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
    @patch("amzn_nova_customization_sdk.model.result.job_result.logger")
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
        get_job_result.stdout = json.dumps(
            {"Status": {"conditions": [{"type": "Succeeded"}]}}
        )

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
        get_job_result.stdout = json.dumps(
            {"Status": {"conditions": [{"type": "Failed"}]}}
        )

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
    @patch("amzn_nova_customization_sdk.model.result.job_result.logger")
    def test_get_job_status_connect_cluster_error(self, mock_logger, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "hyperpod")

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status, "Unknown")
        mock_logger.error.assert_called_once()

    @patch("subprocess.run")
    @patch("amzn_nova_customization_sdk.model.result.job_result.logger")
    def test_get_job_status_json_decode_error(self, mock_logger, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = "invalid json"

        mock_run.side_effect = [connect_result, get_job_result]

        status, raw_status = self.manager.get_job_status("test-job")

        self.assertEqual(status, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status, "Unknown")
        mock_logger.error.assert_called_once()

    @patch("subprocess.run")
    def test_get_job_status_caching(self, mock_run):
        connect_result = Mock()
        connect_result.stderr = ""

        get_job_result = Mock()
        get_job_result.stdout = json.dumps(
            {"Status": {"conditions": [{"type": "Succeeded"}]}}
        )

        mock_run.side_effect = [connect_result, get_job_result]

        # First call
        status1, raw_status1 = self.manager.get_job_status("test-job")
        self.assertEqual(status1, JobStatus.COMPLETED)

        # Second call should use cache
        status2, raw_status2 = self.manager.get_job_status("test-job")
        self.assertEqual(status2, JobStatus.COMPLETED)

        # Should only call subprocess twice (connect + get-job) for first call, then cache
        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
