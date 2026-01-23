import json
import tarfile
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.result import (
    JobStatus,
    SMTJBatchInferenceResult,
)


class TestSMTJBatchInferenceResult(unittest.TestCase):
    def setUp(self):
        self.job_id = "test-job-123"
        self.started_time = datetime.now()
        self.inference_output_path = "s3://test-bucket/output/test-job/output.tar.gz"
        self.mock_sagemaker_client = Mock()

        self.result = SMTJBatchInferenceResult(
            job_id=self.job_id,
            started_time=self.started_time,
            inference_output_path=self.inference_output_path,
            sagemaker_client=self.mock_sagemaker_client,
        )

    def tearDown(self):
        # Clean up cache dir
        if hasattr(self.result, "_cached_results_dir"):
            self.result.clean()

    def test_get_job_status_completed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        status, raw_status = self.result.get_job_status()

        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertEqual(raw_status, "Completed")

    def test_get_job_status_unknown_maps_to_failed(self):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Stopped"
        }

        status, raw_status = self.result.get_job_status()

        self.assertEqual(status, JobStatus.FAILED)
        self.assertEqual(raw_status, "Stopped")

    @patch("builtins.print")
    def test_show_in_progress(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Job 'test-job-123' still running in progress. Please wait until the job is completed."
        )

    @patch("builtins.print")
    def test_show_failed(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Cannot show inference result. Job 'test-job-123' in Failed status."
        )

    @patch("builtins.print")
    def test_show_stopped(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Stopped"
        }

        self.result.show()

        mock_print.assert_called_with(
            "Cannot show inference result. Job 'test-job-123' in Stopped status."
        )

    @patch("boto3.client")
    @patch("builtins.print")
    def test_show_completed_with_results(self, mock_print, mock_boto3_client):
        # Mock SageMaker response
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # Mock S3 client
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create mock tar.gz file with results
        test_results = {
            "prompt": "'role': 'system', 'content': 'test_system' 'role': 'user', 'content': 'test_user'",
            "inference": "test_inference",
            "gold": "test_gold",
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create tar.gz file in source directory
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                # Write JSONL file
                results_file = results_dir / "inference_output.jsonl"
                with open(results_file, "w") as f:
                    f.write(json.dumps(test_results) + "\n")

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                # Mock S3 download to copy from source to cache directory
                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                # Mock mkdtemp to return cache_dir
                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    self.result.show()

                # Expected result
                expected_result = {
                    "system": "test_system",
                    "query": "test_user",
                    "gold_response": "test_gold",
                    "inference_response": "test_inference",
                }

                # Verify print calls
                expected_calls = [
                    unittest.mock.call("Job 'test-job-123' completed successfully."),
                    unittest.mock.call(f"\nInference Results for job_id=test-job-123:"),
                    unittest.mock.call(json.dumps(expected_result, ensure_ascii=False)),
                ]
                mock_print.assert_has_calls(expected_calls)

    @patch("boto3.client")
    @patch("builtins.print")
    def test_show_completed_no_results_file(self, mock_print, mock_boto3_client):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create empty tar.gz file in source directory
                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass  # Empty tar file

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    self.result.show()

                mock_print.assert_any_call(
                    "No inference output jsonl file found for job test-job-123"
                )

    @patch("boto3.client")
    @patch("builtins.print")
    def test_caching_mechanism(self, mock_print, mock_boto3_client):
        """Test caching mechanism, second .show() call should not re-download the file"""
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        test_results = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create test file in source dir
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "inference_output.json"
                with open(results_file, "w") as f:
                    json.dump(test_results, f)

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    # First call, should call s3 to download file
                    self.result.show()
                    self.assertEqual(mock_s3_client.download_file.call_count, 1)

                    # Second call, should use cache
                    self.result.show()
                    self.assertEqual(
                        mock_s3_client.download_file.call_count, 1
                    )  # Should still 1, no increase

    def test_clean_method(self):
        """Test temp cache dir was cleaned"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate cache dir
            self.result._cached_results_dir = temp_dir
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")

            self.assertTrue(test_file.exists())

            # Call clean method
            self.result.clean()

            # Assert cache dir has been cleaned
            self.assertFalse(Path(temp_dir).exists())
            self.assertIsNone(self.result._cached_results_dir)

    def test_job_status_caching_completed(self):
        # Return COMPLETED at first call
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.COMPLETED)
        self.assertEqual(raw_status1, "Completed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Second call - should use cache rather than making api call
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.COMPLETED)
        self.assertEqual(raw_status2, "Completed")
        self.assertEqual(
            self.mock_sagemaker_client.describe_training_job.call_count, 1
        )  # Still 1

    def test_job_status_caching_in_progress_then_completed(self):
        """Test job status change from IN_PROGRESS to COMPLETED"""
        # First call return IN_PROGRESS
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.IN_PROGRESS)
        self.assertEqual(raw_status1, "InProgress")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Change job status to COMPLETED
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        # Second call - should call API to get new status
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.COMPLETED)
        self.assertEqual(raw_status2, "Completed")
        self.assertEqual(
            self.mock_sagemaker_client.describe_training_job.call_count, 2
        )  # Should increase to 2

        # Third call - should use cache now
        status3, raw_status3 = self.result.get_job_status()
        self.assertEqual(status3, JobStatus.COMPLETED)
        self.assertEqual(raw_status3, "Completed")
        self.assertEqual(
            self.mock_sagemaker_client.describe_training_job.call_count, 2
        )  # Still 2

    def test_job_status_caching_failed_no_cache(self):
        """Test FAILED status should be cached"""
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        # First call
        status1, raw_status1 = self.result.get_job_status()
        self.assertEqual(status1, JobStatus.FAILED)
        self.assertEqual(raw_status1, "Failed")
        self.assertEqual(self.mock_sagemaker_client.describe_training_job.call_count, 1)

        # Second call - should use cache
        status2, raw_status2 = self.result.get_job_status()
        self.assertEqual(status2, JobStatus.FAILED)
        self.assertEqual(raw_status2, "Failed")
        self.assertEqual(
            self.mock_sagemaker_client.describe_training_job.call_count, 1
        )  # Still 1)

    @patch("boto3.client")
    @patch("builtins.print")
    def test_get_completed_without_save(self, mock_print, mock_boto3_client):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        test_results = {
            "prompt": "'role': 'system', 'content': 'test_system' 'role': 'user', 'content': 'test_user'",
            "inference": "test_inference",
            "gold": "test_gold",
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "inference_output.jsonl"
                with open(results_file, "w") as f:
                    f.write(json.dumps(test_results) + "\n")

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    result = self.result.get()

                expected_result = {
                    "system": "test_system",
                    "query": "test_user",
                    "gold_response": "test_gold",
                    "inference_response": "test_inference",
                }

                self.assertIn("inference_results", result)
                self.assertEqual(len(result["inference_results"]), 1)
                self.assertEqual(result["inference_results"][0], expected_result)

                mock_print.assert_any_call("Job 'test-job-123' completed successfully.")

    @patch("boto3.client")
    @patch("builtins.print")
    def test_get_completed_with_s3_save(self, mock_print, mock_boto3_client):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        test_results = {
            "prompt": "'role': 'system', 'content': 'test_system' 'role': 'user', 'content': 'test_user'",
            "inference": "test_inference",
            "gold": "test_gold",
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "inference_output.jsonl"
                with open(results_file, "w") as f:
                    f.write(json.dumps(test_results) + "\n")

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    s3_save_path = "s3://test-bucket/results/output.jsonl"
                    result = self.result.get(s3_path=s3_save_path)

                mock_s3_client.put_object.assert_called_once()
                call_args = mock_s3_client.put_object.call_args

                self.assertEqual(call_args[1]["Bucket"], "test-bucket")
                self.assertEqual(call_args[1]["Key"], "results/output.jsonl")
                self.assertEqual(call_args[1]["ContentType"], "application/jsonlines")

                body_content = call_args[1]["Body"].decode("utf-8")
                lines = body_content.strip().split("\n")
                self.assertEqual(len(lines), 1)
                parsed = json.loads(lines[0])
                self.assertIn("system", parsed)
                self.assertIn("query", parsed)

                mock_print.assert_any_call(
                    f"Successfully saved the results to {s3_save_path}."
                )

    @patch("builtins.print")
    def test_get_completed_with_local_save(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        test_results = {
            "prompt": "'role': 'system', 'content': 'test_system' 'role': 'user', 'content': 'test_user'",
            "inference": "test_inference",
            "gold": "test_gold",
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                with tempfile.TemporaryDirectory() as output_dir:
                    results_dir = Path(source_dir) / "run_name" / "eval_results"
                    results_dir.mkdir(parents=True)

                    results_file = results_dir / "inference_output.jsonl"
                    with open(results_file, "w") as f:
                        f.write(json.dumps(test_results) + "\n")

                    tar_path = Path(source_dir) / "output.tar.gz"
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(results_dir.parent, arcname="run_name")

                    with patch("boto3.client") as mock_boto3:
                        mock_s3_client = Mock()
                        mock_boto3.return_value = mock_s3_client

                        def mock_download_file(bucket, key, local_path):
                            import shutil

                            shutil.copy2(tar_path, local_path)

                        mock_s3_client.download_file.side_effect = mock_download_file

                        with patch("tempfile.mkdtemp", return_value=cache_dir):
                            local_save_path = (
                                Path(output_dir) / "results" / "output.jsonl"
                            )
                            result = self.result.get(s3_path=str(local_save_path))

                    self.assertTrue(local_save_path.exists())

                    with open(local_save_path, "r") as f:
                        content = f.read()
                        lines = content.strip().split("\n")
                        self.assertEqual(len(lines), 1)
                        parsed = json.loads(lines[0])
                        self.assertIn("system", parsed)
                        self.assertIn("query", parsed)

                    mock_print.assert_any_call(
                        f"Successfully saved the results to {local_save_path}."
                    )

    @patch("builtins.print")
    def test_get_in_progress(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }

        result = self.result.get()

        self.assertEqual(result, {})

        mock_print.assert_any_call(
            "Job 'test-job-123' still running in progress. Please wait until the job is completed."
        )

    @patch("builtins.print")
    def test_get_failed(self, mock_print):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed"
        }

        result = self.result.get()

        self.assertEqual(result, {})

        mock_print.assert_any_call(
            "Cannot show inference result. Job 'test-job-123' in Failed status."
        )

    @patch("boto3.client")
    @patch("builtins.print")
    def test_get_completed_no_results_file(self, mock_print, mock_boto3_client):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    pass  # Empty file

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    result = self.result.get()

                self.assertEqual(result, {})

                mock_print.assert_any_call(
                    "No inference output jsonl file found for job test-job-123"
                )

    @patch("boto3.client")
    @patch("builtins.print")
    def test_get_save_error_handling(self, mock_print, mock_boto3_client):
        self.mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        test_results = {
            "prompt": "'role': 'system', 'content': 'test_system' 'role': 'user', 'content': 'test_user'",
            "inference": "test_inference",
            "gold": "test_gold",
        }

        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                results_dir = Path(source_dir) / "run_name" / "eval_results"
                results_dir.mkdir(parents=True)

                results_file = results_dir / "inference_output.jsonl"
                with open(results_file, "w") as f:
                    f.write(json.dumps(test_results) + "\n")

                tar_path = Path(source_dir) / "output.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(results_dir.parent, arcname="run_name")

                def mock_download_file(bucket, key, local_path):
                    import shutil

                    shutil.copy2(tar_path, local_path)

                mock_s3_client.download_file.side_effect = mock_download_file

                from botocore.exceptions import ClientError

                mock_s3_client.put_object.side_effect = ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                    "PutObject",
                )

                with patch("tempfile.mkdtemp", return_value=cache_dir):
                    s3_save_path = "s3://test-bucket/results/output.jsonl"

                    with self.assertRaises(ClientError):
                        self.result.get(s3_path=s3_save_path)

                    error_calls = [
                        call
                        for call in mock_print.call_args_list
                        if "Error saving inference results" in str(call)
                    ]
                    self.assertGreater(len(error_calls), 0)


if __name__ == "__main__":
    unittest.main()
