import json
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.model.result.job_result import JobStatus
from amzn_nova_customization_sdk.util.checkpoint_util import (
    extract_checkpoint_path_from_job_output,
)


class TestExtractCheckpointPath(unittest.TestCase):
    def setUp(self):
        self.output_s3_path = "s3://test-bucket/output"
        self.job_id = "test-job-123"
        self.checkpoint_path = "s3://escrow-bucket/checkpoint/path"
        self.manifest_data = {"checkpoint_s3_bucket": self.checkpoint_path}

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_success(self, mock_boto_client):
        """Test successful checkpoint extraction"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Create a temporary tar.gz file with manifest.json
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w:gz") as tar:
                # Create manifest.json in memory
                manifest_json = json.dumps(self.manifest_data).encode()
                with tempfile.NamedTemporaryFile() as manifest_file:
                    manifest_file.write(manifest_json)
                    manifest_file.flush()
                    tar.add(manifest_file.name, arcname="manifest.json")

            # Mock S3 operations
            mock_s3.head_object.return_value = {}
            # Mock SMHP format to fail (404) so it tries SMTJ format
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "GetObject"
            )
            mock_s3.download_file.side_effect = lambda bucket, key, filename: None

            # Mock the tarfile content by copying our test file
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = tmp_tar.name

                result = extract_checkpoint_path_from_job_output(
                    self.output_s3_path, self.job_id
                )

        self.assertEqual(result, self.checkpoint_path)
        # Verify head_object was called for both SMHP and SMTJ paths
        self.assertEqual(mock_s3.head_object.call_count, 2)
        mock_s3.head_object.assert_any_call(
            Bucket="test-bucket", Key=f"output/{self.job_id}/manifest.json"
        )
        mock_s3.head_object.assert_any_call(
            Bucket="test-bucket", Key=f"output/{self.job_id}/output/output.tar.gz"
        )

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_with_job_result_completed(self, mock_boto_client):
        """Test extraction with completed job result"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock job result
        mock_job_result = MagicMock()
        mock_job_result.get_job_status.return_value = (JobStatus.COMPLETED, "Completed")

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w:gz") as tar:
                manifest_json = json.dumps(self.manifest_data).encode()
                with tempfile.NamedTemporaryFile() as manifest_file:
                    manifest_file.write(manifest_json)
                    manifest_file.flush()
                    tar.add(manifest_file.name, arcname="manifest.json")

            mock_s3.head_object.return_value = {}
            # Mock SMHP format to fail so it tries SMTJ format
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "GetObject"
            )

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = tmp_tar.name

                result = extract_checkpoint_path_from_job_output(
                    self.output_s3_path, self.job_id, mock_job_result
                )

        self.assertEqual(result, self.checkpoint_path)

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_job_not_completed(self, mock_boto_client):
        """Test extraction fails when job is not completed"""
        mock_job_result = MagicMock()
        mock_job_result.get_job_status.return_value = (
            JobStatus.IN_PROGRESS,
            "InProgress",
        )

        with self.assertRaises(Exception) as context:
            extract_checkpoint_path_from_job_output(
                self.output_s3_path, self.job_id, mock_job_result
            )

        self.assertIn("not completed", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_output_not_found(self, mock_boto_client):
        """Test extraction fails when output file doesn't exist"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock both SMHP and SMTJ head_object to fail with 404
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )

        with self.assertRaises(Exception) as context:
            extract_checkpoint_path_from_job_output(self.output_s3_path, self.job_id)

        self.assertIn("Failed to extract manifest", str(context.exception))
        self.assertIn(
            "Job may not be completed or output path is incorrect",
            str(context.exception),
        )

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_permission_denied(self, mock_boto_client):
        """Test extraction fails when access denied"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock both SMHP and SMTJ head_object to fail with 403
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Access Denied"}}, "HeadObject"
        )

        with self.assertRaises(Exception) as context:
            extract_checkpoint_path_from_job_output(self.output_s3_path, self.job_id)

        self.assertIn("Failed to extract manifest", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_permission_denied_download(self, mock_boto_client):
        """Test extraction fails when access denied on download"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # head_object succeeds but both SMHP and SMTJ formats fail
        mock_s3.head_object.return_value = {}
        # SMHP format fails with 404
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "GetObject"
        )
        # SMTJ format fails with 403
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Access Denied"}}, "GetObject"
        )

        with self.assertRaises(Exception) as context:
            extract_checkpoint_path_from_job_output(self.output_s3_path, self.job_id)

        self.assertIn("Failed to extract manifest", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_missing_manifest(self, mock_boto_client):
        """Test extraction fails when manifest.json is missing"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Create tar without manifest.json
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w:gz") as tar:
                with tempfile.NamedTemporaryFile() as dummy_file:
                    dummy_file.write(b"dummy content")
                    dummy_file.flush()
                    tar.add(dummy_file.name, arcname="dummy.txt")

            mock_s3.head_object.return_value = {}
            # SMHP format fails with 404
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "GetObject"
            )

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = tmp_tar.name

                with self.assertRaises(Exception) as context:
                    extract_checkpoint_path_from_job_output(
                        self.output_s3_path, self.job_id
                    )

        self.assertIn("Failed to extract manifest", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_missing_checkpoint_field(self, mock_boto_client):
        """Test extraction fails when checkpoint_s3_bucket field is missing"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Create manifest without checkpoint_s3_bucket field
        invalid_manifest = {"other_field": "value"}

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w:gz") as tar:
                manifest_json = json.dumps(invalid_manifest).encode()
                with tempfile.NamedTemporaryFile() as manifest_file:
                    manifest_file.write(manifest_json)
                    manifest_file.flush()
                    tar.add(manifest_file.name, arcname="manifest.json")

            mock_s3.head_object.return_value = {}
            # SMHP format fails with 404
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "GetObject"
            )

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = tmp_tar.name

                with self.assertRaises(Exception) as context:
                    extract_checkpoint_path_from_job_output(
                        self.output_s3_path, self.job_id
                    )

        self.assertIn("checkpoint_s3_bucket not found", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.checkpoint_util.boto3.client")
    def test_extract_checkpoint_empty_checkpoint_path(self, mock_boto_client):
        """Test extraction fails when checkpoint path is empty"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Create manifest with empty checkpoint path
        empty_manifest = {"checkpoint_s3_bucket": ""}

        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            with tarfile.open(tmp_tar.name, "w:gz") as tar:
                manifest_json = json.dumps(empty_manifest).encode()
                with tempfile.NamedTemporaryFile() as manifest_file:
                    manifest_file.write(manifest_json)
                    manifest_file.flush()
                    tar.add(manifest_file.name, arcname="manifest.json")

            mock_s3.head_object.return_value = {}
            # SMHP format fails with 404
            mock_s3.get_object.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "GetObject"
            )

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = tmp_tar.name

                with self.assertRaises(Exception) as context:
                    extract_checkpoint_path_from_job_output(
                        self.output_s3_path, self.job_id
                    )

        self.assertIn("checkpoint_s3_bucket is empty", str(context.exception))


if __name__ == "__main__":
    unittest.main()
