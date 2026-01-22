import json
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.util.checkpoint_util import (
    extract_checkpoint_path_from_job_output,
    validate_checkpoint_uri,
)


class TestCheckpointUtil(unittest.TestCase):
    @patch("boto3.client")
    def test_validate_checkpoint_uri_success(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_object.return_value = {}

        checkpoint_uri = "s3://my-bucket/path/to/checkpoint.pth"
        region = "us-east-1"

        validate_checkpoint_uri(checkpoint_uri, region)

        mock_boto_client.assert_called_once_with("s3", region_name=region)
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my-bucket", Key="path/to/checkpoint.pth"
        )

    @patch("boto3.client")
    def test_validate_checkpoint_uri_invalid_format(self, mock_boto_client):
        checkpoint_uri = "/local/path/to/checkpoint.path"
        region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            validate_checkpoint_uri(checkpoint_uri, region)

        self.assertIn("Model path must be an S3 URI", str(context.exception))
        self.assertIn(checkpoint_uri, str(context.exception))

    @patch("boto3.client")
    def test_validate_checkpoint_uri_bucket_not_found(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        error_response = {
            "Error": {
                "Code": "NoSuchBucket",
                "Message": "The specified bucket does not exist",
            }
        }
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        checkpoint_uri = "s3://nonexistent-bucket/path/to/checkpoint.pth"
        region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            validate_checkpoint_uri(checkpoint_uri, region)

        self.assertIn(
            "S3 bucket nonexistent-bucket does not exist", str(context.exception)
        )
        self.assertIn(checkpoint_uri, str(context.exception))

    @patch("boto3.client")
    def test_validate_checkpoint_uri_key_not_found(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        error_response = {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist",
            }
        }
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        checkpoint_uri = "s3://my-bucket/nonexistent/checkpoint.pth"
        region = "us-east-1"

        with self.assertRaises(ValueError) as context:
            validate_checkpoint_uri(checkpoint_uri, region)

        self.assertIn("Model checkpoint does not exist at", str(context.exception))
        self.assertIn(checkpoint_uri, str(context.exception))

    @patch("boto3.client")
    def test_validate_checkpoint_uri_nested_path(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_object.return_value = {}

        checkpoint_uri = "s3://my-bucket/deep/nested/path/to/model/checkpoint.pth"
        region = "us-east-1"

        validate_checkpoint_uri(checkpoint_uri, region)

        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my-bucket", Key="deep/nested/path/to/model/checkpoint.pth"
        )

    @patch("boto3.client")
    def test_validate_checkpoint_uri_bucket_only(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_object.return_value = {}

        checkpoint_uri = "s3://my-bucket/checkpoint.pth"
        region = "us-east-1"

        validate_checkpoint_uri(checkpoint_uri, region)

        mock_s3_client.head_object.assert_called_once_with(
            Bucket="my-bucket", Key="checkpoint.pth"
        )

    @patch("boto3.client")
    def test_validate_checkpoint_uri_access_denied(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
        mock_s3_client.head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )

        checkpoint_uri = "s3://my-bucket/checkpoint.pth"
        region = "us-east-1"

        try:
            validate_checkpoint_uri(checkpoint_uri, region)
        except Exception:
            self.fail("Function should not raise exception")

    @patch("boto3.client")
    def test_validate_checkpoint_uri_generic_exception(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_object.side_effect = Exception("Some random error")

        checkpoint_uri = "s3://my-bucket/checkpoint.pth"
        region = "us-east-1"

        try:
            validate_checkpoint_uri(checkpoint_uri, region)
        except Exception:
            self.fail("Function should not raise exception for generic errors")

    @patch("boto3.client")
    def test_validate_checkpoint_uri_different_regions(self, mock_boto_client):
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client
        mock_s3_client.head_object.return_value = {}

        test_regions = ["us-east-1", "eu-west-1", "us-west-2"]

        for region in test_regions:
            mock_boto_client.reset_mock()
            validate_checkpoint_uri("s3://bucket/key", region)
            mock_boto_client.assert_called_once_with("s3", region_name=region)


class TestExtractCheckpointPath(unittest.TestCase):
    @patch("boto3.client")
    def test_smhp_manifest_path(self, mock_boto_client):
        """Test that SMHP jobs look for manifest.json at {base_key}/{job_id}/manifest.json"""
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        # SMHP manifest structure
        smhp_manifest = {
            "checkpoint_s3_bucket": "s3://customer-escrow-123/job-id/outputs/checkpoints"
        }

        # Mock head_object to fail for tar.gz (SMTJ format)
        def head_object_side_effect(Bucket, Key):
            if Key.endswith("output.tar.gz"):
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {}

        mock_s3_client.head_object.side_effect = head_object_side_effect

        # Mock get_object to return SMHP manifest
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(smhp_manifest).encode())
        }

        result = extract_checkpoint_path_from_job_output(
            output_s3_path="s3://my-bucket/output", job_id="test-job-id"
        )

        # Verify manifest was fetched from correct SMHP path
        mock_s3_client.get_object.assert_called_with(
            Bucket="my-bucket", Key="output/test-job-id/manifest.json"
        )
        self.assertEqual(result, "s3://customer-escrow-123/job-id/outputs/checkpoints")

    @patch("boto3.client")
    @patch("tempfile.NamedTemporaryFile")
    @patch("tarfile.open")
    def test_smtj_manifest_path(
        self, mock_tarfile_open, mock_tempfile, mock_boto_client
    ):
        """Test that SMTJ jobs extract manifest from {base_key}/{job_id}/output/output.tar.gz"""
        mock_s3_client = MagicMock()
        mock_boto_client.return_value = mock_s3_client

        # SMTJ manifest structure
        smtj_manifest = {
            "checkpoint_s3_bucket": "s3://customer-escrow-123/job-id/checkpoints/192"
        }

        # Mock get_object to fail (SMHP format not found)
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "GetObject"
        )

        # Mock download_file for SMTJ
        mock_s3_client.download_file.return_value = None

        # Mock tempfile
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test.tar.gz"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        # Mock tarfile extraction
        mock_tar = MagicMock()
        mock_manifest_file = MagicMock()
        mock_manifest_file.read.return_value = json.dumps(smtj_manifest).encode()
        mock_tar.extractfile.return_value = mock_manifest_file
        mock_tarfile_open.return_value.__enter__.return_value = mock_tar

        result = extract_checkpoint_path_from_job_output(
            output_s3_path="s3://my-bucket/output", job_id="test-job-id"
        )

        # Verify tar.gz was downloaded from correct SMTJ path
        mock_s3_client.download_file.assert_called_once_with(
            "my-bucket", "output/test-job-id/output/output.tar.gz", "/tmp/test.tar.gz"
        )
        self.assertEqual(result, "s3://customer-escrow-123/job-id/checkpoints/192")


if __name__ == "__main__":
    unittest.main()
