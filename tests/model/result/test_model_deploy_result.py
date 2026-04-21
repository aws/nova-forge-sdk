"""Tests for ModelDeployResult core behavior."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.model.model_config import (
    ESCROW_URI_TAG_KEY,
    DeploymentResult,
    EndpointInfo,
    ModelDeployResult,
    ModelStatus,
)
from amzn_nova_forge.model.model_enums import DeployPlatform


class TestModelDeployResult(unittest.TestCase):
    def setUp(self):
        self.created_at = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
        self.result = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            model_name="test-model",
            escrow_uri="s3://escrow-bucket/checkpoint/",
            created_at=self.created_at,
        )

    def test_fields(self):
        self.assertEqual(
            self.result.model_arn,
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
        )
        self.assertEqual(self.result.model_name, "test-model")
        self.assertEqual(self.result.escrow_uri, "s3://escrow-bucket/checkpoint/")
        self.assertEqual(self.result.created_at, self.created_at)

    def test_to_dict_and_from_dict_roundtrip(self):
        d = self.result._to_dict()
        restored = ModelDeployResult._from_dict(d)
        self.assertEqual(restored.model_arn, self.result.model_arn)
        self.assertEqual(restored.model_name, self.result.model_name)
        self.assertEqual(restored.escrow_uri, self.result.escrow_uri)
        self.assertEqual(restored.created_at, self.result.created_at)

    def test_from_dict_missing_optional_fields(self):
        d = {
            "model_arn": "arn:test",
            "model_name": "test",
            "created_at": self.created_at.isoformat(),
        }
        restored = ModelDeployResult._from_dict(d)
        self.assertEqual(restored.escrow_uri, "")

    def test_dump_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.result.dump(file_path=tmpdir, file_name="test.json")
            self.assertTrue(path.exists())

            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["__class_name__"], "ModelDeployResult")

            loaded = ModelDeployResult.load(str(path))
            self.assertEqual(loaded.model_arn, self.result.model_arn)
            self.assertEqual(loaded.escrow_uri, self.result.escrow_uri)

    def test_from_arn_bedrock(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            "modelName": "my-model",
            "creationTime": "2026-04-01T12:00:00+00:00",
        }

        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            bedrock_client=mock_client,
        )

        self.assertEqual(result.model_name, "my-model")
        self.assertEqual(result.escrow_uri, "")

    def test_from_arn_datetime_creation_time(self):
        mock_client = Mock()
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            "modelName": "test",
            "creationTime": dt,
        }
        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            bedrock_client=mock_client,
        )
        self.assertEqual(result.created_at, dt)

    def test_from_arn_missing_creation_time(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            "modelName": "test",
        }
        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test",
            bedrock_client=mock_client,
        )
        self.assertIsInstance(result.created_at, datetime)

    def test_smi_model_deploy_result_empty_escrow(self):
        """SMI path creates ModelDeployResult with empty escrow_uri."""
        smi_result = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-east-1:123456789012:model/test-model",
            model_name="test-model",
            escrow_uri="",
            created_at=self.created_at,
        )
        self.assertEqual(smi_result.escrow_uri, "")

    def test_from_arn_sagemaker(self):
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "ModelName": "my-sm-model",
            "ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model",
            "CreationTime": datetime(2026, 4, 1, tzinfo=timezone.utc),
        }
        mock_client.list_tags.return_value = {
            "Tags": [{"Key": ESCROW_URI_TAG_KEY, "Value": "s3://bucket/escrow/"}]
        }

        result = ModelDeployResult.from_arn(
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model",
            sagemaker_client=mock_client,
        )

        self.assertEqual(result.model_name, "my-sm-model")
        self.assertEqual(result.escrow_uri, "s3://bucket/escrow/")
        self.assertEqual(
            result.model_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model",
        )
        mock_client.describe_model.assert_called_once_with(ModelName="my-sm-model")
        mock_client.list_tags.assert_called_once_with(
            ResourceArn="arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model"
        )

    def test_from_arn_sagemaker_no_tags(self):
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "ModelName": "my-sm-model",
            "ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model",
            "CreationTime": datetime(2026, 4, 1, tzinfo=timezone.utc),
        }
        mock_client.list_tags.return_value = {"Tags": []}

        result = ModelDeployResult.from_arn(
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-sm-model",
            sagemaker_client=mock_client,
        )

        self.assertEqual(result.escrow_uri, "")

    def test_from_arn_unknown_format(self):
        with self.assertRaises(ValueError) as ctx:
            ModelDeployResult.from_arn("arn:aws:lambda:us-east-1:123:function/foo")
        self.assertIn("Unrecognized ARN format", str(ctx.exception))

    def test_from_arn_bedrock_with_escrow_tag(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            "modelName": "my-model",
            "creationTime": "2026-04-01T12:00:00+00:00",
        }
        mock_client.list_tags_for_resource.return_value = {
            "tags": [{"key": ESCROW_URI_TAG_KEY, "value": "s3://bucket/escrow/"}]
        }

        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            bedrock_client=mock_client,
        )

        self.assertEqual(result.escrow_uri, "s3://bucket/escrow/")


class TestDeploymentResultModelPublish(unittest.TestCase):
    def test_model_publish_defaults_to_none(self):
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="test",
            uri="arn:test",
            model_artifact_path="s3://test",
        )
        result = DeploymentResult(
            endpoint=endpoint,
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(result.model_publish)
        self.assertIsNone(result.escrow_uri)

    def test_escrow_uri_delegates_to_model_publish(self):
        publish = ModelDeployResult(
            model_arn="arn:test",
            model_name="test",
            escrow_uri="s3://bucket/escrow/",
            created_at=datetime.now(timezone.utc),
        )
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="test",
            uri="arn:test",
            model_artifact_path="s3://test",
        )
        result = DeploymentResult(
            endpoint=endpoint,
            created_at=datetime.now(timezone.utc),
            model_publish=publish,
        )
        self.assertEqual(result.escrow_uri, "s3://bucket/escrow/")

    def test_smi_deployment_has_model_publish(self):
        """SMI deployments should also have model_publish set."""
        publish = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-east-1:123456789012:model/test",
            model_name="test-model",
            escrow_uri="s3://escrow-bucket/checkpoint/",
            created_at=datetime.now(timezone.utc),
        )
        endpoint = EndpointInfo(
            platform=DeployPlatform.SAGEMAKER,
            endpoint_name="test-smi",
            uri="arn:aws:sagemaker:us-east-1:123:endpoint/test-smi",
            model_artifact_path="s3://escrow-bucket/checkpoint/",
        )
        result = DeploymentResult(
            endpoint=endpoint,
            created_at=datetime.now(timezone.utc),
            model_publish=publish,
        )
        self.assertIsNotNone(result.model_publish)
        self.assertEqual(
            result.model_publish.model_arn,
            "arn:aws:sagemaker:us-east-1:123456789012:model/test",
        )
        self.assertEqual(result.escrow_uri, "s3://escrow-bucket/checkpoint/")

    # --- _platform / platform tests ---

    def test_platform_bedrock_arn(self):
        mdr = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/imported/abc123",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(mdr.platform, "bedrock")

    def test_platform_sagemaker_arn(self):
        mdr = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-west-2:123456789012:model/my-model",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(mdr.platform, "sagemaker")

    def test_platform_none_for_unknown_arn(self):
        mdr = ModelDeployResult(
            model_arn="arn:aws:s3:::my-bucket",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(mdr.platform)

    def test_platform_none_for_empty_arn(self):
        mdr = ModelDeployResult(
            model_arn="",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(mdr.platform)

    def test_platform_ignores_bedrock_in_resource_name(self):
        """ARN with bedrock in the wrong position should not match."""
        mdr = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-east-1:123456789012:model/my-bedrock-model",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(mdr.platform, "sagemaker")

    def test_platform_rejects_sagemaker_in_resource_name(self):
        """Bedrock ARN with 'sagemaker' in the model name should still be bedrock."""
        mdr = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/sagemaker-test",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(mdr.platform, "bedrock")

    def test_platform_static_method(self):
        self.assertEqual(
            ModelDeployResult._platform("arn:aws:bedrock:eu-west-2:123456789012:custom-model/x"),
            "bedrock",
        )
        self.assertEqual(
            ModelDeployResult._platform("arn:aws:sagemaker:ap-southeast-1:123456789012:model/y"),
            "sagemaker",
        )
        self.assertIsNone(ModelDeployResult._platform("not-an-arn"))

    def test_platform_wrong_account_format(self):
        """Account ID must be 12 digits."""
        mdr = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:short:custom-model/x",
            model_name="test",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(mdr.platform)

    def test_platform_works_after_dump_load(self):
        """Platform detection works on deserialized ModelDeployResult."""
        original = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/imported/abc",
            model_name="test-model",
            escrow_uri="s3://bucket/path/",
            created_at=datetime.now(timezone.utc),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = original.dump(file_path=tmpdir, file_name="test.json")
            loaded = ModelDeployResult.load(str(path))
        self.assertEqual(loaded.platform, "bedrock")
        self.assertEqual(loaded.model_arn, original.model_arn)

    def test_platform_works_after_dump_load_sagemaker(self):
        original = ModelDeployResult(
            model_arn="arn:aws:sagemaker:us-west-2:123456789012:model/my-sm-model",
            model_name="my-sm-model",
            escrow_uri="s3://bucket/path/",
            created_at=datetime.now(timezone.utc),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = original.dump(file_path=tmpdir, file_name="test.json")
            loaded = ModelDeployResult.load(str(path))
        self.assertEqual(loaded.platform, "sagemaker")


class TestModelDeployResultStatusTracking(unittest.TestCase):
    """Tests for model status tracking: _platform(), _region(), status, cross-platform guards."""

    BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model"
    SAGEMAKER_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model/test-model"

    def test_platform_bedrock_arn(self):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(r.platform, "bedrock")

    def test_platform_sagemaker_arn(self):
        r = ModelDeployResult(
            model_arn=self.SAGEMAKER_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(r.platform, "sagemaker")

    def test_platform_invalid_arn(self):
        r = ModelDeployResult(
            model_arn="arn:aws:lambda:us-east-1:123456789012:function/foo",
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(r.platform)

    def test_platform_rejects_substring_trick(self):
        """ARN with sagemaker in resource but wrong service field is rejected."""
        r = ModelDeployResult(
            model_arn="arn:aws:lambda:us-east-1:123456789012:sagemaker-thing/foo",
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertIsNone(r.platform)

    def test_region_extraction(self):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(r._region(), "us-east-1")

    def test_status_bedrock_active(self):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        r._bedrock_client = Mock()
        r._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        self.assertEqual(r.status, ModelStatus.ACTIVE)

    def test_status_bedrock_creating(self):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        r._bedrock_client = Mock()
        r._bedrock_client.get_custom_model.return_value = {"modelStatus": "Creating"}
        self.assertEqual(r.status, ModelStatus.CREATING)

    def test_status_sagemaker_exists(self):
        r = ModelDeployResult(
            model_arn=self.SAGEMAKER_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        r._sagemaker_client = Mock()
        r._sagemaker_client.describe_model.return_value = {}
        self.assertEqual(r.status, ModelStatus.ACTIVE)

    def test_status_sagemaker_deleted(self):
        r = ModelDeployResult(
            model_arn=self.SAGEMAKER_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        r._sagemaker_client = Mock()
        r._sagemaker_client.describe_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "not found"}},
            "DescribeModel",
        )
        self.assertEqual(r.status, ModelStatus.FAILED)

    def test_status_no_client_warns(self):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(r.status, ModelStatus.UNKNOWN)

    @patch("amzn_nova_forge.model.model_config.boto3.client")
    def test_load_refreshes_clients(self, mock_boto_client):
        r = ModelDeployResult(
            model_arn=self.BEDROCK_ARN,
            model_name="t",
            escrow_uri="",
            created_at=datetime.now(timezone.utc),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = r.dump(file_path=tmpdir, file_name="test.json")
            loaded = ModelDeployResult.load(str(path))
            mock_boto_client.assert_called_with("bedrock", region_name="us-east-1")
            self.assertIsNotNone(loaded._bedrock_client)

    def test_from_arn_stores_bedrock_client(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": self.BEDROCK_ARN,
            "modelName": "test-model",
            "creationTime": datetime.now(timezone.utc),
        }
        result = ModelDeployResult.from_arn(self.BEDROCK_ARN, bedrock_client=mock_client)
        self.assertIs(result._bedrock_client, mock_client)

    def test_from_arn_stores_sagemaker_client(self):
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "ModelName": "test-model",
            "CreationTime": datetime.now(timezone.utc),
        }
        mock_client.list_tags.return_value = {"Tags": []}
        result = ModelDeployResult.from_arn(self.SAGEMAKER_ARN, sagemaker_client=mock_client)
        self.assertIs(result._sagemaker_client, mock_client)


if __name__ == "__main__":
    unittest.main()
