"""Tests for ModelDeployResult and deploy-decoupling feature."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from amzn_nova_forge.model.model_config import (
    DeploymentResult,
    EndpointInfo,
    ModelDeployResult,
)
from amzn_nova_forge.model.model_enums import DeployPlatform


class TestModelDeployResult(unittest.TestCase):
    """Tests for ModelDeployResult dataclass."""

    def setUp(self):
        self.created_at = datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc)
        self.result = ModelDeployResult(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model",
            model_name="test-model-abc123",
            escrow_uri="s3://bucket/escrow/checkpoint/",
            created_at=self.created_at,
        )

    def test_construction(self):
        self.assertEqual(
            self.result.model_arn,
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/test-model",
        )
        self.assertEqual(self.result.model_name, "test-model-abc123")
        self.assertEqual(self.result.escrow_uri, "s3://bucket/escrow/checkpoint/")
        self.assertEqual(self.result.created_at, self.created_at)

    def test_to_dict(self):
        d = self.result._to_dict()
        self.assertEqual(d["model_arn"], self.result.model_arn)
        self.assertEqual(d["model_name"], self.result.model_name)
        self.assertEqual(d["escrow_uri"], self.result.escrow_uri)
        self.assertEqual(d["created_at"], self.created_at.isoformat())

    def test_from_dict(self):
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

            # Verify JSON content
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["__class_name__"], "ModelDeployResult")
            self.assertEqual(data["model_arn"], self.result.model_arn)

            # Load back
            loaded = ModelDeployResult.load(str(path))
            self.assertEqual(loaded.model_arn, self.result.model_arn)
            self.assertEqual(loaded.model_name, self.result.model_name)
            self.assertEqual(loaded.escrow_uri, self.result.escrow_uri)

    def test_dump_default_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.result.dump(file_path=tmpdir)
            self.assertIn(self.result.model_name, str(path))

    def test_from_arn(self):
        mock_client = Mock()
        mock_client.get_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            "modelName": "my-model",
            "creationTime": "2026-03-31T12:00:00+00:00",
        }

        result = ModelDeployResult.from_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model",
            bedrock_client=mock_client,
        )

        mock_client.get_custom_model.assert_called_once_with(
            modelIdentifier="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        )
        self.assertEqual(result.model_name, "my-model")
        self.assertEqual(result.escrow_uri, "")  # Not available from API

    def test_from_arn_datetime_creation_time(self):
        """GetCustomModel may return creationTime as datetime object."""
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
        # Should use datetime.now(utc) as fallback
        self.assertIsInstance(result.created_at, datetime)


class TestDeploymentResultModelPublish(unittest.TestCase):
    """Tests for DeploymentResult.model_publish and escrow_uri property."""

    def test_model_publish_defaults_to_none(self):
        """Backward compat: existing construction still works."""
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


if __name__ == "__main__":
    unittest.main()
