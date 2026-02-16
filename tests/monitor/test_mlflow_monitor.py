"""
Tests for MLflowMonitor class.

This module tests the MLflowMonitor configuration functionality,
including default discovery and explicit configuration.
"""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_customization_sdk.monitor.mlflow_monitor import MLflowMonitor


class TestMLflowMonitor(unittest.TestCase):
    """Test suite for MLflowMonitor configuration class."""

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_mlflow_monitor_explicit_tracking_uri(self, mock_validate):
        """Test MLflowMonitor with explicit tracking URI."""
        # Setup mock
        mock_validate.return_value = []  # No validation errors

        # Create monitor with explicit URI
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-CUSTOM"
        )

        # Verify configuration
        self.assertEqual(
            monitor.tracking_uri,
            "arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-CUSTOM",
        )
        self.assertIsNone(monitor.experiment_name)
        self.assertIsNone(monitor.run_name)

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_mlflow_monitor_full_configuration(self, mock_validate):
        """Test MLflowMonitor with all parameters specified."""
        # Setup mock
        mock_validate.return_value = []  # No validation errors

        # Create monitor with full configuration
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server",
            experiment_name="my_experiment",
            run_name="run_2024_01_15",
        )

        # Verify all fields are set
        self.assertEqual(
            monitor.tracking_uri,
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server",
        )
        self.assertEqual(monitor.experiment_name, "my_experiment")
        self.assertEqual(monitor.run_name, "run_2024_01_15")

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_mlflow_monitor_partial_configuration(self, mock_validate):
        """Test MLflowMonitor with partial configuration."""
        # Setup mock
        mock_validate.return_value = []  # No validation errors

        # Create monitor with only tracking URI and experiment name
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-XYZ",
            experiment_name="test_experiment",
        )

        # Verify configuration
        self.assertEqual(
            monitor.tracking_uri,
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-XYZ",
        )
        self.assertEqual(monitor.experiment_name, "test_experiment")
        self.assertIsNone(monitor.run_name)

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_mlflow_monitor_to_dict_full(self, mock_validate):
        """Test MLflowMonitor to_dict method with full configuration."""
        # Setup mock
        mock_validate.return_value = []  # No validation errors

        # Create monitor with full configuration
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABC",
            experiment_name="experiment_1",
            run_name="run_1",
        )

        # Convert to dict
        result = monitor.to_dict()

        # Verify dict structure
        expected = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABC",
            "mlflow_experiment_name": "experiment_1",
            "mlflow_run_name": "run_1",
        }
        self.assertEqual(result, expected)

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_mlflow_monitor_to_dict_partial(self, mock_validate):
        """Test MLflowMonitor to_dict method with partial configuration."""
        # Setup mock
        mock_validate.return_value = []  # No validation errors

        # Create monitor with only tracking URI
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-DEF"
        )

        # Convert to dict
        result = monitor.to_dict()

        # Verify dict only contains non-None values
        expected = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-DEF"
        }
        self.assertEqual(result, expected)

    def test_mlflow_monitor_empty_strings(self):
        """Test MLflowMonitor with empty strings (should be treated as None)."""
        # Create monitor with empty strings
        monitor = MLflowMonitor(tracking_uri="", experiment_name="", run_name="")

        # Verify empty strings are treated as None
        self.assertEqual(monitor.tracking_uri, "")
        self.assertEqual(monitor.experiment_name, "")
        self.assertEqual(monitor.run_name, "")

        # Verify to_dict skips empty strings
        result = monitor.to_dict()
        self.assertEqual(result, {})

    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.get_default_mlflow_tracking_uri"
    )
    def test_mlflow_monitor_override_default_discovery(
        self, mock_get_default, mock_validate
    ):
        """Test that explicit tracking URI overrides default discovery."""
        # Setup mocks
        mock_get_default.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-DEFAULT"
        )
        mock_validate.return_value = []  # No validation errors

        # Create monitor with explicit URI
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-EXPLICIT"
        )

        # Verify default discovery was NOT called
        mock_get_default.assert_not_called()

        # Verify explicit URI is used
        self.assertEqual(
            monitor.tracking_uri,
            "arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-EXPLICIT",
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_get_presigned_url_success(self, mock_validate, mock_boto_client):
        """Test successful generation of presigned URL for MLflow app."""
        # Setup mocks
        mock_validate.return_value = []
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": "https://mlflow.example.com/presigned?token=abc123"
        }

        # Create monitor with mlflow-app ARN
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-TEST"
        )

        # Generate presigned URL
        url = monitor.get_presigned_url()

        # Verify URL was returned
        self.assertEqual(url, "https://mlflow.example.com/presigned?token=abc123")

        # Verify boto3 client was created with correct region
        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-east-1")

        # Verify API was called with correct parameters (full ARN for mlflow-app)
        mock_sagemaker.create_presigned_mlflow_app_url.assert_called_once_with(
            Arn="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-TEST",
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=43200,
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_get_presigned_url_custom_expiration(self, mock_validate, mock_boto_client):
        """Test presigned URL generation with custom expiration duration."""
        # Setup mocks
        mock_validate.return_value = []
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": "https://mlflow.example.com/presigned?token=xyz789"
        }

        # Create monitor with mlflow-app ARN
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-CUSTOM"
        )

        # Generate presigned URL with custom expiration (1 hour session, 60s URL)
        url = monitor.get_presigned_url(
            session_expiration_duration_in_seconds=3600, expires_in_seconds=60
        )

        # Verify URL was returned
        self.assertEqual(url, "https://mlflow.example.com/presigned?token=xyz789")

        # Verify API was called with custom expiration
        mock_sagemaker.create_presigned_mlflow_app_url.assert_called_once_with(
            Arn="arn:aws:sagemaker:us-west-2:987654321098:mlflow-app/app-CUSTOM",
            ExpiresInSeconds=60,
            SessionExpirationDurationInSeconds=3600,
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_get_presigned_url_tracking_server(self, mock_validate, mock_boto_client):
        """Test presigned URL generation for mlflow-tracking-server (legacy)."""
        # Setup mocks
        mock_validate.return_value = []
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.create_presigned_mlflow_tracking_server_url.return_value = {
            "AuthorizedUrl": "https://mlflow-server.example.com/presigned?token=legacy123"
        }

        # Create monitor with mlflow-tracking-server ARN (legacy format)
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server"
        )

        # Generate presigned URL
        url = monitor.get_presigned_url()

        # Verify URL was returned
        self.assertEqual(
            url, "https://mlflow-server.example.com/presigned?token=legacy123"
        )

        # Verify API was called with tracking server name (not full ARN)
        mock_sagemaker.create_presigned_mlflow_tracking_server_url.assert_called_once_with(
            TrackingServerName="my-server",
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=43200,
        )

    def test_get_presigned_url_no_tracking_uri(self):
        """Test that get_presigned_url raises error when tracking_uri is not set."""
        # Create monitor without tracking URI
        monitor = MLflowMonitor(tracking_uri=None)

        # Verify error is raised
        with self.assertRaises(ValueError) as context:
            monitor.get_presigned_url()

        self.assertIn("tracking_uri is not set", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_get_presigned_url_api_failure(self, mock_validate, mock_boto_client):
        """Test that get_presigned_url handles API failures gracefully."""
        # Setup mocks
        mock_validate.return_value = []
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.create_presigned_mlflow_app_url.side_effect = Exception(
            "API Error"
        )

        # Create monitor
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-FAIL"
        )

        # Verify error is raised
        with self.assertRaises(RuntimeError) as context:
            monitor.get_presigned_url()

        self.assertIn("Failed to generate presigned URL", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.monitor.mlflow_monitor.validate_mlflow_overrides"
    )
    def test_get_presigned_url_no_url_in_response(
        self, mock_validate, mock_boto_client
    ):
        """Test that get_presigned_url handles missing URL in API response."""
        # Setup mocks
        mock_validate.return_value = []
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker.create_presigned_mlflow_app_url.return_value = {}

        # Create monitor
        monitor = MLflowMonitor(
            tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-EMPTY"
        )

        # Verify error is raised
        with self.assertRaises(RuntimeError) as context:
            monitor.get_presigned_url()

        self.assertIn("No URL returned from API", str(context.exception))


if __name__ == "__main__":
    unittest.main()
