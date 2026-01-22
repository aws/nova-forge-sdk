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


if __name__ == "__main__":
    unittest.main()
