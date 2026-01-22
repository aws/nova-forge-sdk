"""
Tests for MLflow utility functions.

This module tests the MLflow auto-discovery and configuration functionality,
including edge cases for AWS environments and error handling.
"""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_customization_sdk.util.mlflow import (
    get_default_mlflow_tracking_uri,
    validate_mlflow_arn_exists,
)


class TestMLflowUtilities(unittest.TestCase):
    """Test suite for MLflow utility functions."""

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_success(self, mock_client_func):
        """Test successful auto-discovery of DefaultMLFlowApp."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "DefaultMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123",
                    "Status": "Created",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertEqual(
            result, "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        mock_client_func.assert_called_once_with("sagemaker", region_name="us-west-2")
        mock_client.list_mlflow_apps.assert_called_once()

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_no_region(self, mock_client_func):
        """Test auto-discovery without specifying region."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "DefaultMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-XYZ789",
                    "Status": "Created",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri()

        # Verify
        self.assertEqual(
            result, "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-XYZ789"
        )
        mock_client_func.assert_called_once_with("sagemaker")

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_updating_status(self, mock_client_func):
        """Test auto-discovery with app in Updating status."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "DefaultMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-UPDATE1",
                    "Status": "Updating",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify - Updating status should still return the ARN
        self.assertEqual(
            result, "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-UPDATE1"
        )

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_update_failed_status(
        self, mock_client_func
    ):
        """Test auto-discovery with app in UpdateFailed status."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "DefaultMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-FAILED1",
                    "Status": "UpdateFailed",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify - UpdateFailed status should still return the ARN (might be functional)
        self.assertEqual(
            result, "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-FAILED1"
        )

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_invalid_status(self, mock_client_func):
        """Test auto-discovery with app in unusable status."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "DefaultMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-DEL123",
                    "Status": "Deleting",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify - Deleting status should still return the ARN (accept any status)
        self.assertEqual(
            result, "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-DEL123"
        )

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_not_found(self, mock_client_func):
        """Test when DefaultMLFlowApp doesn't exist."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "CustomMLFlowApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-CUSTOM",
                    "Status": "Created",
                }
            ]
        }

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_empty_list(self, mock_client_func):
        """Test when no MLFlow apps exist."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {"Summaries": []}

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_access_denied(self, mock_client_func):
        """Test handling of access denied error."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client

        # Simulate ClientError
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "AccessDeniedException"}}
        mock_client.list_mlflow_apps.side_effect = ClientError(
            error_response, "list_mlflow_apps"
        )

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_client_error(self, mock_client_func):
        """Test handling of other client errors."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client

        # Simulate ClientError
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        mock_client.list_mlflow_apps.side_effect = ClientError(
            error_response, "list_mlflow_apps"
        )

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_no_credentials(self, mock_client_func):
        """Test handling when AWS credentials are not configured."""
        # Simulate NoCredentialsError
        from botocore.exceptions import NoCredentialsError

        mock_client_func.side_effect = NoCredentialsError()

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_no_region_error(self, mock_client_func):
        """Test handling when AWS region is not configured."""
        # Simulate NoRegionError
        from botocore.exceptions import NoRegionError

        mock_client_func.side_effect = NoRegionError()

        # Test
        result = get_default_mlflow_tracking_uri()

        # Verify
        self.assertIsNone(result)

    @patch("boto3.client")
    def test_get_default_mlflow_tracking_uri_unexpected_error(self, mock_client_func):
        """Test handling of unexpected errors."""
        # Setup mock to raise unexpected exception
        mock_client_func.side_effect = RuntimeError("Unexpected error")

        # Test
        result = get_default_mlflow_tracking_uri("us-west-2")

        # Verify
        self.assertIsNone(result)


class TestMLflowValidation(unittest.TestCase):
    """Test suite for MLflow validation in base_validator."""

    def test_validate_mlflow_tracking_uri_valid_server(self):
        """Test validation of valid MLflow tracking server ARN."""
        from amzn_nova_customization_sdk.util.mlflow import (
            validate_mlflow_tracking_uri_format,
        )

        uri = (
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
        )
        result = validate_mlflow_tracking_uri_format(uri)
        self.assertTrue(result)

    def test_validate_mlflow_tracking_uri_valid_app(self):
        """Test validation of valid MLflow app ARN."""
        from amzn_nova_customization_sdk.util.mlflow import (
            validate_mlflow_tracking_uri_format,
        )

        uri = "arn:aws:sagemaker:us-east-1:987654321098:mlflow-app/app-XYZ123"
        result = validate_mlflow_tracking_uri_format(uri)
        self.assertTrue(result)

    def test_validate_mlflow_tracking_uri_empty_string(self):
        """Test validation allows empty string."""
        from amzn_nova_customization_sdk.util.mlflow import (
            validate_mlflow_tracking_uri_format,
        )

        result = validate_mlflow_tracking_uri_format("")
        self.assertTrue(result)

    def test_validate_mlflow_tracking_uri_invalid_format(self):
        """Test validation rejects invalid URI format."""
        from amzn_nova_customization_sdk.util.mlflow import (
            validate_mlflow_tracking_uri_format,
        )

        invalid_uris = [
            "not-an-arn",
            "arn:aws:s3:::my-bucket",  # Wrong service
            "arn:aws:sagemaker:us-west-2:123456789012:wrong-resource/name",
            "http://localhost:5000",  # HTTP URL not allowed
            "arn:aws:sagemaker:us-west-2:abc:mlflow-app/app-123",  # Invalid account ID
        ]

        for uri in invalid_uris:
            with self.subTest(uri=uri):
                result = validate_mlflow_tracking_uri_format(uri)
                self.assertFalse(result)

    def test_validate_mlflow_overrides_all_valid(self):
        """Test MLflow override validation with all valid values."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123",
            "mlflow_experiment_name": "my_experiment",
            "mlflow_run_name": "run_123",
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(errors, [])

    def test_validate_mlflow_overrides_all_valid_tracking_server(self):
        """Test MLflow override validation with all valid values."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/tracking-server-123",
            "mlflow_experiment_name": "my_experiment",
            "mlflow_run_name": "run_123",
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(errors, [])

    def test_validate_mlflow_overrides_invalid_uri(self):
        """Test MLflow override validation with invalid URI."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": "invalid-uri",
            "mlflow_experiment_name": "my_experiment",
            "mlflow_run_name": "run_123",
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid MLflow tracking URI format", errors[0])

    def test_validate_mlflow_overrides_empty_experiment_name(self):
        """Test MLflow override validation with empty experiment name."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123",
            "mlflow_experiment_name": "",
            "mlflow_run_name": "run_123",
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(len(errors), 1)
        self.assertIn("MLflow experiment_name cannot be an empty string", errors[0])

    def test_validate_mlflow_overrides_empty_run_name(self):
        """Test MLflow override validation with empty run name."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123",
            "mlflow_experiment_name": "my_experiment",
            "mlflow_run_name": "",
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(len(errors), 1)
        self.assertIn("MLflow run_name cannot be an empty string", errors[0])

    def test_validate_mlflow_overrides_no_tracking_uri_with_names(self):
        """Test warning when experiment/run names provided without tracking URI."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_experiment_name": "my_experiment",
            "mlflow_run_name": "run_123",
        }

        # This should log a warning but not return an error
        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(errors, [])

    def test_validate_mlflow_overrides_none_values(self):
        """Test MLflow override validation with None values (Optional fields)."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        overrides = {
            "mlflow_tracking_uri": None,
            "mlflow_experiment_name": None,
            "mlflow_run_name": None,
        }

        errors = validate_mlflow_overrides(overrides, check_exists=False)
        self.assertEqual(errors, [])

    def test_validate_mlflow_overrides_empty_dict(self):
        """Test MLflow override validation with empty overrides."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        errors = validate_mlflow_overrides({}, check_exists=False)
        self.assertEqual(errors, [])

    def test_validate_mlflow_overrides_none_input(self):
        """Test MLflow override validation with None input."""
        from amzn_nova_customization_sdk.util.mlflow import validate_mlflow_overrides

        errors = validate_mlflow_overrides(None, check_exists=False)
        self.assertEqual(errors, [])


class TestMLflowArnValidation(unittest.TestCase):
    """Test suite for MLflow ARN existence validation."""

    def test_validate_mlflow_arn_exists_empty_uri(self):
        """Test validation with empty URI."""
        is_valid, message = validate_mlflow_arn_exists("")
        self.assertFalse(is_valid)
        self.assertEqual(message, "MLflow tracking URI is empty")

    def test_validate_mlflow_arn_exists_non_arn_format(self):
        """Test validation with non-ARN format URI."""
        is_valid, message = validate_mlflow_arn_exists("http://localhost:5000")
        self.assertTrue(is_valid)
        self.assertEqual(message, "Non-ARN format URI - existence check skipped")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_app_found(self, mock_client_func):
        """Test validation when MLflow app exists."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {
            "Summaries": [
                {
                    "Name": "TestApp",
                    "Arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123",
                    "Status": "Created",
                }
            ]
        }

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertTrue(is_valid)
        self.assertEqual(message, "MLflow app exists (status: Created)")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_app_not_found(self, mock_client_func):
        """Test validation when MLflow app doesn't exist."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_apps.return_value = {"Summaries": []}

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertFalse(is_valid)
        self.assertEqual(message, "MLflow app not found: app-ABC123")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_tracking_server_found(self, mock_client_func):
        """Test validation when MLflow tracking server exists."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_tracking_servers.return_value = {
            "TrackingServerSummaries": [
                {
                    "TrackingServerName": "my-server",
                    "TrackingServerArn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server",
                    "IsActive": True,
                }
            ]
        }

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
        )
        self.assertTrue(is_valid)
        self.assertEqual(message, "MLflow tracking server exists (status: True)")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_tracking_server_not_found(
        self, mock_client_func
    ):
        """Test validation when MLflow tracking server doesn't exist."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client
        mock_client.list_mlflow_tracking_servers.return_value = {
            "TrackingServerSummaries": []
        }

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
        )
        self.assertFalse(is_valid)
        self.assertEqual(message, "MLflow tracking server not found: my-server")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_access_denied(self, mock_client_func):
        """Test validation with access denied error."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client

        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "AccessDeniedException"}}
        mock_client.list_mlflow_apps.side_effect = ClientError(
            error_response, "list_mlflow_apps"
        )

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertTrue(is_valid)
        self.assertEqual(message, "Access denied - assuming ARN is valid")

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_client_error(self, mock_client_func):
        """Test validation with other client error."""
        mock_client = MagicMock()
        mock_client_func.return_value = mock_client

        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        mock_client.list_mlflow_apps.side_effect = ClientError(
            error_response, "list_mlflow_apps"
        )

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertFalse(is_valid)
        self.assertIn("Error checking MLflow resource", message)

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_no_credentials(self, mock_client_func):
        """Test validation with no AWS credentials."""
        from botocore.exceptions import NoCredentialsError

        mock_client_func.side_effect = NoCredentialsError()

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertTrue(is_valid)
        self.assertEqual(
            message, "AWS credentials not configured - cannot validate MLflow ARN"
        )

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_no_region(self, mock_client_func):
        """Test validation with no AWS region."""
        from botocore.exceptions import NoRegionError

        mock_client_func.side_effect = NoRegionError()

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertTrue(is_valid)
        self.assertEqual(
            message, "AWS region not configured - cannot validate MLflow ARN"
        )

    @patch("boto3.client")
    def test_validate_mlflow_arn_exists_unexpected_error(self, mock_client_func):
        """Test validation with unexpected error."""
        mock_client_func.side_effect = RuntimeError("Unexpected error")

        is_valid, message = validate_mlflow_arn_exists(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABC123"
        )
        self.assertTrue(is_valid)
        self.assertIn("Unexpected error - assuming ARN is valid", message)


if __name__ == "__main__":
    unittest.main()
