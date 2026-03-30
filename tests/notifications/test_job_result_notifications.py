import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from amzn_nova_forge.model.model_config import ModelArtifacts
from amzn_nova_forge.model.model_enums import Platform
from amzn_nova_forge.model.result.job_result import BaseJobResult
from amzn_nova_forge.notifications.notification_manager import (
    NotificationManagerInfraError,
)
from amzn_nova_forge.notifications.smtj_notification_manager import (
    SMTJNotificationManager,
)


class MockJobResult(BaseJobResult):
    """Mock JobResult for testing."""

    def __init__(
        self,
        job_id,
        started_time,
        model_artifacts=None,
    ):
        """
        Initialize mock job result.

        Args:
            job_id: Job identifier
            started_time: Job start time
            model_artifacts: ModelArtifacts object or None. If None, the job result
                will not have a model_artifacts attribute (simulates jobs that don't
                have model_artifacts created)
        """
        # Mock boto3.client to avoid NoRegionError in tests
        with patch("boto3.client"):
            super().__init__(job_id, started_time)
        self.model_artifacts = model_artifacts

    def _create_status_manager(self):
        """Create a mock status manager."""
        from amzn_nova_forge.model.result.job_result import (
            SMTJStatusManager,
        )

        # Create with a mocked sagemaker client to avoid region issues
        mock_client = MagicMock()
        return SMTJStatusManager(sagemaker_client=mock_client)

    def get(self):
        """Mock get method."""
        return {"job_id": self.job_id}

    def show(self):
        """Mock show method."""
        print(f"Job: {self.job_id}")


class TestJobResultNotifications(unittest.TestCase):
    """Test suite for job result notification integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.job_id = "test-job-123"
        self.started_time = datetime.now()

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_with_model_artifacts(self, mock_manager_class):
        """Test enable_job_notifications with model_artifacts containing output_s3_path."""
        # Create mock manager instance
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Create job result with model artifacts (like real TrainingResult)
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/path",
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]
        job_result.enable_job_notifications(emails=emails)

        # Verify SMTJNotificationManager was created with correct region
        mock_manager_class.assert_called_once_with(region="us-east-1")

        # Verify enable_notifications was called with correct parameters
        mock_manager.enable_notifications.assert_called_once_with(
            job_name=self.job_id, emails=emails, output_s3_path="s3://bucket/path"
        )

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_with_explicit_output_path(
        self, mock_manager_class
    ):
        """Test enable_job_notifications with explicitly provided output_s3_path."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Create job result without model artifacts
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=None,
        )

        emails = ["user@example.com"]
        explicit_path = "s3://explicit-bucket/explicit-path"

        job_result.enable_job_notifications(emails=emails, output_s3_path=explicit_path)

        # Verify enable_notifications was called with explicit path
        mock_manager.enable_notifications.assert_called_once_with(
            job_name=self.job_id, emails=emails, output_s3_path=explicit_path
        )

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_with_custom_region(self, mock_manager_class):
        """Test enable_job_notifications with custom region."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/path",
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]
        custom_region = "us-west-2"

        job_result.enable_job_notifications(emails=emails, region=custom_region)

        # Verify SMTJNotificationManager was created with custom region
        mock_manager_class.assert_called_once_with(region=custom_region)

    def test_enable_job_notifications_no_model_artifacts_no_path(self):
        """Test enable_job_notifications raises error when no output_s3_path available."""
        # Job result without model artifacts and no explicit path
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=None,
        )

        emails = ["user@example.com"]

        with self.assertRaises(ValueError) as context:
            job_result.enable_job_notifications(emails=emails)

        self.assertIn("output_s3_path is required", str(context.exception))
        self.assertIn("can't be found", str(context.exception))

    def test_enable_job_notifications_model_artifacts_empty_path(self):
        """Test enable_job_notifications raises error when model_artifacts has empty path."""
        # Model artifacts with empty output_s3_path
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="",  # Empty path
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]

        with self.assertRaises(ValueError) as context:
            job_result.enable_job_notifications(emails=emails)

        self.assertIn("output_s3_path is required", str(context.exception))
        self.assertIn("not set", str(context.exception))

    def test_enable_job_notifications_model_artifacts_none_path(self):
        """Test enable_job_notifications raises error when model_artifacts has None path."""
        # Model artifacts with None output_s3_path
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path=None,
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]

        with self.assertRaises(ValueError) as context:
            job_result.enable_job_notifications(emails=emails)

        self.assertIn("output_s3_path is required", str(context.exception))

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_explicit_overrides_model_artifacts(
        self, mock_manager_class
    ):
        """Test explicit output_s3_path overrides model_artifacts path."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Model artifacts with one path
        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/model-artifacts-path",
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]
        explicit_path = "s3://bucket/explicit-override-path"

        job_result.enable_job_notifications(emails=emails, output_s3_path=explicit_path)

        # Verify explicit path was used, not model_artifacts path
        mock_manager.enable_notifications.assert_called_once_with(
            job_name=self.job_id, emails=emails, output_s3_path=explicit_path
        )

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_multiple_emails(self, mock_manager_class):
        """Test enable_job_notifications with multiple email addresses."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/path",
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user1@example.com", "user2@example.com", "user3@example.com"]

        job_result.enable_job_notifications(emails=emails)

        # Verify all emails were passed
        mock_manager.enable_notifications.assert_called_once_with(
            job_name=self.job_id, emails=emails, output_s3_path="s3://bucket/path"
        )

    @patch("amzn_nova_forge.notifications.SMTJNotificationManager")
    def test_enable_job_notifications_propagates_errors(self, mock_manager_class):
        """Test enable_job_notifications propagates NotificationManager errors."""
        mock_manager = MagicMock()
        mock_manager.enable_notifications.side_effect = NotificationManagerInfraError(
            "Infrastructure setup failed"
        )
        mock_manager_class.return_value = mock_manager

        model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint/",
            output_s3_path="s3://bucket/path",
        )
        job_result = MockJobResult(
            self.job_id,
            self.started_time,
            model_artifacts=model_artifacts,
        )

        emails = ["user@example.com"]

        with self.assertRaises(NotificationManagerInfraError) as context:
            job_result.enable_job_notifications(emails=emails)

        self.assertIn("Infrastructure setup failed", str(context.exception))

    def test_enable_job_notifications_default_region(self):
        """Test enable_job_notifications uses default region when not specified."""
        with patch(
            "amzn_nova_forge.notifications.SMTJNotificationManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            model_artifacts = ModelArtifacts(
                checkpoint_s3_path="s3://bucket/checkpoint/",
                output_s3_path="s3://bucket/path",
            )
            job_result = MockJobResult(
                self.job_id,
                self.started_time,
                model_artifacts=model_artifacts,
            )

            emails = ["user@example.com"]

            # Don't specify region
            job_result.enable_job_notifications(emails=emails)

            # Verify default region was used
            mock_manager_class.assert_called_once_with(region="us-east-1")


if __name__ == "__main__":
    unittest.main()
