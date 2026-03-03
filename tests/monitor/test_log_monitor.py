import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from amzn_nova_customization_sdk.model.model_enums import Platform
from amzn_nova_customization_sdk.model.result.job_result import (
    BaseJobResult,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_customization_sdk.monitor.log_monitor import CloudWatchLogMonitor


class TestCloudWatchLogMonitor(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock()
        self.job_id = "test-job-123"
        self.platform = Platform.SMTJ
        self.started_time = 1234567890

    def test_init_smtj_platform(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        self.assertEqual(monitor.job_id, self.job_id)
        self.assertEqual(monitor.platform, self.platform)
        self.assertEqual(monitor.log_group_name, "/aws/sagemaker/TrainingJobs")
        self.assertEqual(monitor.log_stream_name, "test-job-123/algo-1-1234567890")

    def test_find_log_stream_no_streams(self):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )

        self.assertIsNone(monitor.log_stream_name)

    def test_get_logs_no_stream(self):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(result, [])

    def test_get_logs_basic(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {
                "events": [{"message": "log1"}, {"message": "log2"}],
                "nextBackwardToken": "token1",
            },
            {"events": [], "nextBackwardToken": "token1"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["message"], "log1")
        self.assertEqual(self.mock_client.get_log_events.call_count, 2)

    def test_get_logs_with_limit(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.return_value = {
            "events": [{"message": "log1"}, {"message": "log2"}, {"message": "log3"}],
            "nextBackwardToken": "token1",
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs(limit=2)

        self.assertEqual(len(result), 2)

    def test_get_logs_pagination(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {"events": [{"message": "log1"}], "nextBackwardToken": "token2"},
            {"events": [{"message": "log2"}], "nextBackwardToken": "token2"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        result = monitor.get_logs()

        self.assertEqual(len(result), 2)
        self.assertEqual(self.mock_client.get_log_events.call_count, 2)

    @patch("builtins.print")
    def test_show_logs_with_events(self, mock_print):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.side_effect = [
            {
                "events": [{"message": "log1\n"}, {"message": "log2\n"}],
                "nextBackwardToken": "token1",
            },
            {"events": [], "nextBackwardToken": "token1"},
        ]

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.show_logs()

        mock_print.assert_any_call("log1")
        mock_print.assert_any_call("log2")
        self.assertEqual(mock_print.call_count, 2)

    @patch("builtins.print")
    def test_show_logs_no_events(self, mock_print):
        self.mock_client.describe_log_streams.return_value = {"logStreams": []}

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        monitor.show_logs()

        mock_print.assert_called_once_with(f"No logs found for job {self.job_id} yet")

    def test_get_logs_with_start_and_end_time(self):
        self.mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "test-job-123/algo-1-1234567890"}]
        }
        self.mock_client.get_log_events.return_value = {
            "events": [{"message": "log1"}],
            "nextBackwardToken": "token1",
        }

        monitor = CloudWatchLogMonitor(
            job_id=self.job_id,
            platform=self.platform,
            started_time=self.started_time,
            cloudwatch_logs_client=self.mock_client,
        )
        end_time = 1234567999
        monitor.get_logs(end_time=end_time)

        self.mock_client.get_log_events.assert_called_with(
            endTime=end_time,
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamName="test-job-123/algo-1-1234567890",
            startFromHead=False,
            startTime=self.started_time,
            nextToken="token1",
        )

    def test_from_job_id_smtj(self):
        job_id = "test-job-456"
        platform = Platform.SMTJ
        started_time = datetime(2023, 1, 1, 12, 0, 0)

        with patch.object(
            CloudWatchLogMonitor, "__init__", return_value=None
        ) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_id(
                job_id=job_id, platform=platform, started_time=started_time
            )

            mock_init.assert_called_once_with(
                job_id=job_id,
                platform=platform,
                started_time=int(started_time.timestamp() * 1000),
            )

    def test_from_job_id_smhp(self):
        job_id = "test-job-789"
        platform = Platform.SMHP
        cluster_name = "test-cluster"
        namespace = "test-namespace"

        with patch.object(
            CloudWatchLogMonitor, "__init__", return_value=None
        ) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_id(
                job_id=job_id,
                platform=platform,
                cluster_name=cluster_name,
                namespace=namespace,
            )

            mock_init.assert_called_once_with(
                job_id=job_id,
                platform=platform,
                started_time=None,
                cluster_name=cluster_name,
                namespace=namespace,
            )

    def test_from_job_result_smtj(self):
        mock_job_result = Mock(spec=BaseJobResult)
        mock_job_result.job_id = "smtj-job-123"
        mock_job_result.platform = Platform.SMTJ
        mock_job_result.started_time = datetime(2023, 1, 1, 12, 0, 0)

        mock_client = Mock()

        with patch.object(
            CloudWatchLogMonitor, "__init__", return_value=None
        ) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_result(
                job_result=mock_job_result, cloudwatch_logs_client=mock_client
            )

            mock_init.assert_called_once_with(
                job_id="smtj-job-123",
                platform=Platform.SMTJ,
                started_time=int(mock_job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=mock_client,
            )

    def test_from_job_result_smhp(self):
        mock_job_result = Mock(spec=BaseJobResult)
        mock_job_result.job_id = "smhp-job-456"
        mock_job_result.platform = Platform.SMHP
        mock_job_result.started_time = datetime(2023, 1, 1, 12, 0, 0)

        mock_status_manager = Mock(spec=SMHPStatusManager)
        mock_status_manager.cluster_name = "test-cluster"
        mock_status_manager.namespace = "test-namespace"
        mock_job_result.status_manager = mock_status_manager

        mock_client = Mock()

        with patch.object(
            CloudWatchLogMonitor, "__init__", return_value=None
        ) as mock_init:
            monitor = CloudWatchLogMonitor.from_job_result(
                job_result=mock_job_result, cloudwatch_logs_client=mock_client
            )

            mock_init.assert_called_once_with(
                job_id="smhp-job-456",
                platform=Platform.SMHP,
                started_time=int(mock_job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=mock_client,
                cluster_name="test-cluster",
                namespace="test-namespace",
            )


if __name__ == "__main__":
    unittest.main()
