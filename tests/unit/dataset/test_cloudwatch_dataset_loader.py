# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for CloudWatchDatasetLoader.

All boto3 calls are mocked via ``unittest.mock.patch`` so no real network
calls are made. Telemetry is auto-mocked globally by
``tests/unit/conftest.py::_mock_telemetry``.
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge.dataset.cloudwatch_dataset_loader import (
    CloudWatchDatasetLoader,
)
from amzn_nova_forge.dataset.operations.base import DataPrepError

LOG_GROUP = "/test/app-logs"
QUERY = "fields @timestamp, @message | limit 100"
START_TIME = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2024, 6, 2, 0, 0, 0, tzinfo=timezone.utc)
QUERY_ID = "12345678-1234-1234-1234-123456789012"


def _complete_response(results):
    """Build a GetQueryResults response with status Complete."""
    return {"status": "Complete", "results": results}


def _running_response():
    """Build a GetQueryResults response with status Running."""
    return {"status": "Running", "results": []}


class TestLoadIsLazy:
    """load() stores config only; no boto3 calls happen at load time."""

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_no_boto3_calls_at_load_time(self, mock_boto3):
        loader = CloudWatchDatasetLoader()
        result = loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        mock_boto3.client.assert_not_called()
        assert result is loader
        assert loader._load_path == f"cw:///{LOG_GROUP}"


class TestStateReset:
    """load() resets _last_state, _is_materialized, and _session_id."""

    def test_state_fields_reset_on_load(self):
        loader = CloudWatchDatasetLoader()
        loader._last_state = "something"
        loader._is_materialized = True
        loader._session_id = "old-session"

        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        assert loader._last_state is None
        assert loader._is_materialized is False
        assert loader._session_id is None


class TestMakeSingleFileGenerator:
    """_make_single_file_generator raises NotImplementedError."""

    def test_raises_not_implemented(self):
        loader = CloudWatchDatasetLoader()
        with pytest.raises(NotImplementedError) as exc_info:
            loader._make_single_file_generator("anything")

        assert "not file-based" in str(exc_info.value)


class TestGeneratorHappyPath:
    """Generator executes query and yields flat dicts on iteration."""

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_yields_converted_dicts(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = _complete_response(
            [
                [
                    {"field": "@timestamp", "value": "2024-06-01 10:30:00.000"},
                    {"field": "@message", "value": '{"action": "invoke"}'},
                    {"field": "requestId", "value": "req-001"},
                ],
                [
                    {"field": "@timestamp", "value": "2024-06-01 10:31:00.000"},
                    {"field": "@message", "value": '{"action": "complete"}'},
                    {"field": "requestId", "value": "req-002"},
                ],
            ]
        )

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)
        results = list(loader.dataset())

        assert results == [
            {
                "@timestamp": "2024-06-01 10:30:00.000",
                "@message": '{"action": "invoke"}',
                "requestId": "req-001",
            },
            {
                "@timestamp": "2024-06-01 10:31:00.000",
                "@message": '{"action": "complete"}',
                "requestId": "req-002",
            },
        ]

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_start_query_receives_correct_parameters(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = _complete_response([])

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)
        list(loader.dataset())

        mock_client.start_query.assert_called_once_with(
            logGroupName=LOG_GROUP,
            queryString=QUERY,
            startTime=int(START_TIME.timestamp()),
            endTime=int(END_TIME.timestamp()),
        )

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_timestamps_converted_to_epoch_seconds(self, mock_boto3):
        """Datetimes are passed as int(dt.timestamp()) — epoch seconds, not millis."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = _complete_response([])

        est = timezone(timedelta(hours=-5))
        start = datetime(2024, 6, 15, 7, 0, 0, tzinfo=est)
        end = datetime(2024, 6, 16, 7, 0, 0, tzinfo=est)

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, start, end)
        list(loader.dataset())

        call_kwargs = mock_client.start_query.call_args.kwargs
        assert call_kwargs["startTime"] == int(start.timestamp())
        assert call_kwargs["endTime"] == int(end.timestamp())


class TestPollingBehavior:
    """Polling calls sleep between attempts and passes correct query ID."""

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.time.sleep")
    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_sleep_called_between_polls(self, mock_boto3, mock_sleep):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.side_effect = [
            _running_response(),
            _running_response(),
            _complete_response([[{"field": "@timestamp", "value": "2024-06-01"}]]),
        ]

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)
        list(loader.dataset())

        assert mock_sleep.call_count == 2

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.time.sleep")
    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_correct_query_id_passed_to_get_results(self, mock_boto3, mock_sleep):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = _complete_response([])

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)
        list(loader.dataset())

        mock_client.get_query_results.assert_called_with(queryId=QUERY_ID)


class TestEmptyResults:
    """Empty results yield nothing and log a warning."""

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_empty_results_yields_nothing_and_warns(self, mock_boto3, caplog):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = _complete_response([])

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        with caplog.at_level(logging.WARNING):
            results = list(loader.dataset())

        assert results == []
        assert any("0 results" in rec.message for rec in caplog.records)


class TestErrorMapping:
    """CloudWatch exceptions are wrapped in DataPrepError with chaining."""

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_start_query_error_wrapped_and_chained(self, mock_boto3):
        original = RuntimeError("something went wrong")
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.side_effect = original

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        assert "something went wrong" in str(exc_info.value)
        assert exc_info.value.__cause__ is original

    @pytest.mark.parametrize(
        "status",
        ["Failed", "Cancelled", "Timeout", "Unknown"],
    )
    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.time.sleep")
    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_terminal_failure_statuses(self, mock_boto3, mock_sleep, status):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.return_value = {
            "status": status,
            "results": [],
        }

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        assert status in str(exc_info.value)

    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.time.sleep")
    @patch("amzn_nova_forge.dataset.cloudwatch_dataset_loader.boto3")
    def test_poll_error_wrapped_and_chained(self, mock_boto3, mock_sleep):
        original = RuntimeError("network timeout")
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.start_query.return_value = {"queryId": QUERY_ID}
        mock_client.get_query_results.side_effect = original

        loader = CloudWatchDatasetLoader()
        loader.load(LOG_GROUP, QUERY, START_TIME, END_TIME)

        with pytest.raises(DataPrepError) as exc_info:
            list(loader.dataset())

        assert exc_info.value.__cause__ is original
