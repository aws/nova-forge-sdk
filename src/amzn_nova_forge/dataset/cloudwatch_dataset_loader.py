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
"""CloudWatch Logs Insights dataset loader."""

import time
from datetime import datetime, timezone

import boto3

from ..telemetry import Feature, _telemetry_emitter
from ..util.logging import logger
from .dataset_loader import DatasetLoader
from .operations.base import DataPrepError

_TERMINAL_FAILURE_STATUSES = {"Failed", "Cancelled", "Timeout", "Unknown"}

_POLL_INTERVAL_SECONDS = 10


class CloudWatchDatasetLoader(DatasetLoader):
    """Load datasets from CloudWatch Logs Insights queries."""

    _EXTENSIONS: set[str] = set()
    _FORMAT: str = "cloudwatch"

    def _make_single_file_generator(self, path: str):
        raise NotImplementedError(
            "CloudWatch datasets are not file-based. "
            "Use load(log_group=..., query=..., start_time=..., end_time=...) instead."
        )

    @_telemetry_emitter(Feature.DATA_PREP, "load")
    def load(
        self,
        log_group: str,
        query: str,
        start_time: datetime,
        end_time: datetime,
    ) -> "CloudWatchDatasetLoader":
        """Load data from a CloudWatch Logs Insights query.

        Args:
            log_group: CloudWatch log group name.
            query: Insights query string.
            start_time: Query start time (inclusive).
            end_time: Query end time (exclusive).

        Returns:
            self (for method chaining)

        Raises:
            DataPrepError: If the query fails during iteration.
        """
        cw_log_group = log_group
        cw_query = query
        cw_start_time = start_time
        cw_end_time = end_time

        def _generator():
            start_utc = cw_start_time.astimezone(timezone.utc)
            end_utc = cw_end_time.astimezone(timezone.utc)
            logger.info(
                "Querying CloudWatch log group '%s' from %s to %s (UTC).\n  Query: %s",
                cw_log_group,
                start_utc.strftime("%Y-%m-%d %H:%M:%S"),
                end_utc.strftime("%Y-%m-%d %H:%M:%S"),
                cw_query,
            )

            try:
                client = boto3.client("logs")
                response = client.start_query(
                    logGroupName=cw_log_group,
                    queryString=cw_query,
                    startTime=int(cw_start_time.timestamp()),
                    endTime=int(cw_end_time.timestamp()),
                )
            except Exception as e:
                raise DataPrepError(f"Failed to query CloudWatch Logs: {e}") from e

            results = self._poll_results(client, response["queryId"])

            if not results:
                logger.warning(
                    "CloudWatch Insights query returned 0 results for log group '%s'.",
                    cw_log_group,
                )
                return

            for row in results:
                yield {field["field"]: field["value"] for field in row}

        self.dataset = _generator
        self._load_path = f"cw:///{log_group}"
        self._last_state = None
        self._is_materialized = False
        self._session_id = None
        return self

    @staticmethod
    def _poll_results(client, query_id: str) -> list:
        """Poll GetQueryResults until the query completes or times out."""
        elapsed = 0
        while True:
            try:
                result = client.get_query_results(queryId=query_id)
            except Exception as e:
                raise DataPrepError(f"Failed to query CloudWatch Logs: {e}") from e

            status = result["status"]

            if status == "Complete":
                return result.get("results", [])

            if status in _TERMINAL_FAILURE_STATUSES:
                raise DataPrepError(
                    f"Query did not complete successfully. Status: {status}"
                ) from None

            elapsed += _POLL_INTERVAL_SECONDS
            if elapsed % 30 == 0:
                logger.info("Query still running... (elapsed: %ds)", elapsed)

            time.sleep(_POLL_INTERVAL_SECONDS)
