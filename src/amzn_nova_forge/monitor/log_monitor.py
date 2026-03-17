# Copyright 2025 Amazon Inc

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
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import boto3
import pandas
from matplotlib import pyplot

from amzn_nova_forge.model.model_enums import Platform, TrainingMethod
from amzn_nova_forge.model.result.job_result import (
    BaseJobResult,
    JobStatus,
    SMHPStatusManager,
    SMTJStatusManager,
)
from amzn_nova_forge.util.bedrock import (
    get_bedrock_job_details,
    log_bedrock_job_status,
)
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.metric_util import get_metrics

DEFAULT_SMHP_NAMESPACE = "kubeflow"


class PlatformStrategy(ABC):
    @abstractmethod
    def get_log_group_name(self, job_id: str) -> str:
        pass

    @abstractmethod
    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        pass

    @abstractmethod
    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        pass

    @abstractmethod
    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        pass


class SMTJStrategy(PlatformStrategy):
    def get_log_group_name(self, job_id: str) -> str:
        return "/aws/sagemaker/TrainingJobs"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        response = cloudwatch_logs_client.describe_log_streams(
            logGroupName=log_group_name, logStreamNamePrefix=job_id
        )
        return (
            response["logStreams"][0]["logStreamName"]
            if response["logStreams"]
            else None
        )

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        if not log_stream_name:
            return []

        all_events: List[Dict] = []
        next_token = None

        end_time = end_time or int(datetime.now().timestamp() * 1000)
        while True:
            params: Dict[str, Any] = {
                "endTime": end_time,
                "logGroupName": log_group_name,
                "logStreamName": log_stream_name,
                "startFromHead": start_from_head,
            }

            if limit:
                params["limit"] = min(limit - len(all_events), 10000)

            if start_time:
                params["startTime"] = start_time

            if next_token:
                params["nextToken"] = next_token

            response = cloudwatch_logs_client.get_log_events(**params)
            events = response["events"]

            all_events.extend(events)

            if limit and len(all_events) >= limit:
                all_events = all_events[:limit]
                break

            current_token = next_token
            next_token = response.get(
                "nextForwardToken" if start_from_head else "nextBackwardToken"
            )
            if next_token == current_token:
                break

        return all_events

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        return get_metrics(
            platform=Platform.SMTJ,
            training_method=training_method,
            logs=logs,
            metrics=metrics,
        )


class SMHPStrategy(PlatformStrategy):
    def __init__(self, cluster_name: str, namespace: str, sagemaker_client=None):
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.sagemaker_client = sagemaker_client or boto3.client("sagemaker")
        self._cluster_id: Optional[str] = None

    def get_log_group_name(self, job_id: str) -> str:
        if not self._cluster_id:
            response = self.sagemaker_client.describe_cluster(
                ClusterName=self.cluster_name
            )
            cluster_arn = response["ClusterArn"]
            self._cluster_id = cluster_arn.split("/")[-1]
        return f"/aws/sagemaker/Clusters/{self.cluster_name}/{self._cluster_id}"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        # TODO: add logic to find log stream if we can find nodeId from job_id
        # Currently the SMHP log stream is separated by nodeID rather than job id
        return None

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        all_events: List[Dict] = []
        next_token = None

        end_time = end_time or int(datetime.now().timestamp() * 1000)
        while True:
            # TODO: Add log_stream_name into filter params if it's not None
            params: Dict[str, Any] = {
                "endTime": end_time,
                "logGroupName": log_group_name,
                "logStreamNamePrefix": "SagemakerHyperPodTrainingJob",
                "filterPattern": f"%{job_id}%",
            }

            if limit:
                params["limit"] = min(limit - len(all_events), 10000)

            if start_time:
                params["startTime"] = start_time

            if next_token:
                params["nextToken"] = next_token

            # TODO: change to use get_log_events once SMHP supports separating log stream by job id
            response = cloudwatch_logs_client.filter_log_events(**params)
            events = response["events"]

            all_events.extend(events)

            if limit and len(all_events) >= limit:
                all_events = all_events[:limit]
                break

            next_token = response.get("nextToken")
            if not next_token:
                break

        return all_events

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        return get_metrics(
            platform=Platform.SMHP,
            training_method=training_method,
            logs=logs,
            metrics=metrics,
        )


class BedrockStrategy(PlatformStrategy):
    def __init__(self, bedrock_client=None):
        self.bedrock_client = bedrock_client or boto3.client("bedrock")

    def get_log_group_name(self, job_id: str) -> str:
        # Bedrock customization jobs do not create CloudWatch logs
        # This method is kept for interface compatibility
        return "/aws/bedrock/modelcustomizationjobs"

    def find_log_stream(
        self, job_id: str, cloudwatch_logs_client, log_group_name: str
    ) -> Optional[str]:
        # Bedrock customization jobs do not create CloudWatch logs
        # Return None to indicate logs are not available
        return None

    def get_logs(
        self,
        job_id: str,
        cloudwatch_logs_client,
        log_group_name: str,
        log_stream_name: str,
        limit: Optional[int],
        start_from_head: bool,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        """
        Bedrock customization jobs do not stream logs to CloudWatch.
        Instead, display job status and provide guidance on monitoring.

        Returns empty list to maintain interface compatibility.
        """

        logger.warning(
            "CloudWatch logs are not available for Bedrock customization jobs."
        )

        # Get and display current job status using shared utility
        try:
            response = get_bedrock_job_details(self.bedrock_client, job_id)
            log_bedrock_job_status(response)
        except Exception as e:
            logger.error(f"Error retrieving job status: {e}")

        # Return empty list to maintain interface compatibility
        return []

    def get_metrics(
        self,
        training_method: TrainingMethod,
        logs: Optional[List[Dict]] = None,
        metrics: Optional[List] = None,
    ) -> pandas.DataFrame:
        raise NotImplementedError(f"Metrics not available for Bedrock jobs")


class CloudWatchLogMonitor:
    def __init__(
        self,
        job_id: str,
        platform: Platform,
        started_time: Optional[int] = None,
        cloudwatch_logs_client=None,
        **kwargs,
    ):
        self.job_id = job_id
        self.platform = platform
        self.started_time = started_time
        self.cloudwatch_logs_client = cloudwatch_logs_client or boto3.client("logs")
        self.strategy = self._create_strategy(platform, **kwargs)
        self.log_group_name = self._get_log_group_name()
        self.log_stream_name = self._find_log_stream()
        self.job_status_manager = self._create_job_status_manager()
        self.logs: Optional[List[Dict]] = None

    @staticmethod
    def _create_strategy(platform: Platform, **kwargs):
        if platform == Platform.SMTJ:
            return SMTJStrategy()
        elif platform == Platform.SMHP:
            cluster_name = kwargs.get("cluster_name")
            namespace = kwargs.get("namespace")
            sagemaker_client = kwargs.get("sagemaker_client")
            if not namespace:
                namespace = DEFAULT_SMHP_NAMESPACE
                logger.info(f"No namespace provided, using {namespace}` as default")
            if not cluster_name:
                raise ValueError("SMHP platform requires 'cluster_name' parameters")
            return SMHPStrategy(cluster_name, namespace, sagemaker_client)
        elif platform == Platform.BEDROCK:
            bedrock_client = kwargs.get("bedrock_client")
            return BedrockStrategy(bedrock_client)
        else:
            raise NotImplementedError(f"Unsupported platform: {platform}")

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        platform: Platform,
        started_time: Optional[datetime] = None,
        **kwargs,
    ):
        return cls(
            job_id=job_id,
            platform=platform,
            started_time=int(started_time.timestamp() * 1000) if started_time else None,
            **kwargs,
        )

    @classmethod
    def from_job_result(cls, job_result: BaseJobResult, cloudwatch_logs_client=None):
        if job_result.platform == Platform.SMTJ:
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
            )
        elif job_result.platform == Platform.SMHP:
            job_status_manager = cast(SMHPStatusManager, job_result.status_manager)
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
                cluster_name=job_status_manager.cluster_name,
                namespace=job_status_manager.namespace,
            )
        elif job_result.platform == Platform.BEDROCK:
            # Bedrock doesn't use CloudWatch logs, but we still create the monitor
            # for interface compatibility. The BedrockStrategy will handle showing
            # job status instead of logs.
            return cls(
                job_id=job_result.job_id,
                platform=job_result.platform,
                started_time=int(job_result.started_time.timestamp() * 1000),
                cloudwatch_logs_client=cloudwatch_logs_client,
            )
        else:
            raise NotImplementedError(f"Unsupported platform: {job_result.platform}")

    def _get_log_group_name(self):
        return self.strategy.get_log_group_name(self.job_id)

    def _find_log_stream(self):
        return self.strategy.find_log_stream(
            self.job_id, self.cloudwatch_logs_client, self.log_group_name
        )

    def _create_job_status_manager(self):
        if self.platform == Platform.SMTJ:
            return SMTJStatusManager()
        elif self.platform == Platform.SMHP:
            strategy = cast(SMHPStrategy, self.strategy)
            return SMHPStatusManager(strategy.cluster_name, strategy.namespace)

    def _get_in_range_dataframe(
        self,
        metrics_df: pandas.DataFrame,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
    ):
        metrics_df = metrics_df.drop_duplicates(subset=["global_step"], keep="last")
        if starting_step and ending_step:
            metrics_df = metrics_df[
                (metrics_df["global_step"] >= starting_step)
                & (metrics_df["global_step"] <= ending_step)
            ]
        elif starting_step:
            metrics_df = metrics_df[metrics_df["global_step"] >= starting_step]
        elif ending_step:
            metrics_df = metrics_df[metrics_df["global_step"] <= ending_step]

        if metrics_df.empty:
            raise ValueError(
                f"No metrics found in the specified step range [{starting_step or 'start'}-{ending_step or 'end'}]"
            )
        return metrics_df

    def get_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> Optional[List[Dict]]:
        self.log_stream_name = self.log_stream_name or self._find_log_stream()
        # Cache the latest logs
        self.logs = self.strategy.get_logs(
            job_id=self.job_id,
            cloudwatch_logs_client=self.cloudwatch_logs_client,
            log_group_name=self.log_group_name,
            log_stream_name=self.log_stream_name,
            limit=limit,
            start_from_head=start_from_head,
            start_time=self.started_time,
            end_time=end_time,
        )

        return self.logs

    def show_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ):
        events = self.get_logs(
            limit=limit, start_from_head=start_from_head, end_time=end_time
        )
        if events:
            for event in events:
                print(event["message"].strip())
        else:
            print(f"No logs found for job {self.job_id} yet")

    def plot_metrics(
        self,
        training_method: TrainingMethod,
        metrics: Optional[List] = None,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
    ):
        if starting_step and ending_step and starting_step > ending_step:
            raise ValueError(
                "Starting iteration must be less than or equal to ending iteration"
            )

        try:
            job_in_progress = (
                self.job_status_manager.get_job_status(self.job_id)[0]
                == JobStatus.IN_PROGRESS
            )
        except Exception:
            job_in_progress = False

        if (not self.logs) or job_in_progress:
            self.get_logs()
        if not self.logs:
            raise ValueError("No logs found for this job")

        metrics_df = self.strategy.get_metrics(training_method, self.logs, metrics)
        metrics_df = self._get_in_range_dataframe(
            metrics_df, starting_step, ending_step
        )
        metrics_df = metrics_df.sort_values("global_step").reset_index(drop=True)

        pyplot.figure(figsize=(8, 5))

        for col in metrics_df.columns.drop("global_step"):
            pyplot.plot(metrics_df["global_step"], metrics_df[col], label=col)
        pyplot.xlabel("global_step")
        pyplot.title("Training Metrics")
        pyplot.legend()
        pyplot.grid(True)
        pyplot.style.use("seaborn-v0_8-white")
        pyplot.show()
