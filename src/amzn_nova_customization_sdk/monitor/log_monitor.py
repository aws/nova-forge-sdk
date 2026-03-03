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

from amzn_nova_customization_sdk.model.model_enums import Platform
from amzn_nova_customization_sdk.model.result.job_result import (
    BaseJobResult,
    SMHPStatusManager,
)
from amzn_nova_customization_sdk.util.logging import logger

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
        else:
            raise NotImplementedError(f"Unsupported platform: {job_result.platform}")

    def _get_log_group_name(self):
        return self.strategy.get_log_group_name(self.job_id)

    def _find_log_stream(self):
        return self.strategy.find_log_stream(
            self.job_id, self.cloudwatch_logs_client, self.log_group_name
        )

    def get_logs(
        self,
        limit: Optional[int] = None,
        start_from_head: bool = False,
        end_time: Optional[int] = None,
    ) -> List[Dict]:
        self.log_stream_name = self.log_stream_name or self._find_log_stream()
        return self.strategy.get_logs(
            job_id=self.job_id,
            cloudwatch_logs_client=self.cloudwatch_logs_client,
            log_group_name=self.log_group_name,
            log_stream_name=self.log_stream_name,
            limit=limit,
            start_from_head=start_from_head,
            start_time=self.started_time,
            end_time=end_time,
        )

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
