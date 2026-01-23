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
import enum
import importlib
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import boto3

from amzn_nova_customization_sdk.model.model_enums import Platform
from amzn_nova_customization_sdk.util.logging import logger


class JobStatus(enum.Enum):
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def _missing_(cls, value: object):
        # Handle aliases
        aliases = {
            "Created": cls.IN_PROGRESS,
            "Running": cls.IN_PROGRESS,
            "Succeeded": cls.COMPLETED,
        }
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        # Treat all other case as FAILED
        return cls.FAILED


class JobStatusManager(ABC):
    def __init__(self):
        self._job_status = JobStatus.IN_PROGRESS
        self._raw_status: str = JobStatus.IN_PROGRESS.value

    @abstractmethod
    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        """
        Get the status of the job

        Returns:
            str: JobStatus, raw status from the job platform
        """
        pass


class SMTJStatusManager(JobStatusManager):
    def __init__(self, sagemaker_client=None):
        super().__init__()
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker")

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if (
            self._job_status == JobStatus.COMPLETED
            or self._job_status == JobStatus.FAILED
        ):
            return self._job_status, self._raw_status

        # Call sagemaker api to get job status
        response = self._sagemaker_client.describe_training_job(TrainingJobName=job_id)
        raw_status = response["TrainingJobStatus"]
        job_status = JobStatus(raw_status)

        # Cache job status
        self._job_status = job_status
        self._raw_status = raw_status

        return job_status, raw_status


class SMHPStatusManager(JobStatusManager):
    def __init__(self, cluster_name: str, namespace: str):
        super().__init__()
        from amzn_nova_customization_sdk.validation.validator import Validator

        Validator.validate_cluster_name(cluster_name=cluster_name)
        Validator.validate_namespace(namespace=namespace)

        self.cluster_name = cluster_name
        self.namespace = namespace

    def _connect_cluster(self):
        response = subprocess.run(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                self.cluster_name,
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if response.stderr:
            logger.error(
                f"Unable to connect to HyperPod cluster {self.cluster_name}: {response.stderr}"
            )
            raise RuntimeError(response.stderr)

        logger.info(
            f"Successfully connected to HyperPod cluster '{self.cluster_name}' in namespace '{self.namespace}'."
        )

    def get_job_status(self, job_id: str) -> tuple[JobStatus, str]:
        if (
            self._job_status == JobStatus.COMPLETED
            or self._job_status == JobStatus.FAILED
        ):
            return self._job_status, self._raw_status

        from amzn_nova_customization_sdk.validation.validator import Validator

        Validator.validate_job_name(job_name=job_id)

        try:
            # Connect cluster before making call
            self._connect_cluster()
            # Call hyperpod CLI to get job status
            result = subprocess.run(
                ["hyperpod", "get-job", "--job-name", job_id],
                capture_output=True,
                text=True,
                check=True,
            )

            response = json.loads(result.stdout)
            status = response.get("Status")

            if status is None:
                # Status is null, job is still pending
                raw_status = "Pending"
                job_status = JobStatus.IN_PROGRESS
            else:
                conditions = status.get("conditions", [])
                if conditions:
                    # Get the last condition (most recent)
                    latest_condition = conditions[-1]
                    raw_status = latest_condition.get("type", "Unknown")
                    job_status = JobStatus(raw_status)
                else:
                    # No conditions but status exists, still in progress
                    raw_status = "Pending"
                    job_status = JobStatus.IN_PROGRESS

            # Cache job status
            self._job_status = job_status
            self._raw_status = raw_status

            return job_status, raw_status

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return JobStatus.IN_PROGRESS, "Unknown"


@dataclass
class BaseJobResult(ABC):
    job_id: str
    started_time: datetime

    def __init__(self, job_id: str, started_time: datetime):
        self.job_id = job_id
        self.started_time = started_time
        self._status_manager: JobStatusManager = self._create_status_manager()
        self._platform = (
            Platform.SMTJ
            if isinstance(self._status_manager, SMTJStatusManager)
            else Platform.SMHP
        )

    @property
    def status_manager(self):
        return self._status_manager

    @property
    def platform(self):
        return self._platform

    @abstractmethod
    def _create_status_manager(self) -> JobStatusManager:
        """
        Create status manager for this job
        :return:
        """
        pass

    def get_job_status(self) -> tuple[JobStatus, str]:
        """
        Get the status of the job

        Returns:
            str: Job status
        """
        return self._status_manager.get_job_status(self.job_id)

    @abstractmethod
    def get(self) -> Dict:
        """
        Get the job result as dict
        :return: job result dict
        """
        pass

    @abstractmethod
    def show(self):
        """
        Print the job result
        """
        pass

    def _to_dict(self):
        """
        Convert the job result to dict
        :return: object as dict
        """
        return asdict(self)

    @classmethod
    def _from_dict(cls, data) -> "BaseJobResult":
        """
        Load the job result from json
        :return: object as dict
        """
        return cls(**data)

    def dump(
        self, file_path: Optional[str] = None, file_name: Optional[str] = None
    ) -> Path:
        """
        Save the job result to file_path path
        :param file_path: Directory path to save the result. Saves to current directory if not provided
        :param file_name: The file name of the result. Default to <job_id>_<platform>.json if not provided
        :return The full result file path
        """
        file_name = file_name or f"{self.job_id}_{self._platform.value}.json"

        if file_path is None:
            full_path = Path(file_name)
        else:
            full_path = Path(file_path) / file_name

        data = self._to_dict()
        data["__class_name__"] = self.__class__.__name__
        with open(full_path, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Job result saved to {full_path}")
        print(f"Job result saved to {full_path}")
        return full_path

    @classmethod
    def load(cls, file_path: str) -> "BaseJobResult":
        """
        Load the job result from file_path path
        :param file_path: file_path to load the result
        return The Job result object
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        class_name = data.pop("__class_name__", None)
        if class_name:
            try:
                module = importlib.import_module(
                    "amzn_nova_customization_sdk.model.result"
                )
                target_class = getattr(module, class_name, None)
                if target_class and issubclass(target_class, BaseJobResult):
                    return target_class._from_dict(data)
                else:
                    raise ValueError(
                        f"Class {class_name} not found or not a subclass of BaseJobResult"
                    )
            except (ImportError, AttributeError, TypeError) as e:
                logger.error(f"Failed to load job result from {file_path}, due to: {e}")
                raise ValueError(
                    f"Unable to load job result from {file_path}, due to {e}"
                ) from e

        raise ValueError(
            f"Unable to load job result from {file_path}, no class name found"
        )
