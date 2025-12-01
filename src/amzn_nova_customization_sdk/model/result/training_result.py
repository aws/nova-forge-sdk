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
from abc import ABC
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict

import boto3

from amzn_nova_customization_sdk.model.model_config import ModelArtifacts
from amzn_nova_customization_sdk.model.model_enums import TrainingMethod
from amzn_nova_customization_sdk.model.result.job_result import (
    BaseJobResult,
    JobStatusManager,
    SMHPStatusManager,
    SMTJStatusManager,
)


@dataclass
class TrainingResult(BaseJobResult, ABC):
    method: TrainingMethod
    model_artifacts: ModelArtifacts
    # metrics: Dict[str, float] # TODO: Implement metrics

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
    ):
        self.method = method
        self.model_artifacts = model_artifacts
        super().__init__(job_id, started_time)

    def get(self) -> Dict:
        # TODO: Implement getting detailed train result from s3 output path
        return self._to_dict()

    def show(self):
        # TODO: Implement showing train metrics from train result in s3 output path
        result = self.get()
        if result:
            print(result)


@dataclass
class SMTJTrainingResult(TrainingResult):
    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        sagemaker_client=None,
    ):
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker")
        super().__init__(job_id, started_time, method, model_artifacts)

    def _create_status_manager(self) -> JobStatusManager:
        return SMTJStatusManager(self._sagemaker_client)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "method": self.method.value,
            "model_artifacts": asdict(self.model_artifacts),
        }

    @classmethod
    def _from_dict(cls, data) -> "SMTJTrainingResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            method=TrainingMethod(data["method"]),
            model_artifacts=ModelArtifacts(**data["model_artifacts"]),
        )


@dataclass
class SMHPTrainingResult(TrainingResult):
    cluster_name: str
    namespace: str

    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        method: TrainingMethod,
        model_artifacts: ModelArtifacts,
        cluster_name: str,
        namespace: str = "kubeflow",
    ):
        self.cluster_name = cluster_name
        self.namespace = namespace
        super().__init__(job_id, started_time, method, model_artifacts)

    def _create_status_manager(self) -> JobStatusManager:
        return SMHPStatusManager(self.cluster_name, self.namespace)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "method": self.method.value,
            "model_artifacts": asdict(self.model_artifacts),
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
        }

    @classmethod
    def _from_dict(cls, data) -> "SMHPTrainingResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            method=TrainingMethod(data["method"]),
            model_artifacts=ModelArtifacts(**data["model_artifacts"]),
            cluster_name=data["cluster_name"],
            namespace=data["namespace"],
        )
