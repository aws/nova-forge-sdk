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
import json
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import boto3
import sagemaker
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    HYPERPOD_RECIPE_PATH,
)
from amzn_nova_customization_sdk.util.logging import logger


class RuntimeManager(ABC):
    @property
    @abstractmethod
    def instance_type(self) -> str:
        """Type of instance (e.g., ml.p5.48xlarge)."""
        pass

    @property
    @abstractmethod
    def instance_count(self) -> int:
        """Number of instances used."""
        pass

    @abstractmethod
    def _setup(self) -> None:
        """Prepare environment and dependencies"""
        pass

    @abstractmethod
    def execute(
        self,
        job_name: str,
        data_s3_path: Optional[str],
        output_s3_path: str,
        image_uri: str,
        recipe: str,
        input_s3_data_type: Optional[str],
    ) -> str:
        """Launch a job and return a job id."""
        pass

    @abstractmethod
    def cleanup(self, job_id: str) -> None:
        """Tear down or release resources."""
        pass


class SMTJRuntimeManager(RuntimeManager):
    def __init__(
        self,
        instance_type: str,
        instance_count: int,
        execution_role: Optional[str] = None,
    ):
        self._instance_type = instance_type
        self._instance_count = instance_count
        # NOTE: Not setting execution_role directly due to issues with mypy type inference
        self._execution_role = execution_role
        self._setup()

    @property
    def instance_type(self) -> str:
        return self._instance_type

    @property
    def instance_count(self) -> int:
        return self._instance_count

    def _setup(self) -> None:
        boto_session = boto3.session.Session()
        self.region = boto_session.region_name or "us-east-1"
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, sagemaker_client=self.sagemaker_client
        )

        if self._execution_role is None:
            self.execution_role = sagemaker.get_execution_role(use_default=True)
        else:
            self.execution_role = self._execution_role
        # Delete temporary attribute so customers don't confuse it with the actual attribute
        del self._execution_role

    def execute(
        self,
        job_name: str,
        data_s3_path: Optional[str],
        output_s3_path: str,
        image_uri: str,
        recipe: str,
        input_s3_data_type: Optional[str],
    ) -> str:
        """
        Start a SageMaker training job

        Args:
            job_name: Name of the training job
            data_s3_path: S3 path to input data
            output_s3_path: S3 path for output artifacts
            image_uri: Image URI for training
            recipe: Training recipe
            input_s3_data_type: The s3_data_type of TrainingInput

        Returns:
            str: Training job name
        """
        try:
            tensorboard_output_config = TensorBoardOutputConfig(
                s3_output_path=output_s3_path,
            )

            estimator_config = {
                "output_path": output_s3_path,
                "base_job_name": job_name,
                "role": self.execution_role,
                "instance_count": self.instance_count,
                "instance_type": self.instance_type,
                "training_recipe": recipe,
                "sagemaker_session": self.sagemaker_session,
                "image_uri": image_uri,
                "tensorboard_output_config": tensorboard_output_config,
                "disable_profiler": True,
                "debugger_hook_config": False,
            }

            estimator = PyTorch(**estimator_config)

            # For eval job, the input could be none
            # https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-notebook
            if data_s3_path:
                train_kwargs: Dict[str, Any] = {
                    "s3_data": data_s3_path,
                    "distribution": "FullyReplicated",
                }

                if input_s3_data_type is not None:
                    train_kwargs["s3_data_type"] = input_s3_data_type

                train = TrainingInput(**train_kwargs)

                inputs = {"train": train}

                estimator.fit(inputs=inputs, job_name=job_name, wait=False)
            else:
                estimator.fit(job_name=job_name, wait=False)

            return job_name

        except Exception as e:
            logger.error(f"Failed to start training job: {str(e)}")
            raise

    def cleanup(self, job_name: str) -> None:
        """
        Cleanup resources associated with the training job

        Args:
            job_name: Training job to clean up
        """
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            self.sagemaker_client.close()
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_name}: {str(e)}")
            raise


# TODO: Might need to take RIG as input in case of multiple RIGs
class SMHPRuntimeManager(RuntimeManager):
    def __init__(
        self, instance_type: str, instance_count: int, cluster_name: str, namespace: str
    ):
        self._instance_type = instance_type
        self._instance_count = instance_count
        self.cluster_name = cluster_name
        self.namespace = namespace
        self._setup()

    @property
    def instance_type(self) -> str:
        return self._instance_type

    @property
    def instance_count(self) -> int:
        return self._instance_count

    def _setup(self) -> None:
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

    # TODO: Should adjust the input params of the ABC because HyperPod is a bit different from SMTJ
    def execute(
        self,
        job_name: str,
        data_s3_path: Optional[str],
        output_s3_path: str,
        image_uri: str,
        recipe: str,
        input_s3_data_type: Optional[str],
    ) -> str:
        """
        Start a SageMaker HyperPod job

        Args:
            job_name: Name of the HyperPod job
            data_s3_path: S3 path to input data
            output_s3_path: S3 path for output artifacts
            image_uri: Image URI for training
            recipe: Training recipe
            input_s3_data_type: The s3_data_type of TrainingInput

        Returns:
            str: HyperPod job ID
        """
        try:
            # Scrub recipe path so that it will be recognized by the HyperPod CLI
            recipe = (
                recipe.split(HYPERPOD_RECIPE_PATH, 1)[1]
                .lstrip("/")
                .lstrip("\\")
                .removesuffix(".yaml")
            )

            override_parameters = json.dumps(
                {
                    "instance_type": self.instance_type,
                    "container": image_uri,
                }
            )
            response = subprocess.run(
                [
                    "hyperpod",
                    "start-job",
                    "--namespace",
                    self.namespace,
                    "--recipe",
                    recipe,
                    "--override-parameters",
                    override_parameters,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if matched_job_name := re.search(r"NAME: (\S+)", response.stdout):
                return matched_job_name.group(1)
            raise ValueError(f"Could not find job name in output: {response.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start HyperPod job: {e.stderr}")
            raise

    def cleanup(self, job_name: str) -> None:
        """
        Cleanup resources associated with the HyperPod job

        Args:
            job_name: HyperPod job to clean up
        """
        try:
            response = subprocess.run(
                [
                    "hyperpod",
                    "cancel-job",
                    "--job-name",
                    job_name,
                    "--namespace",
                    self.namespace,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if response.stderr:
                logger.error(f"Failed to cleanup HyperPod job: {response.stderr}")

        except Exception as e:
            logger.error(f"Failed to cleanup HyperPod job '{job_name}': {str(e)}")
            raise
