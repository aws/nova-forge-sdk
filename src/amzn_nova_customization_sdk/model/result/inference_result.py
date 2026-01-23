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
import shutil
import tarfile
import tempfile
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import boto3

from amzn_nova_customization_sdk.model.result.job_result import (
    BaseJobResult,
    JobStatus,
    JobStatusManager,
    SMTJStatusManager,
)
from amzn_nova_customization_sdk.util.logging import logger


@dataclass
class InferenceResult(BaseJobResult, ABC):
    inference_output_path: str

    def __init__(self, job_id: str, started_time: datetime, inference_output_path: str):
        self.inference_output_path = inference_output_path
        self._cached_results_dir: Optional[str] = None
        super().__init__(job_id, started_time)

    @staticmethod
    def _reformat_inference_results(line) -> Dict:
        """
        This reformats each JSONL line into the following format:
        {   "system_prompt":
            "user_response":
            "inference_response":
            "gold_response":
        }

        Args:
            line: This is the line that we want to reformat into the above JSONL format.

        Returns:
            The reformatted JSONL line in the new format.
        """
        entry = json.loads(line.strip())

        # Extract system and user using regex because of more complex structure.
        system_match = re.search(
            r"'role':\s*'system',\s*'content':\s*'([^']+)'", entry["prompt"]
        )
        system_prompt = system_match.group(1) if system_match else ""
        user_match = re.search(
            r"'role':\s*'user',\s*'content':\s*['\"]([^'\"]+)['\"]", entry["prompt"]
        )
        user_response = user_match.group(1) if user_match else ""

        # Clean the inference line
        inference_response = (
            entry["inference"]
            .replace("\\n", "")
            .replace("\\", "")
            .replace("['", "")
            .replace("']", "")
        )
        gold_response = entry["gold"]

        try:
            metadata = entry["metadata"]
        except KeyError:
            metadata = {}

        transformed_entry = {
            "system": system_prompt,
            "query": user_response,
            "gold_response": gold_response,
            "inference_response": inference_response,
        }

        # If metadata is included, append it to the entry before it's returned.
        if metadata:
            transformed_entry.update({"metadata": metadata})

        return transformed_entry

    def _download_inference_results(self) -> str:
        # Check if results are already cached
        if self._cached_results_dir and Path(self._cached_results_dir).exists():
            return self._cached_results_dir
        try:
            # Download and extract results
            logger.info(f"Downloading raw results from {self.inference_output_path}")
            parsed = urlparse(self.inference_output_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            s3_client = boto3.client("s3")

            # Create temp dir for caching results
            self._cached_results_dir = tempfile.mkdtemp(
                prefix=f"inference_results_{self.job_id}_"
            )

            local_file = Path(self._cached_results_dir) / "output.tar.gz"
            s3_client.download_file(bucket, key, str(local_file))
            logger.info(
                f"Successfully downloaded inference results to {self._cached_results_dir}"
            )

            # Only extract the inference_output jsonl file.
            with tarfile.open(local_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("inference_output.jsonl"):
                        tar.extract(member, self._cached_results_dir)
                        break
            return self._cached_results_dir
        except Exception as e:
            logger.error(
                f"Error retrieving inference results: {e}\n"
                f"Results available at: {self.inference_output_path} in the eval_results folder."
            )
            raise

    # 'print' statements are used here to make them visible in the console when a user calls the "show()" method.
    def get(self, s3_path=None) -> Dict:
        job_status, raw_status = self.get_job_status()
        inference_dataset: list[dict] = []
        if job_status == JobStatus.COMPLETED:
            print(f"Job '{self.job_id}' completed successfully.")
            logger.info(f"Job '{self.job_id}' completed successfully.")

            results_dir = self._download_inference_results()
            results_file = list(Path(results_dir).rglob("inference_output.jsonl"))
            if results_file:
                with open(results_file[0], "r") as f:
                    for line in f:
                        formatted_line = self._reformat_inference_results(line)
                        inference_dataset.append(formatted_line)

                    # If a file path is given, try to save the file to that location.
                    if s3_path:
                        try:
                            jsonl_content = "\n".join(
                                json.dumps(inference, ensure_ascii=False)
                                for inference in inference_dataset
                            )

                            if s3_path.startswith("s3://"):
                                s3_path_stripped = s3_path[5:]  # Remove 's3://'
                                bucket, key = s3_path_stripped.split("/", 1)

                                s3_client = boto3.client("s3")
                                s3_client.put_object(
                                    Bucket=bucket,
                                    Key=key,
                                    Body=jsonl_content.encode("utf-8"),
                                    ContentType="application/jsonlines",
                                )
                            else:
                                local_path = Path(s3_path)
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                local_path.write_text(jsonl_content, encoding="utf-8")

                            print(f"Successfully saved the results to {s3_path}.")
                            logger.info(f"Successfully saved the results to {s3_path}.")
                        except Exception as e:
                            print(f"Error saving inference results to {s3_path}: {e}")
                            logger.error(
                                f"Error saving inference results to {s3_path}: {e}"
                            )
                            raise
                    return {"inference_results": inference_dataset}
            else:
                print(f"No inference output jsonl file found for job {self.job_id}")
                logger.info(
                    f"No inference output jsonl file found for job {self.job_id}"
                )
                return {}
        elif job_status == JobStatus.IN_PROGRESS:
            print(
                f"Job '{self.job_id}' still running in progress. Please wait until the job is completed."
            )
            logger.info(
                f"Job '{self.job_id}' still running in progress. Please wait until the job is completed."
            )
            return {}
        else:
            print(
                f"Cannot show inference result. Job '{self.job_id}' in {raw_status} status."
            )
            logger.info(
                f"Cannot show inference result. Job '{self.job_id}' in {raw_status} status."
            )
            return {}

    def show(self):
        results = self.get()
        if results:
            print(f"\nInference Results for job_id={self.job_id}:")
            for result in results["inference_results"]:
                print(json.dumps(result, ensure_ascii=False))

    def clean(self):
        if (
            hasattr(self, "_cached_results_dir")
            and self._cached_results_dir
            and Path(self._cached_results_dir).exists()
        ):
            try:
                shutil.rmtree(self._cached_results_dir)
                logger.debug(
                    f"Successfully cleaned temp file in {self._cached_results_dir}"
                )
                self._cached_results_dir = None
            except Exception as e:
                logger.error(
                    f"Failed to remove cached results directory: {self._cached_results_dir} due to {str(e)}"
                )

    def __del__(self):
        self.clean()


@dataclass
class SMTJBatchInferenceResult(InferenceResult):
    def __init__(
        self,
        job_id: str,
        started_time: datetime,
        inference_output_path: str,
        sagemaker_client=None,
    ):
        self._cached_results_dir: Optional[str] = None
        self._sagemaker_client = sagemaker_client or boto3.client("sagemaker")
        super().__init__(job_id, started_time, inference_output_path)

    def _create_status_manager(self) -> JobStatusManager:
        return SMTJStatusManager(self._sagemaker_client)

    def _to_dict(self):
        return {
            "job_id": self.job_id,
            "started_time": self.started_time.isoformat(),
            "inference_output_path": self.inference_output_path,
        }

    @classmethod
    def _from_dict(cls, data) -> "SMTJBatchInferenceResult":
        return cls(
            job_id=data["job_id"],
            started_time=datetime.fromisoformat(data["started_time"]),
            inference_output_path=data["inference_output_path"],
        )
