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
"""Operation for saving datasets to local or S3 storage."""

from typing import Any

from ...util.dataset_writer import DatasetWriter
from ...util.iterator_utils import peek
from ...util.logging import logger
from .base import DataPrepError, NovaForgeSaveOperation


class SaveOperation(NovaForgeSaveOperation):
    """Save the current dataset to a local file or S3 path."""

    def execute(self, loader: Any, **kwargs) -> str:
        """
        Save the dataset to the specified path.

        Args:
            loader: The DatasetLoader instance.
            save_path: Path where to save the file (.json or .jsonl).

        Returns:
            The path where the file was saved.

        Raises:
            DataPrepError: If the format is unsupported or saving fails.
        """
        save_path: str = kwargs["save_path"]

        # Get iterator from the current dataset
        dataset_iter = loader.dataset()
        peeked_value, dataset_iter = peek(dataset_iter)

        if peeked_value is None:
            logger.warning("Warning: Dataset is empty. An empty dataset will be saved.")
            dataset_iter = iter([])

        try:
            # Determine format
            if save_path.endswith(".jsonl"):
                is_jsonl = True
            elif save_path.endswith(".json"):
                is_jsonl = False
            else:
                raise DataPrepError(
                    "Unsupported format. Use '.json' or '.jsonl' extension"
                )

            # Save to S3 or local file using DatasetWriter
            if save_path.startswith("s3://"):
                DatasetWriter.save_to_s3(save_path, dataset_iter, is_jsonl)
            else:
                DatasetWriter.save_to_local(save_path, dataset_iter, is_jsonl)

            logger.info(f"Dataset saved successfully to {save_path}")
            return save_path

        except Exception as e:
            raise DataPrepError(f"Error saving dataset: {str(e)}")
