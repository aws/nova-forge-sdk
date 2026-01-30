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
"""
Eval dataset validator for Nova evaluation format.

This module implements validation for evaluation datasets,
ensuring they meet all requirements for model evaluation.
"""

from typing import Dict, Iterator, List, Optional

from pydantic import BaseModel, field_validator

from amzn_nova_customization_sdk.model.model_enums import Model
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask

from ...util.logging import logger
from .dataset_validator import (
    BaseDatasetValidator,
    check_forbidden_keywords,
)

FORBIDDEN_KEYWORDS = ["Bot:", "<image>", "<video>", "[EOS]", "Assistant:", "User:"]
OPTIONAL_FIELDS = ["system", "images"]  # Check if metadata should be included here
# TODO: Find and add restrictions for image type and metadata.


class ImageContent(BaseModel):
    """Represents and validates image content with base64 data."""

    data: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, data):
        """Validates that the data is a valid base64 image format."""
        if not data.startswith("data:image/"):
            raise ValueError("Image data must start with 'data:image/'")
        if ";base64," not in data.lower():
            raise ValueError("Image data must contain ';base64,' format")
        return data


class EvalDatasetSample(BaseModel):
    """Represents an evaluation dataset sample."""

    query: str
    response: str
    system: Optional[str] = None
    images: Optional[List[ImageContent]] = None
    metadata: Optional[str] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, query):
        """Validates query field."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        found_keywords = check_forbidden_keywords(query)
        if found_keywords:
            keywords_str = ", ".join(found_keywords)
            raise ValueError(
                f"Invalid query, please do not use these keywords: {keywords_str}"
            )
        return query

    @field_validator("response")
    @classmethod
    def validate_response(cls, response):
        """Validates response field."""
        if not response.strip():
            raise ValueError("Response cannot be empty")
        found_keywords = check_forbidden_keywords(response)
        if found_keywords:
            keywords_str = ", ".join(found_keywords)
            raise ValueError(
                f"Invalid response, please do not use these keywords: {keywords_str}"
            )
        return response

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, metadata):
        """Validates metadata field if it exists, else return."""
        if not metadata.strip():
            return metadata
        found_keywords = check_forbidden_keywords(metadata)
        if found_keywords:
            keywords_str = ", ".join(found_keywords)
            raise ValueError(
                f"Invalid response, please do not use these keywords: {keywords_str}"
            )
        return metadata

    @field_validator("system")
    @classmethod
    def validate_system(cls, system):
        """Validates system field."""
        if system is not None:
            found_keywords = check_forbidden_keywords(system)
            if found_keywords:
                keywords_str = ", ".join(found_keywords)
                raise ValueError(
                    f"Invalid system prompt, please do not use these keywords: {keywords_str}"
                )
        return system


class EvalDatasetValidator(BaseDatasetValidator):
    """
    Validator for evaluation datasets.
    """

    def __init__(
        self,
        eval_task: Optional[EvaluationTask] = None,
    ):
        """
        Initialize the evaluation dataset validator.

        Args:
            eval_task: Optional evaluation task type
        """
        super().__init__()
        self.eval_task = eval_task

    # Helper functions for the validate function
    def get_sample_model(self) -> type[BaseModel]:
        return EvalDatasetSample

    def get_success_message(self) -> str:
        return f"Validation succeeded for {self.num_samples} samples on an Evaluation BYOD dataset."

    def validate(self, dataset: Iterator[Dict], model: Model) -> None:
        if self.eval_task and self.eval_task in (
            EvaluationTask.LLM_JUDGE,
            EvaluationTask.RUBRIC_LLM_JUDGE,
        ):
            logger.warning(
                "LLM Judge validation not yet implemented, skipping validation."
            )
            return
        super().validate(dataset, model)

    def get_optional_fields(self) -> List[str]:
        """
        Returns:
            OPTIONAL_FIELDS: A list of all the main optional fields for eval.
        """
        return OPTIONAL_FIELDS
