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
CPT (Continued Pre-Training) dataset validator for Converse JSONL format.

This module implements validation for Nova CPT datasets in Converse JSONL format,
ensuring they meet all requirements for continued pre-training.
"""

from typing import Dict, Iterator, List

from pydantic import BaseModel, field_validator, model_validator

from amzn_nova_customization_sdk.model.model_enums import Model

from .dataset_validator import BaseDatasetValidator

OPTIONAL_FIELDS: List[str] = []


class CPTDatasetSample(BaseModel):
    """Represents a CPT dataset sample.

    https://docs.aws.amazon.com/sagemaker/latest/dg/nova-cpt-1.html
    https://docs.aws.amazon.com/sagemaker/latest/dg/nova-cpt-2.html
    """

    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, text):
        """Validates that text is not empty or whitespace-only."""
        if not text or not text.strip():
            raise ValueError("Invalid text, cannot be empty or whitespace-only")
        return text

    @model_validator(mode="after")
    def validate_no_extra_fields(self) -> "CPTDatasetSample":
        """Validates that only the 'text' field is present. Pydantic already prevents extra fields by default."""
        return self

    class Config:
        extra = "forbid"


class CPTDatasetValidator(BaseDatasetValidator):
    """
    Validator for CPT (Continued Pre-Training) datasets in Converse JSONL format.

    CPT is supported on all Nova models and requires:
    - Each sample must contain only a "text" field
    - Text must be a non-empty, non-whitespace-only string
    - No other fields are allowed
    """

    def get_sample_model(self) -> type[BaseModel]:
        """
        Returns:
            CPTDatasetSample: The Pydantic model for CPT dataset validation
        """
        return CPTDatasetSample

    def get_success_message(self) -> str:
        """
        Returns:
            str: Success message with sample count
        """
        return f"Validation succeeded for {self.num_samples} samples on a CPT dataset."

    def get_optional_fields(self) -> List[str]:
        """
        Returns:
            OPTIONAL_FIELDS: A list of all the main optional fields for CPT
        """
        return OPTIONAL_FIELDS

    def validate(self, dataset: Iterator[Dict], model: Model) -> None:
        """
        Validates the entire CPT dataset.

        Args:
            dataset: List of dataset samples to validate
            model: The Nova model being used

        Raises:
            ValueError: If validation fails
        """
        # Validate each sample against the Pydantic model
        sample_model = self.get_sample_model()
        for index, sample in enumerate(dataset):
            try:
                sample_model(**sample)
                self.num_samples += 1
            except Exception as e:
                raise ValueError(f"Sample {index} validation failed: {str(e)}")

        print(f"{self.get_success_message()}")
