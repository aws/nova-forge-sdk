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
Base validator module for dataset validation across different training methods.

This module provides the abstract base class that all specific validators
must inherit from to ensure consistent validation interface.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

from pydantic import BaseModel, ValidationError

from amzn_nova_customization_sdk.model.model_enums import Model

from ...util.iterator_utils import peek


class BaseDatasetValidator(ABC):
    """
    Abstract base class for dataset validators.

    All training method-specific validators should inherit
    from this class and implement the validate() method.
    """

    def __init__(self):
        """
        Initialize the base dataset validator.
        """
        self.num_samples = 0

    @abstractmethod
    def get_sample_model(self) -> type[BaseModel]:
        """Return the Pydantic model class used to validate individual samples."""
        pass

    @abstractmethod
    def get_success_message(self) -> str:
        """Return the success message for this validator type."""
        pass

    @abstractmethod
    def get_optional_fields(self) -> List[str]:
        """Return a list of optional fields for this validator type."""
        pass

    def validate(self, dataset: Iterator[Dict], model: Model) -> None:
        """
        Validates the entire conversation dataset against Nova format requirements.
        """
        error_message = ""
        failed_samples_id_list = []

        # Track optional field consistency with minimal memory
        optional_fields = self.get_optional_fields()
        field_consistency: Dict[str, bool | None] = {
            field: None for field in optional_fields
        }
        first_sample_with_field: Dict[str, int] = {}

        # Checks the first line of the dataset to quickly validate that required fields are there.
        first_item, dataset = peek(dataset)
        if first_item:
            sample_keys = set(first_item.keys())
            if "messages" not in sample_keys and (
                "question" in sample_keys or "answer" in sample_keys
            ):
                raise ValueError(
                    "Dataset appears to be in a generic format (CSV, plain JSON, etc). "
                    "Please use the loader.transform() method to transform your data to Converse format first."
                )

        # Validate each data entry
        for i, sample in enumerate(dataset):
            try:
                sample_model = self.get_sample_model()
                sample_model.model_validate(sample, context={"model": model})

                # Check optional field consistency
                self._check_optional_field_consistency(
                    sample,
                    i,
                    optional_fields,
                    field_consistency,
                    first_sample_with_field,
                )

                self.num_samples += 1
            except ValidationError as e:
                failed_samples_id_list.append(i)
                error_message += f"\nSample {i}:\n"
                for err in e.errors():
                    err["msg"] = err["msg"].replace("Value error, ", "")
                    sample_error_message = f"  - Location {err['loc']}: {err['msg']} (type={err['type']})\n"
                    error_message += sample_error_message
            except Exception as e:
                raise ValueError(f"Unexpected error in sample {i}: {e}")

        # Report any failed validation results
        if error_message:
            failed_samples_str = format_failed_samples(failed_samples_id_list)
            final_err_msg = (
                f"Validation failed for samples: {failed_samples_str}\n{error_message}"
            )
            raise ValueError(final_err_msg)

        print(f"{self.get_success_message()}")

    def _check_optional_field_consistency(
        self,
        sample: Dict,
        sample_index: int,
        optional_fields: List[str],
        field_consistency: Dict[str, bool | None],
        first_sample_with_field: Dict[str, int],
    ) -> None:
        """
        Check that optional fields are used consistently across all samples.

        If an optional field appears in any sample, it must appear in all samples.

        Args:
            sample: The current sample being validated
            sample_index: The index of the current sample
            optional_fields: List of field names to check for consistency
            field_consistency: Dict tracking field state (None=never seen, True=always present)
            first_sample_with_field: Dict tracking which sample first introduced each field

        Raises:
            ValueError: If an optional field is present in some samples but not others
        """
        for field_name in optional_fields:
            has_field = self._has_nested_field(sample, field_name)

            if field_consistency[field_name] is None:
                # First time seeing this field - record its state
                field_consistency[field_name] = has_field
                if has_field:
                    first_sample_with_field[field_name] = sample_index
            elif field_consistency[field_name] != has_field:
                # Inconsistency detected - fail fast with detailed error
                if has_field:
                    raise ValueError(
                        f"Dataset consistency error: If any sample contains '{field_name}', "
                        f"all samples must contain '{field_name}'. Field first appeared in sample "
                        f"{first_sample_with_field[field_name]} but is missing in earlier samples."
                    )
                else:
                    raise ValueError(
                        f"Dataset consistency error: If any sample contains '{field_name}', "
                        f"all samples must contain '{field_name}'. Field present in sample "
                        f"{first_sample_with_field[field_name]} but missing in sample {sample_index}."
                    )

    def _has_nested_field(self, sample: Dict, field_path: str) -> bool:
        """Check if a top-level field or nested field exists."""
        keys = field_path.split(".")
        current = sample

        for i, key in enumerate(keys):
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                # Check if any item in the list has the key
                return any(
                    self._has_nested_field(item, ".".join(keys[i:]))
                    for item in current
                    if isinstance(item, dict)
                )
            else:
                return False
        return True


# Global helper functions
def check_forbidden_keywords(text: str) -> List[str]:
    """Checks if text contains any forbidden keywords (case-insensitive)."""
    forbidden_keywords = ["Bot:", "<image>", "<video>", "[EOS]", "Assistant:", "User:"]
    found_keywords = []
    text_lower = text.lower()
    for keyword in forbidden_keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    return found_keywords


def is_valid_path(file_path: str) -> None:
    """Validates that file path contains only alphanumeric characters, underscores, hyphens, slashes, and dots."""
    pattern = r"^[\w\-/\.]+$"
    if not re.match(pattern, file_path):
        raise ValueError(
            f"Invalid characters in 'uri'. Only alphanumeric, underscores, hyphens, slashes, and dots are allowed"
        )


def format_failed_samples(failed_samples_id_list: List[int]) -> str:
    """Format the list of failed sample IDs for error messages."""
    if len(failed_samples_id_list) > 3:
        first_sample_id = failed_samples_id_list[0]
        second_sample_id = failed_samples_id_list[1]
        last_sample_id = failed_samples_id_list[-1]
        return f"[{first_sample_id}, {second_sample_id}, ...{last_sample_id}]"
    else:
        return f"{failed_samples_id_list}"
