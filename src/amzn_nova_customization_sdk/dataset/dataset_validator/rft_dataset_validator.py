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
RFT (Reinforcement Fine-Tuning) dataset validator for RFT OpenAI format.

This module implements validation for Nova 2.0 RFT datasets in the OpenAI format,
ensuring they meet all requirements for reinforcement fine-tuning with and without tool use.
"""

from typing import Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, field_validator

from amzn_nova_customization_sdk.model.model_enums import Model

from .dataset_validator import (
    BaseDatasetValidator,
)

OPTIONAL_FIELDS = ["id", "reference_answer", "tools"]


class RFTFunctionParameters(BaseModel):
    """Represents parameters for an RFT function."""

    type: str
    properties: dict
    required: Optional[List[str]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, param_type):
        if param_type != "object":
            raise ValueError("Invalid parameters type, must be 'object'")
        return param_type

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, properties):
        if not isinstance(properties, dict):
            raise ValueError("Invalid properties, must be a dictionary")
        return properties


class RFTFunction(BaseModel):
    """Represents an RFT function specification."""

    name: str
    description: str
    parameters: RFTFunctionParameters

    @field_validator("name")
    @classmethod
    def validate_name(cls, name):
        if not name or not name.strip():
            raise ValueError("Invalid function name, cannot be empty")
        return name

    @field_validator("description")
    @classmethod
    def validate_description(cls, description):
        if not description or not description.strip():
            raise ValueError("Invalid function description, cannot be empty")
        return description


class RFTTool(BaseModel):
    """Represents an RFT tool."""

    type: str
    function: RFTFunction

    @field_validator("type")
    @classmethod
    def validate_type(cls, tool_type):
        if tool_type != "function":
            raise ValueError("Invalid tool type, must be 'function'.")
        return tool_type


class RFTMessage(BaseModel):
    """Represents a simple RFT message with optional role and content per RFT specification."""

    role: Optional[str] = None
    content: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, role):
        # Role is optional, but if provided must be valid
        if role is not None:
            valid_roles = ["system", "user", "assistant"]
            if role.lower() not in valid_roles:
                raise ValueError(f"Invalid role, must be one of {valid_roles}")
        return role

    @field_validator("content")
    @classmethod
    def validate_content(cls, content):
        # Content is optional, but if provided must not be empty
        if content is not None:
            if not content.strip():
                raise ValueError("Invalid content, if provided cannot be empty")
        return content


class RFTDatasetSample(BaseModel):
    """Represents an RFT dataset sample with required messages and tools, optional id and reference answer.

    Field requirements per RFT specification:
    - id: Optional - Unique identifier for tracking
    - messages: Required - Array of message objects
    - messages[].role: Optional - "system", "user", or "assistant"
    - messages[].content: Optional - Text content of the message
    - tools: Optional - Tool specifications available to the model
    - reference_answer: Optional - Expected output (string or object)
    """

    id: Optional[str] = None
    messages: List[RFTMessage]
    tools: Optional[List[RFTTool]] = None
    reference_answer: Optional[Union[str, dict]] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, sample_id):
        # id is optional, but if provided must not be empty
        if sample_id is not None and (not sample_id or not sample_id.strip()):
            raise ValueError("Invalid id, if provided cannot be empty.")
        return sample_id

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        if not messages:
            raise ValueError("Invalid messages, you must have at least one message.")

        # Check that messages have valid role sequence if roles are provided
        has_system = any(msg.role and msg.role.lower() == "system" for msg in messages)
        if has_system:
            # If there's a system message, it should be first
            first_role = messages[0].role.lower() if messages[0].role else None
            if first_role != "system":
                raise ValueError(
                    "Invalid messages, system message must be first if present."
                )

        # Check that there's at least one user message
        if not any(msg.role and msg.role.lower() == "user" for msg in messages):
            raise ValueError("Invalid messages, must have at least one user message.")

        return messages

    @field_validator("reference_answer")
    @classmethod
    def validate_reference_answer(cls, reference_answer):
        # reference_answer is optional, but if it's provided, it cannot be empty.
        if reference_answer is not None:
            if isinstance(reference_answer, str):
                if not reference_answer.strip():
                    raise ValueError(
                        "Invalid reference_answer, the string cannot be empty."
                    )
            elif isinstance(reference_answer, dict):
                if not reference_answer:
                    raise ValueError(
                        "Invalid reference_answer, the dict cannot be empty."
                    )
            else:
                raise ValueError(
                    "Invalid reference_answer, must be a string or dictionary."
                )
        return reference_answer

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools):
        # tools is optional, but when provided, it cannot be empty and duplicate names can't exist.
        if tools is not None:
            if len(tools) == 0:
                raise ValueError(
                    "Invalid tools, tools list cannot be empty when provided."
                )

            # Check for duplicate tool names
            tool_names = [tool.function.name for tool in tools]
            if len(tool_names) != len(set(tool_names)):
                raise ValueError("Invalid tools, duplicate tool names found.")
        return tools


class RFTDatasetValidator(BaseDatasetValidator):
    """
    Validator for RFT (Reinforcement Fine-Tuning) datasets in OpenAI format.

    RFT is only supported on Nova 2.0 Lite and requires:
    - Message sequences with proper role ordering
    - Optional tool specifications
    - Optional reference answers for evaluation
    """

    def __init__(self, model: Model):
        """
        Initialize the RFT dataset validator.

        Args:
            model: The Nova model being used (must be NOVA_LITE_2 for RFT)

        Raises:
            ValueError: If the model isn't NOVA_LITE_2.
        """
        super().__init__()
        if model != Model.NOVA_LITE_2:
            raise ValueError(
                f"RFT is only supported on Nova 2.0 Lite (NOVA_LITE_2). "
                f"Current model: {model}. Please use Model.NOVA_LITE_2 for validating and using RFT datasets."
            )

    def get_sample_model(self):
        """
        Returns:
            RFTDatasetSample: The Pydantic model for RFT dataset validation
        """
        return RFTDatasetSample

    def get_success_message(self) -> str:
        """
        Returns:
            str: Success message with sample count
        """
        return f"Validation succeeded for {self.num_samples} samples on an RFT dataset."

    def get_optional_fields(self) -> List[str]:
        """
        Returns:
            OPTIONAL_FIELDS: A list of all the main optional fields for RFT.
        """
        return OPTIONAL_FIELDS
