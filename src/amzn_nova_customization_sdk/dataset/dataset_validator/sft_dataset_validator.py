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
SFT (Supervised Fine-Tuning) dataset validator for Nova Converse format.

This module implements validation for Nova 1.0 and 2.0 datasets in the Nova Converse format,
ensuring they meet all requirements for supervised fine-tuning.
"""

from typing import Dict, Iterator, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from amzn_nova_customization_sdk.model.model_enums import Model, Version

from .dataset_validator import (
    BaseDatasetValidator,
    check_forbidden_keywords,
    is_valid_path,
)

# Format constants, update as necessary.
NOVA_ONE_IMAGE_FORMATS = ["jpeg", "png", "gif", "webp"]
NOVA_TWO_IMAGE_FORMATS = ["png", "jpeg", "gif"]
NOVA_ONE_VIDEO_FORMATS = ["mov", "mkv", "mp4", "webm"]
NOVA_TWO_VIDEO_FORMATS = ["mov", "mkv", "mp4"]
NOVA_TWO_DOC_FORMATS = ["pdf"]
MAX_NUM_IMAGES = 10
FORBIDDEN_KEYWORDS = ["Bot:", "<image>", "<video>", "[EOS]", "Assistant:", "User:"]
OPTIONAL_FIELDS = [
    "system",
    "messages.content.image",
    "messages.content.video",
    "messages.content.reasoningContent",
    "messages.content.toolUse",
    "messages.content.toolResult",
]


class ConverseRoles:
    """Defines the possible roles in a conversation according to converse format"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


CONVERSE_ROLES_WITHOUT_SYSTEM = [ConverseRoles.USER, ConverseRoles.ASSISTANT]


class S3Location(BaseModel):
    """Represents and validates an S3 URI location."""

    uri: str
    bucketOwner: str

    @field_validator("uri")
    def validate_format(cls, uri):
        """Validates that the URI starts with 's3://'."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI, must start with 's3://'")
        is_valid_path(uri.replace("s3://", ""))
        return uri


class Source(BaseModel):
    """Defines the source location for media content."""

    s3Location: S3Location


class ImageContent(BaseModel):
    """Represents and validates image content with format and source."""

    format: str
    source: Source

    @field_validator("format")
    @classmethod
    def validate_format(cls, image_format, info: ValidationInfo):
        """Validates that the image format is supported."""
        model = info.context.get("model") if info.context else None

        if model is not None and model.version == Version.ONE:
            if image_format.lower() not in NOVA_ONE_IMAGE_FORMATS:
                raise ValueError(
                    f"Invalid image format, supported formats are {NOVA_ONE_IMAGE_FORMATS}"
                )
        else:
            if image_format.lower() not in NOVA_TWO_IMAGE_FORMATS:
                raise ValueError(
                    f"Invalid image format, supported formats are {NOVA_TWO_IMAGE_FORMATS}"
                )
        return image_format


class VideoContent(BaseModel):
    """Represents and validates video content with format and source."""

    format: str
    source: Source

    @field_validator("format")
    @classmethod
    def validate_format(cls, video_format, info: ValidationInfo):
        """Validates that the video format is supported."""
        model = info.context.get("model") if info.context else None
        if model is not None and model.version == Version.ONE:
            if video_format.lower() not in NOVA_ONE_VIDEO_FORMATS:
                raise ValueError(
                    f"Invalid video format, supported formats are {NOVA_ONE_VIDEO_FORMATS}"
                )
        else:
            if video_format.lower() not in NOVA_TWO_VIDEO_FORMATS:
                raise ValueError(
                    f"Invalid video format, supported formats are {NOVA_TWO_VIDEO_FORMATS}"
                )
        return video_format


class DocContent(BaseModel):
    """Represents and validates doc content with format and source."""

    format: str
    source: Source

    @field_validator("format")
    @classmethod
    def validate_format(cls, doc_format, info: ValidationInfo):
        """Validates that the image format is supported."""
        model = info.context.get("model") if info.context else None

        if model is not None and model.version != Version.TWO:
            raise ValueError(f"Doc usage is only supported for Nova 2.0.")
        else:
            if doc_format.lower() not in NOVA_TWO_DOC_FORMATS:
                raise ValueError(
                    f"Invalid doc format, supported formats are {NOVA_TWO_DOC_FORMATS}"
                )
        return doc_format


class ReasoningText(BaseModel):
    """Represents reasoning text content."""

    text: str

    @field_validator("text")
    @classmethod
    def validate_text_keywords(cls, text):
        """Validates that reasoning text does not contain forbidden keywords."""
        found_keywords = check_forbidden_keywords(text)
        if found_keywords:
            keywords_str = ", ".join(found_keywords)
            raise ValueError(
                f"Invalid reasoning text, please do not use these keywords: {keywords_str}"
            )
        return text


class ReasoningContent(BaseModel):
    """Represents reasoning content for Nova 2.0."""

    reasoningText: ReasoningText


class InputSchema(BaseModel):
    """Represents the input schema for a tool."""

    json_schema: dict = Field(alias="json")

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema(cls, schema):
        """Validates that the schema is a valid object."""
        if not isinstance(schema, dict):
            raise ValueError(
                "Invalid inputSchema, json field must be a valid JSON Schema object"
            )
        # Basic JSON Schema validation
        if "type" not in schema:
            raise ValueError("Invalid inputSchema, json must have a 'type' field")
        return schema


class ToolSpec(BaseModel):
    """Represents a tool specification."""

    name: str
    description: str
    inputSchema: InputSchema

    @field_validator("name")
    @classmethod
    def validate_name(cls, name):
        if not name or not name.strip():
            raise ValueError("Invalid toolSpec, name cannot be empty")
        return name

    @field_validator("description")
    @classmethod
    def validate_description(cls, description):
        if not description or not description.strip():
            raise ValueError("Invalid toolSpec, description cannot be empty")
        return description


class Tool(BaseModel):
    """Represents a tool with its specification."""

    toolSpec: ToolSpec


class ToolConfig(BaseModel):
    """Represents the tool configuration for a conversation."""

    tools: List[Tool]

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools):
        if not tools:
            raise ValueError("Invalid toolConfig, tools list cannot be empty")
        # Check for duplicate tool names
        tool_names = [tool.toolSpec.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Invalid toolConfig, duplicate tool names found")
        return tools


class ToolUse(BaseModel):
    """Represents a tool use request from the assistant."""

    toolUseId: str
    name: str
    input: dict

    @field_validator("toolUseId")
    @classmethod
    def validate_tool_use_id(cls, tool_use_id):
        if not tool_use_id or not tool_use_id.strip():
            raise ValueError("Invalid toolUse, toolUseId cannot be empty")
        return tool_use_id

    @field_validator("name")
    @classmethod
    def validate_name(cls, name):
        if not name or not name.strip():
            raise ValueError("Invalid toolUse, name cannot be empty")
        return name

    @field_validator("input")
    @classmethod
    def validate_input(cls, input_data):
        if not isinstance(input_data, dict):
            raise ValueError("Invalid toolUse, input must be a JSON object")
        return input_data


class ToolResultContentItem(BaseModel):
    """Represents content within a tool result."""

    text: Optional[str] = None
    json_item: Optional[dict] = Field(None, alias="json")

    @model_validator(mode="after")
    def validate_content(self) -> "ToolResultContentItem":
        if self.text is None and self.json_item is None:
            raise ValueError(
                "Invalid toolResult content, either text or json must be provided"
            )
        if self.text is not None and self.json_item is not None:
            raise ValueError(
                "Invalid toolResult content, cannot have both text and json"
            )
        # Validate text for forbidden keywords if present
        if self.text is not None:
            found_keywords = check_forbidden_keywords(self.text)
            if found_keywords:
                keywords_str = ", ".join(found_keywords)
                raise ValueError(
                    f"Invalid toolResult text content, please do not use these keywords: {keywords_str}"
                )
        return self


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    toolUseId: str
    content: List[ToolResultContentItem]

    @field_validator("toolUseId")
    @classmethod
    def validate_tool_use_id(cls, tool_use_id):
        if not tool_use_id or not tool_use_id.strip():
            raise ValueError("Invalid toolResult, toolUseId cannot be empty")
        return tool_use_id

    @field_validator("content")
    @classmethod
    def validate_content(cls, content):
        if not content:
            raise ValueError("Invalid toolResult, content list cannot be empty")
        return content


class ContentItem(BaseModel):
    """Represents a content item that can contain text, image, video, reasoningContent, toolUse, or toolResult."""

    text: Optional[str] = None
    image: Optional[ImageContent] = None
    video: Optional[VideoContent] = None
    document: Optional[DocContent] = None
    reasoningContent: Optional[ReasoningContent] = None
    toolUse: Optional[ToolUse] = None
    toolResult: Optional[ToolResult] = None

    @model_validator(mode="after")
    def validate_model_fields(self) -> "ContentItem":
        """Validates that at least one content type is provided and enforces content rules."""
        if not any(
            getattr(self, field) is not None
            for field in self.__class__.model_fields.keys()
        ):
            raise ValueError(
                f"Invalid content, at least one of {list(self.__class__.model_fields.keys())} must be provided"
            )

        # Validate that toolUse and toolResult cannot coexist in the same ContentItem
        if self.toolUse is not None and self.toolResult is not None:
            raise ValueError(
                "Invalid content, toolUse and toolResult cannot coexist in the same ContentItem"
            )

        return self

    @field_validator("text")
    @classmethod
    def validate_text_keywords(cls, text):
        """Validates that text does not contain forbidden keywords."""
        if text is not None:
            found_keywords = check_forbidden_keywords(text)
            if found_keywords:
                keywords_str = ", ".join(found_keywords)
                raise ValueError(
                    f"Invalid text content, please do not use these keywords: {keywords_str}"
                )
        return text


class Message(BaseModel):
    """Represents a conversation message with role and content."""

    role: str
    content: List[ContentItem]

    @field_validator("role")
    @classmethod
    def validate_role(cls, role):
        """Validates that the role is either user or assistant."""
        if role.lower() not in CONVERSE_ROLES_WITHOUT_SYSTEM:
            raise ValueError(
                f"Invalid value for role, valid values are {CONVERSE_ROLES_WITHOUT_SYSTEM}"
            )
        return role

    @model_validator(mode="after")
    def validate_content_rules(self, info: ValidationInfo) -> "Message":
        """Validates content rules for provided user and assistant messages."""
        content_items = self.content
        has_video = any(item.video is not None for item in content_items)
        has_image = any(item.image is not None for item in content_items)
        has_document = any(item.document is not None for item in content_items)
        has_reasoning = any(item.reasoningContent is not None for item in content_items)
        has_tool_use = any(item.toolUse is not None for item in content_items)
        has_tool_result = any(item.toolResult is not None for item in content_items)

        model = info.context.get("model") if info.context else None

        if has_image or has_video or has_document:
            if self.role.lower() == "assistant":
                raise ValueError(
                    "Invalid content, multimodal data cannot be included when role is 'assistant'."
                )
            if model == Model.NOVA_MICRO:
                raise ValueError(
                    "Invalid content, multimodal data cannot be used with Nova Micro. Please use another model."
                )

        if has_reasoning and self.role.lower() != "assistant":
            raise ValueError(
                "Invalid content. 'reasoningContent' can only be included when role is 'assistant'."
            )

        if has_reasoning and (model is not None and model.version != Version.TWO):
            raise ValueError(
                "Invalid content. 'reasoningContent' is only supported for Nova 2.0 model training."
            )

        # Validate tool use rules
        if has_tool_use and self.role.lower() != "assistant":
            raise ValueError(
                "Invalid content, toolUse can only be included in assistant messages"
            )

        if has_tool_result and self.role.lower() != "user":
            raise ValueError(
                "Invalid content, toolResult can only be included in user messages"
            )

        return self

    @field_validator("content")
    @classmethod
    def validate_content(cls, content):
        """Validates each message's content against Nova's rules."""
        has_text = any(item.text is not None for item in content)
        has_video = any(item.video is not None for item in content)
        has_image = any(item.image is not None for item in content)
        has_document = any(item.document is not None for item in content)
        has_reasoning = any(item.reasoningContent is not None for item in content)

        total_text_length = sum(
            len(item.text) for item in content if item.text is not None
        )
        if has_text and not (has_image or has_video) and total_text_length == 0:
            raise ValueError("Invalid content, empty text content")

        if sum(1 for item in content if item.video is not None) > 1:
            raise ValueError("Only one video is allowed per sample")

        # Check that a sample only has ONE multimodal type (image, video, doc)
        if sum([has_video, has_image, has_document]) > 1:
            raise ValueError(
                "'content' list can ONLY contain one of the following multimodal types: image, video, OR a document for a given sample.'"
            )

        num_images = sum(1 for item in content if item.image is not None)
        if num_images > MAX_NUM_IMAGES:
            raise ValueError(
                f"Invalid content, number of images {num_images} exceed maximum allowed limit of {MAX_NUM_IMAGES}"
            )

        # Multimodal reasoning content is not supported for SFT. Reasoning mode applies to text-only inputs.
        if has_reasoning and (has_image or has_video or has_document):
            raise ValueError(
                "Multimodal reasoning content is not supported. reasoningContent cannot be combined with image, video, or documents."
            )

        return content


class SystemMessage(BaseModel):
    """Represents a system message with text content."""

    text: str

    @field_validator("text")
    @classmethod
    def validate_text_keywords(cls, text):
        """Validates that system message text does not contain forbidden keywords."""
        found_keywords = check_forbidden_keywords(text)
        if found_keywords:
            keywords_str = ", ".join(found_keywords)
            raise ValueError(
                f"Invalid system message text, please do not use these keywords: {keywords_str}"
            )
        return text


class SFTConverseDatasetSample(BaseModel):
    """Represents a complete conversation sample with system message and message turns."""

    schemaVersion: str
    system: Optional[List[SystemMessage]] = None
    toolConfig: Optional[ToolConfig] = None
    messages: List[Message]

    @field_validator("schemaVersion")
    @classmethod
    def validate_schema_version(cls, schema_version):
        """Validates that schemaVersion is 'bedrock-conversation-2024' which is what is currently required."""
        if schema_version != "bedrock-conversation-2024":
            raise ValueError(
                f"Invalid schemaVersion '{schema_version}', must be 'bedrock-conversation-2024'"
            )
        return schema_version

    @field_validator("messages")
    @classmethod
    def validate_data_sample_rules(cls, messages):
        """Validates the order of the roles between user and assistant in the conversation."""
        check_roles_order(messages)
        return messages

    @model_validator(mode="after")
    def validate_tool_use_rules(
        self, info: ValidationInfo
    ) -> "SFTConverseDatasetSample":
        """Validates tool use rules across the conversation."""
        if self.toolConfig is not None:
            # Check if model supports tool configuration
            model = info.context.get("model") if info.context else None
            if model is not None and model.version != Version.TWO:
                raise ValueError(
                    "Invalid toolConfig. Tool configuration is only supported for Nova Lite 2.0 model training."
                )
            validate_tool_use_in_conversation(self.messages, self.toolConfig)
        return self


def check_roles_order(messages):
    """Validates that messages alternate between user and assistant roles."""
    if len(messages) < 2:
        raise ValueError(
            f"Invalid messages, both {CONVERSE_ROLES_WITHOUT_SYSTEM} are needed in sample"
        )

    for i, message in enumerate(messages):
        if i % 2 == 0 and message.role != ConverseRoles.USER:
            raise ValueError(
                f"Invalid messages, expected {ConverseRoles.USER} role but found {message.role}"
            )
        elif i % 2 == 1 and message.role != ConverseRoles.ASSISTANT:
            raise ValueError(
                f"Invalid messages, expected {ConverseRoles.ASSISTANT} role but found {message.role}"
            )

    if messages[-1].role != ConverseRoles.ASSISTANT:
        raise ValueError(
            f"Invalid messages, last turn should have {ConverseRoles.ASSISTANT} role"
        )


def validate_tool_use_in_conversation(messages: List[Message], tool_config: ToolConfig):
    """Validates tool use consistency across the conversation.

    Ensures:
    - Tool names match configured tools
    - Each toolUseId is used exactly once
    - Tool results reference valid preceding toolUseIds
    - Tool uses appear before their corresponding results
    """
    # TODO confirm if there needs to be a check on unused tools

    # Get configured tool names
    configured_tool_names = {tool.toolSpec.name for tool in tool_config.tools}

    # Track tool use IDs and their positions
    tool_use_ids = {}  # toolUseId -> message_index
    tool_result_ids = set()

    for msg_idx, message in enumerate(messages):
        for content_item in message.content:
            # Check tool uses
            if content_item.toolUse is not None:
                tool_use = content_item.toolUse

                # Validate tool name exists in configuration
                if tool_use.name not in configured_tool_names:
                    raise ValueError(
                        f"Invalid toolUse, tool '{tool_use.name}' not found in toolConfig. "
                        f"Available tools: {', '.join(sorted(configured_tool_names))}"
                    )

                # Check for duplicate toolUseId
                if tool_use.toolUseId in tool_use_ids:
                    raise ValueError(
                        f"Invalid toolUse, duplicate toolUseId '{tool_use.toolUseId}' found. "
                        f"Each toolUseId must be unique within a conversation"
                    )

                tool_use_ids[tool_use.toolUseId] = msg_idx

            # Check tool results
            if content_item.toolResult is not None:
                tool_result = content_item.toolResult

                # Check if toolResult references a valid toolUseId
                if tool_result.toolUseId not in tool_use_ids:
                    raise ValueError(
                        f"Invalid toolResult, toolUseId '{tool_result.toolUseId}' not found in any preceding toolUse"
                    )

                # Check if toolResult appears after its corresponding toolUse
                if msg_idx <= tool_use_ids[tool_result.toolUseId]:
                    raise ValueError(
                        f"Invalid toolResult, toolResult with toolUseId '{tool_result.toolUseId}' "
                        f"must appear after its corresponding toolUse"
                    )

                # Check for duplicate toolResult for same toolUseId
                if tool_result.toolUseId in tool_result_ids:
                    raise ValueError(
                        f"Invalid toolResult, duplicate toolResult for toolUseId '{tool_result.toolUseId}'. "
                        f"Each toolUseId must have exactly one corresponding toolResult"
                    )

                tool_result_ids.add(tool_result.toolUseId)


class SFTDatasetValidator(BaseDatasetValidator):
    """
    Validator for SFT (Supervised Fine-Tuning) datasets in Nova Converse format.
    """

    # Helper functions for the validate function.
    def get_sample_model(self) -> type[BaseModel]:
        return SFTConverseDatasetSample

    def get_success_message(self) -> str:
        return f"Validation succeeded for {self.num_samples} samples on an SFT dataset."

    def get_optional_fields(self) -> List[str]:
        """
        Gets the optional fields. This is used to check consistency across the dataset in dataset_validator.

        Returns:
            OPTIONAL_FIELDS: A list of all the main optional fields for SFT.
        """
        return OPTIONAL_FIELDS
