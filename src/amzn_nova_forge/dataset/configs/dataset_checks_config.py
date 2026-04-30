# Copyright Amazon.com, Inc. or its affiliates

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
"""Unified configuration for dataset checks (filtering and validation).

Each check entry must have:
    - ``name``: identifier for the check
    - ``type``: each type corresponds to an executor function
    - ``filterable``: whether the filter operation can use this check to remove samples
    - ``training_methods``: set of TrainingMethod values this check applies to
    - ``platforms``: set of Platform values this check applies to

"""

from typing import Any, Dict, List, Set

from typing_extensions import TypedDict

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod


class _RequiredCheckFields(TypedDict):
    name: str
    type: str
    filterable: bool
    applicable_training_methods: Set[TrainingMethod]
    applicable_platforms: Set[Platform]
    applicable_models: Set[Model]


class DatasetCheckEntry(_RequiredCheckFields, total=False):
    scope: Set[str]
    pre_training_only: bool
    invalid_keywords: List[str]
    valid_contents: List[str]
    valid_formats: List[str]
    valid_max_limit: int
    valid_min_limit: int
    valid_min_limit_recipe_field: str
    valid_models: Set[Model]


# Validator config: TrainingMethod → how to construct the dataset validator.
#
# ``validator``: string class name (resolved at runtime to avoid circular imports).
# ``init_arg``: what the validator __init__ takes beyond ``self``:
#     - ``"none"``: no args  (e.g. SFTDatasetValidator())
#     - ``"model"``: takes ``model``  (e.g. RFTDatasetValidator(model))
#     - ``"eval_task"``: takes ``eval_task``  (e.g. EvalDatasetValidator(eval_task))
#
# Both ``resolve_validator()`` consumers and ``_get_sample_model()`` are driven
# from this single mapping — adding a new training method only requires one
# entry here.
VALIDATOR_CONFIG: Dict[TrainingMethod, Dict[str, Any]] = {
    TrainingMethod.SFT_LORA: {"validator": "SFTDatasetValidator", "init_arg": "none"},
    TrainingMethod.SFT_FULL: {"validator": "SFTDatasetValidator", "init_arg": "none"},
    TrainingMethod.DPO_LORA: {"validator": None},
    TrainingMethod.DPO_FULL: {"validator": None},
    TrainingMethod.RFT_LORA: {"validator": "RFTDatasetValidator", "init_arg": "model"},
    TrainingMethod.RFT_FULL: {"validator": "RFTDatasetValidator", "init_arg": "model"},
    TrainingMethod.RFT_MULTITURN_LORA: {
        "validator": "RFTMultiturnDatasetValidator",
        "init_arg": "model",
    },
    TrainingMethod.RFT_MULTITURN_FULL: {
        "validator": "RFTMultiturnDatasetValidator",
        "init_arg": "model",
    },
    TrainingMethod.CPT: {"validator": "CPTDatasetValidator", "init_arg": "none"},
    TrainingMethod.EVALUATION: {
        "validator": "EvalDatasetValidator",
        "init_arg": "eval_task",
    },
}

# EVALUATION overrides: specific eval_task values that use a different validator.
EVAL_TASK_VALIDATOR_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "RFT_MULTITURN_EVAL": {
        "validator": "RFTMultiturnDatasetValidator",
        "init_arg": "model",
    },
}

CONVERSE_FORMAT_RESERVED_KEYWORDS = [
    "System:",
    "SYSTEM:",
    "User:",
    "USER:",
    "Bot:",
    "BOT:",
    "Assistant:",
    "ASSISTANT:",
    "Thought:",
    "[EOS]",
    "<image>",
    "<video>",
    "<unk>",
]

MAX_IMAGES_PER_MESSAGE = 10
MAX_IMAGE_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_VIDEOS_PER_MESSAGE = 1
MAX_VIDEO_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_VIDEO_DURATION_SECONDS = 90
MIN_IMAGE_DIMENSION_PIXELS = 25

ALL_MODELS = {Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_LITE_2, Model.NOVA_PRO}

DATASET_CHECKS: List[DatasetCheckEntry] = [
    # Do NOT re-order as certain validations should run before others
    # e.g. validate file sizes before downloading the files
    {
        "name": "converse_format_reserved_keywords",
        "type": "keyword",
        "scope": {
            "ReasoningText",
            "ToolResultContentItem",
            "ContentItem",
            "SystemMessage",
        },
        "filterable": True,
        "invalid_keywords": CONVERSE_FORMAT_RESERVED_KEYWORDS,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "image_file_size",
        "scope": {"ImageContent"},
        "type": "file_size",
        "filterable": True,
        "valid_max_limit": MAX_IMAGE_FILE_SIZE_BYTES,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "video_file_size",
        "scope": {"VideoContent"},
        "type": "file_size",
        "filterable": True,
        "valid_max_limit": MAX_VIDEO_FILE_SIZE_BYTES,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "image_count",
        "scope": {"Message"},
        "type": "content_count",
        "filterable": True,
        "valid_max_limit": MAX_IMAGES_PER_MESSAGE,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "video_count",
        "scope": {"Message"},
        "type": "content_count",
        "filterable": True,
        "valid_max_limit": MAX_VIDEOS_PER_MESSAGE,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "image_format",
        "type": "format_allowlist",
        "scope": {"ImageContent"},
        "filterable": True,
        "valid_formats": ["gif", "jpeg", "png", "webp"],
        "valid_models": {Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_LITE_2, Model.NOVA_PRO},
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP, Platform.BEDROCK},
    },
    {
        "name": "video_format",
        "type": "format_allowlist",
        "scope": {"VideoContent"},
        "filterable": True,
        "valid_formats": ["mov", "mkv", "mp4", "webm"],
        "valid_models": {Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_LITE_2, Model.NOVA_PRO},
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP, Platform.BEDROCK},
    },
    {
        "name": "doc_format",
        "type": "format_allowlist",
        "scope": {"DocContent"},
        "filterable": True,
        "valid_formats": ["pdf"],
        "valid_models": {Model.NOVA_LITE_2},
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP, Platform.BEDROCK},
    },
    {
        "name": "image_dimensions",
        "scope": {"ImageContent"},
        "type": "image_dimensions",
        "filterable": True,
        "valid_min_limit": MIN_IMAGE_DIMENSION_PIXELS,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "video_duration",
        "scope": {"VideoContent"},
        "type": "video_duration",
        "filterable": True,
        "valid_max_limit": MAX_VIDEO_DURATION_SECONDS,
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
        },
        "applicable_platforms": {
            Platform.SMTJ,
            Platform.SMHP,
            Platform.BEDROCK,
        },
    },
    {
        "name": "multimodal_content",
        "type": "content_allowlist",
        "scope": {"Message"},
        "filterable": True,
        "valid_contents": ["video", "image", "document"],
        "valid_models": {Model.NOVA_LITE, Model.NOVA_LITE_2, Model.NOVA_PRO},
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP, Platform.BEDROCK},
    },
    {
        "name": "reasoning_content",
        "type": "content_allowlist",
        "scope": {"Message"},
        "filterable": True,
        "valid_contents": ["reasoningContent"],
        "valid_models": {Model.NOVA_LITE_2},
        "applicable_models": ALL_MODELS,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP, Platform.BEDROCK},
    },
    # Bedrock sample-count bounds
    # Source: MODEL_TO_NUM_SAMPLES_MAP and NOVA_2_0_LITE_SAMPLE_BOUNDS in
    # https://github.com/aws-samples/amazon-nova-samples/blob/main/customization/
    # bedrock-finetuning/understanding/dataset_validation/nova_ft_dataset_validator.py
    # MODEL_TO_NUM_SAMPLES_MAP  -> (8, 20_000) for all models
    # NOVA_2_0_LITE_SAMPLE_BOUNDS -> SFT: (200, 20_000), RFT: (100, 20_000)
    # General Bedrock bounds: 8–20,000 for Nova 1.0 models
    {
        "name": "bedrock_sample_bounds",
        "type": "row_count",
        "filterable": False,
        "pre_training_only": True,
        "applicable_training_methods": {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
            TrainingMethod.DPO_LORA,
            TrainingMethod.DPO_FULL,
            TrainingMethod.CPT,
        },
        "applicable_platforms": {Platform.BEDROCK},
        "applicable_models": {Model.NOVA_MICRO, Model.NOVA_LITE, Model.NOVA_PRO},
        "valid_min_limit": 8,
        "valid_max_limit": 20000,
    },
    # Nova 2.0 Lite on Bedrock — SFT: 200–20,000
    {
        "name": "bedrock_sample_bounds_nova_lite_2_sft",
        "type": "row_count",
        "filterable": False,
        "pre_training_only": True,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.BEDROCK},
        "applicable_models": {Model.NOVA_LITE_2},
        "valid_min_limit": 200,
        "valid_max_limit": 20000,
    },
    # Nova 2.0 Lite on Bedrock — RFT: 100–20,000
    {
        "name": "bedrock_sample_bounds_nova_lite_2_rft",
        "type": "row_count",
        "filterable": False,
        "pre_training_only": True,
        "applicable_training_methods": {TrainingMethod.RFT_LORA, TrainingMethod.RFT_FULL},
        "applicable_platforms": {Platform.BEDROCK},
        "applicable_models": {Model.NOVA_LITE_2},
        "valid_min_limit": 100,
        "valid_max_limit": 20000,
    },
    # SMTJ/SMHP: min row count driven by recipe global_batch_size
    {
        "name": "min_dataset_rows_sft_nova_lite_2",
        "type": "row_count",
        "filterable": False,
        "pre_training_only": False,
        "applicable_training_methods": {TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL},
        "applicable_platforms": {Platform.SMTJ, Platform.SMHP},
        "applicable_models": {Model.NOVA_LITE_2},
        "valid_min_limit_recipe_field": "global_batch_size",
    },
]
