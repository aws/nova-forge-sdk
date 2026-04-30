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
"""
Shared enumerations for Nova Forge SDK.

This module contains all enums used across the SDK.
"""

import enum
from enum import Enum, auto


class Platform(Enum):
    """Supported training platforms."""

    SMTJ = "SMTJ"
    SMHP = "SMHP"
    BEDROCK = "BEDROCK"
    SMTJServerless = "SMTJServerless"
    LOCAL = "LOCAL"
    GLUE = "GLUE"

    def __str__(self):
        """Return the platform name."""
        return self.name


class Version(Enum):
    """Supported Nova Versions (i.e. 1.0, 2.0, etc.)"""

    ONE = auto()
    TWO = auto()


class Model(Enum):
    """Supported Nova models."""

    version: Version
    model_type: str
    model_path: str
    hub_content_name: str

    def __new__(
        cls,
        value,
        version: Version,
        model_type: str,
        model_path: str,
        hub_content_name: str,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.version = version
        obj.model_type = model_type
        obj.model_path = model_path
        obj.hub_content_name = hub_content_name
        return obj

    @classmethod
    def from_model_type(cls, model_type: str) -> "Model":
        for model in cls:
            if model.model_type == model_type:
                return model
        raise ValueError(f"Unknown model_type: {model_type}")

    @classmethod
    def from_model_name(cls, model_name: str) -> "Model":
        for model in cls:
            if model.name == model_name:
                return model
        raise ValueError(f"Unknown model name: {model_name}")

    NOVA_MICRO = (
        "nova_micro",
        Version.ONE,
        "amazon.nova-micro-v1:0:128k",
        "nova-micro/prod",
        "nova-textgeneration-micro",
    )

    NOVA_LITE = (
        "nova_lite",
        Version.ONE,
        "amazon.nova-lite-v1:0:300k",
        "nova-lite/prod",
        "nova-textgeneration-lite",
    )

    NOVA_LITE_2 = (
        "nova_lite_2",
        Version.TWO,
        "amazon.nova-2-lite-v1:0:256k",
        "nova-lite-2/prod",
        "nova-textgeneration-lite-v2",
    )

    NOVA_PRO = (
        "nova_pro",
        Version.ONE,
        "amazon.nova-pro-v1:0:300k",
        "nova-pro/prod",
        "nova-textgeneration-pro",
    )


class TrainingMethod(Enum):
    """Supported training methods."""

    CPT = "cpt"
    DPO_LORA = "dpo_lora"
    DPO_FULL = "dpo_full"
    RFT_LORA = "rft_lora"
    RFT_FULL = "rft_full"
    RFT_MULTITURN_LORA = "rft_multiturn_lora"
    RFT_MULTITURN_FULL = "rft_multiturn_full"
    SFT_LORA = "sft_lora"
    SFT_FULL = "sft_full"
    EVALUATION = "evaluation"

    def __str__(self):
        """Return the training method name."""
        return self.name


class DeployPlatform(Enum):
    """Supported deployment platforms."""

    BEDROCK_OD = "bedrock_od"
    BEDROCK_PT = "bedrock_pt"
    SAGEMAKER = "sagemaker"


# TODO: Figure out why the REPLACE options still have Bedrock saying the model endpoint is in use
# Possibly it's just a delay from when deployment stops showing up in Bedrock vs is actually deleted?
class DeploymentMode(Enum):
    """
    Deployment behavior when an endpoint with the same name already exists.

    This enum defines how the deploy() method should handle conflicts when
    attempting to deploy to an endpoint name that already exists.

    Values:
        FAIL_IF_EXISTS: Raise an error if endpoint already exists (safest, default)
        UPDATE_IF_EXISTS: Try in-place update only, fail if not supported (PT only)
    """

    FAIL_IF_EXISTS = "fail_if_exists"
    UPDATE_IF_EXISTS = "update_if_exists"
    # UPDATE_OR_REPLACE: Try in-place update first, fallback to delete/recreate if update fails
    # UPDATE_OR_REPLACE = "update_or_replace"
    # FORCE_REPLACE: Always delete existing endpoint and create new one (destructive)
    # FORCE_REPLACE = "force_replace"


class EvaluationTask(enum.Enum):
    """Enum for evaluation tasks."""

    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    BBH = "bbh"
    GPQA = "gpqa"
    MATH = "math"
    STRONG_REJECT = "strong_reject"
    GEN_QA = "gen_qa"
    IFEVAL = "ifeval"
    MMMU = "mmmu"
    LLM_JUDGE = "llm_judge"
    RUBRIC_LLM_JUDGE = "rubric_llm_judge"
    RFT_EVAL = "rft_eval"
    RFT_MULTITURN_EVAL = "rft_multiturn_eval"  # Maps to "rft_eval" in recipes

    def get_recipe_value(self) -> str:
        """Get the value to use in recipe templates."""
        if self == EvaluationTask.RFT_MULTITURN_EVAL:
            return "rft_eval"
        return self.value


class EvaluationStrategy(enum.Enum):
    """Enum for evaluation strategies."""

    ZERO_SHOT = "zs"
    ZERO_SHOT_COT = "zs_cot"
    FEW_SHOT = "fs"
    FEW_SHOT_COT = "fs_cot"
    GEN_QA = "gen_qa"  # Strategy specific for bring your own dataset.
    JUDGE = "judge"  # Strategy specific for Nova LLM as Judge and rubric_llm_judge.
    RFT_EVAL = "rft_eval"  # Strategy specific for RFT eval
    RFT_MULTITURN_EVAL = "rft_eval"  # Strategy specific for RFT multiturn eval


class EvaluationMetric(enum.Enum):
    """Enum for evaluation metrics."""

    ACCURACY = "accuracy"
    EXACT_MATCH = "exact_match"
    DEFLECTION = "deflection"
    ALL = "all"


class ModelStatus(Enum):
    """Platform-independent model status."""

    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class FilterMethod(Enum):
    """Supported dataset filter methods."""

    DEFAULT_TEXT_FILTER = "default_text_filter"
    EXACT_DEDUP = "exact_dedup_filter"
    FUZZY_DEDUP = "fuzzy_dedup"
    INVALID_RECORDS = "invalid_records_filter"
    LANGUAGE_DETECTION = "language_detection"
