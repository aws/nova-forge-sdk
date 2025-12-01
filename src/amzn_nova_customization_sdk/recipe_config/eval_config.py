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
Enums and Configuration Classes for Nova Evaluation

This module defines the enums and configuration classes used in the Nova evaluation process.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    to_primitive,
)


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
    MM_LLM_JUDGE = "mm_llm_judge"
    RFT_EVAL = "rft_eval"


class EvaluationStrategy(enum.Enum):
    """Enum for evaluation strategies."""

    ZERO_SHOT = "zs"
    ZERO_SHOT_COT = "zs_cot"
    FEW_SHOT = "fs"
    FEW_SHOT_COT = "fs_cot"
    GEN_QA = "gen_qa"  # Strategy specific for bring your own dataset.
    JUDGE = "judge"  # Strategy specific for Nova LLM as Judge and mm_llm_judge.
    RFT_EVAL = "rft_eval"  # Strategy specific for RFT eval


class EvaluationMetric(enum.Enum):
    """Enum for evaluation metrics."""

    ACCURACY = "accuracy"
    EXACT_MATCH = "exact_match"
    DEFLECTION = "deflection"
    ALL = "all"


@dataclass
class EvaluationConfig:
    task: EvaluationTask = EvaluationTask.MMLU
    strategy: EvaluationStrategy = EvaluationStrategy.ZERO_SHOT_COT
    metric: EvaluationMetric = EvaluationMetric.ACCURACY
    subtask: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class InferenceConfig:
    max_new_tokens: int = 8196
    top_k: int = -1
    top_p: float = 1.0
    temperature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)


@dataclass
class ProcessingConfig:
    enabled: bool = True


class Aggregation(enum.Enum):
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


class ProcessorLambdaType(enum.Enum):
    CUSTOM_METRICS = "custom_metrics"
    RFT = "rft"


@dataclass
class ProcessorConfig:
    lambda_arn: str
    lambda_type: ProcessorLambdaType = ProcessorLambdaType.CUSTOM_METRICS
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    postprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    aggregation: Aggregation = Aggregation.AVERAGE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorConfig":
        # Check lambda_arn
        if "lambda_arn" not in data or not isinstance(data["lambda_arn"], str):
            raise ValueError("Must provide a str of lambda_arn for processor_config")

        # Handle lambda_type
        lambda_type = data.get("lambda_type", ProcessorLambdaType.CUSTOM_METRICS)
        if isinstance(lambda_type, str):
            lambda_type = ProcessorLambdaType(lambda_type)

        # Handle preprocessing
        preprocessing_config = data.get("preprocessing", {})
        if isinstance(preprocessing_config, dict):
            if "enabled" in preprocessing_config and isinstance(
                preprocessing_config["enabled"], str
            ):
                preprocessing_config["enabled"] = (
                    preprocessing_config["enabled"].lower() == "true"
                )
            preprocessing = ProcessingConfig(**preprocessing_config)

        # Handle postprocessing
        postprocessing_config = data.get("postprocessing", {})
        if isinstance(postprocessing_config, dict):
            if "enabled" in postprocessing_config and isinstance(
                postprocessing_config["enabled"], str
            ):
                postprocessing_config["enabled"] = (
                    postprocessing_config["enabled"].lower() == "true"
                )
            postprocessing = ProcessingConfig(**postprocessing_config)

        # Handle aggregation
        aggregation = data.get("aggregation", Aggregation.AVERAGE)
        if isinstance(aggregation, str):
            aggregation = Aggregation(aggregation)

        return cls(
            lambda_arn=data["lambda_arn"],
            lambda_type=lambda_type,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            aggregation=aggregation,
        )


@dataclass
class RLEnvConfig:
    reward_lambda_arn: str


@dataclass
class EvalRecipeConfig(BaseRecipeConfig):
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    processor: Optional[ProcessorConfig] = None
    rl_env: Optional[RLEnvConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)
