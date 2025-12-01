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
Evaluation constraints and validation.

This module defines valid subtasks for each evaluation task and provides validation functionality.
"""

from typing import Any, Dict, List, Optional

from amzn_nova_customization_sdk.model.model_enums import Model, Version
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    EvalRecipeConfig,
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
)
from amzn_nova_customization_sdk.util.logging import logger
from amzn_nova_customization_sdk.util.recipe import (
    get_all_key_names,
    get_all_type_hints,
)
from amzn_nova_customization_sdk.validation.base_validator import (
    Constraints,
    InstanceTypeConstraints,
)

BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT = InstanceTypeConstraints(
    allowed_counts={1, 2, 4, 8, 16}
)

BASIC_INSTANCE_TYPE_CONSTRAINTS = {
    "ml.g5.12xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.g5.24xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.g5.48xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.g6.12xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.g6.24xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.g6.48xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.p5.48xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
    "ml.p4d.24xlarge": BASIC_EVAL_INSTANCE_TYPE_CONSTRAINT,
}

MODEL_INSTANCE_TYPE_CONSTRAINTS = {
    Model.NOVA_MICRO: [
        "ml.g5.4xlarge",
        "ml.g5.8xlarge",
        "ml.g5.12xlarge",
        "ml.g5.16xlarge",
        "ml.g5.24xlarge",
        "ml.g6.4xlarge",
        "ml.g6.8xlarge",
        "ml.g6.12xlarge",
        "ml.g6.16xlarge",
        "ml.g6.24xlarge",
        "ml.g6.48xlarge",
        "ml.p5.48xlarge",
    ],
    Model.NOVA_LITE: [
        "ml.g5.12xlarge",
        "ml.g5.24xlarge",
        "ml.g6.12xlarge",
        "ml.g6.24xlarge",
        "ml.g6.48xlarge",
        "ml.p5.48xlarge",
    ],
    Model.NOVA_PRO: ["ml.p5.48xlarge"],
    Model.NOVA_LITE_2: ["ml.p4d.24xlarge", "ml.p5.48xlarge"],
}

EVAL_CONSTRAINTS: Dict[Model, Constraints] = {
    model: Constraints(
        {
            instance_type: instance_constraint
            for instance_type, instance_constraint in BASIC_INSTANCE_TYPE_CONSTRAINTS.items()
            if instance_type in instance_types
        }
    )
    for model, instance_types in MODEL_INSTANCE_TYPE_CONSTRAINTS.items()
}

EVAL_TASK_STRATEGY_MAP: Dict[EvaluationTask, EvaluationStrategy] = {
    EvaluationTask.MMLU: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.MMLU_PRO: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.BBH: EvaluationStrategy.FEW_SHOT_COT,
    EvaluationTask.GPQA: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.MATH: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.STRONG_REJECT: EvaluationStrategy.ZERO_SHOT,
    EvaluationTask.IFEVAL: EvaluationStrategy.ZERO_SHOT,
    EvaluationTask.GEN_QA: EvaluationStrategy.GEN_QA,
    EvaluationTask.MMMU: EvaluationStrategy.ZERO_SHOT_COT,
    EvaluationTask.LLM_JUDGE: EvaluationStrategy.JUDGE,
    EvaluationTask.MM_LLM_JUDGE: EvaluationStrategy.JUDGE,
    EvaluationTask.RFT_EVAL: EvaluationStrategy.RFT_EVAL,
}

EVAL_TASK_METRIC_MAP: Dict[EvaluationTask, EvaluationMetric] = {
    EvaluationTask.MMLU: EvaluationMetric.ACCURACY,
    EvaluationTask.MMLU_PRO: EvaluationMetric.ACCURACY,
    EvaluationTask.BBH: EvaluationMetric.ACCURACY,
    EvaluationTask.GPQA: EvaluationMetric.ACCURACY,
    EvaluationTask.MATH: EvaluationMetric.EXACT_MATCH,
    EvaluationTask.STRONG_REJECT: EvaluationMetric.DEFLECTION,
    EvaluationTask.IFEVAL: EvaluationMetric.ACCURACY,
    EvaluationTask.GEN_QA: EvaluationMetric.ALL,
    EvaluationTask.MMMU: EvaluationMetric.ACCURACY,
    EvaluationTask.LLM_JUDGE: EvaluationMetric.ALL,
    EvaluationTask.MM_LLM_JUDGE: EvaluationMetric.ALL,
    EvaluationTask.RFT_EVAL: EvaluationMetric.ALL,
}

# Available subtasks for each evaluation task (from nova_evaluator.py)
EVAL_AVAILABLE_SUBTASKS: Dict[EvaluationTask, List[str]] = {
    EvaluationTask.MMLU: [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ],
    EvaluationTask.BBH: [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ],
    EvaluationTask.MATH: [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ],
    EvaluationTask.STRONG_REJECT: [
        "dangerous_or_sensitive_topics",
        "offensive_language",
    ],
    EvaluationTask.MMMU: [
        "accounting",
        "agriculture",
        "architecture_and_engineering",
        "art",
        "art_theory",
        "basic_medical_science",
        "biology",
        "chemistry",
        "clinical_medicine",
        "computer_science",
        "design",
        "economics",
        "electronics",
        "energy_and_power",
        "finance",
        "geography",
        "history",
        "literature",
        "manage",
        "marketing",
        "materials",
        "math",
        "mechanical_engineering",
        "music",
        "pharmacy",
        "physics",
        "psychology",
        "public_health",
        "sociology",
    ],
}

BYOD_AVAILABLE_EVAL_TASKS: List[EvaluationTask] = [
    EvaluationTask.GEN_QA,
    EvaluationTask.LLM_JUDGE,
    EvaluationTask.MM_LLM_JUDGE,
    EvaluationTask.RFT_EVAL,
]


class EvaluationValidator:
    """
    Validates evaluation configurations against defined constraints.

    Provides clear, actionable error messages when configurations are invalid.

    TODO: Create an ABC of Validator that EvaluationValidator, SFTValidator, etc. extend
    """

    @staticmethod
    def get_available_subtasks(task: EvaluationTask) -> List[str]:
        """
        Get available subtasks for a given evaluation task.

        Args:
            task: The evaluation task

        Returns:
            List of available subtasks for the task
        """
        return EVAL_AVAILABLE_SUBTASKS.get(task, [])

    @staticmethod
    def get_constraints(model: Model) -> Optional[Constraints]:
        """
        Get the evaluation constraints for a specific model.

        Args:
            model: The Nova model

        Returns:
            Constraints if defined, None otherwise
        """
        return EVAL_CONSTRAINTS.get(model)

    @staticmethod
    def validate(
        eval_task: EvaluationTask,
        data_s3_path: str,
        model: Model,
        subtask: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
        processor_config: Optional[Dict[str, Any]] = None,
        rl_env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Validate that the evaluation configuration is supported.

        Args:
            eval_task: The evaluation task
            data_s3_path: The input dataset s3 path, only needed for BYOD
            model: The Nova model
            subtask: Subtask for the evaluation (optional)
            instance_type: EC2 instance type (optional)
            instance_count: Number of instances (optional)
            overrides: Dictionary of configuration overrides for eval job (optional)
            processor_config: Optional dict for holding processor_config
            rl_env_config: Optional dict for holding rl_env_config
        Raises:
            ValueError: If the configuration is invalid
        """
        errors: List[str] = []

        # Validate Eval task strategy
        if eval_task not in EVAL_TASK_STRATEGY_MAP:
            errors.append(
                f"Evaluation task '{eval_task.value}' is not currently supported"
            )

        # Validate BYOD task
        if data_s3_path:
            if eval_task not in BYOD_AVAILABLE_EVAL_TASKS:
                errors.append(
                    f"BYOD evaluation must use following eval task: {BYOD_AVAILABLE_EVAL_TASKS}"
                )

        # Validate subtask
        if subtask is not None:
            valid_subtasks = EvaluationValidator.get_available_subtasks(eval_task)
            if not valid_subtasks:
                errors.append(f"Task {eval_task.value} does not support subtasks")
            if subtask not in valid_subtasks:
                errors.append(
                    f'Invalid subtask "{subtask}" for task {eval_task.value}. Valid subtasks: {valid_subtasks}'
                )

        # Validate model constraints if provided
        if instance_type is not None and instance_count is not None:
            constraints = EvaluationValidator.get_constraints(model)
            if constraints:
                allowed_types = constraints.get_all_instance_types()
                if instance_type not in allowed_types:
                    errors.append(
                        f"Instance type '{instance_type}' is not supported. "
                        f"Allowed types: {sorted(allowed_types)}"
                    )
                allowed_counts = constraints.get_allowed_counts_for_type(instance_type)
                if allowed_counts and instance_count not in allowed_counts:
                    errors.append(
                        f"Instance count {instance_count} is not supported for instance type '{instance_type}'. "
                        f"Allowed counts for this type: {sorted(allowed_counts)}"
                    )

        # Validate overrides
        if overrides:
            type_hints = get_all_type_hints(EvalRecipeConfig)
            field_names = get_all_key_names(EvalRecipeConfig)
            for key, value in overrides.items():
                if key not in field_names:
                    logger.info(
                        f"Unknown field '{key}' in overrides, will be ignored later"
                    )
                    continue
                if not isinstance(value, type_hints[key]):
                    errors.append(
                        f"Field {key} expects {type_hints[key].__name__}, got {type(value).__name__}"
                    )
                if key == "max_new_tokens":
                    # Add additional validation for key if required in future
                    pass
                if key == "top_k":
                    # Add additional validation for key if required in future
                    pass
                if key == "top_p":
                    if value < 0.0 or value > 1.0:
                        errors.append(
                            f"Field {key} must be a float between 0.0 and 1.0, got {value}"
                        )
                if key == "temperature":
                    if value < 0.0:
                        errors.append(
                            f"Field {key} must be a positive float, got {value}"
                        )

        # Check processor_config
        if processor_config:
            if eval_task != EvaluationTask.GEN_QA:
                errors.append(
                    f"processor_config is only supported for gen_qa task, but got {eval_task.value}"
                )
            if not processor_config.get("lambda_arn"):
                errors.append("processor_config must contain a lambda_arn")

        # Check rl_env_config
        if rl_env_config:
            if model.version is Version.ONE:
                if not rl_env_config.get("reward_lambda_arn"):
                    errors.append(
                        f"rl_env_config must contain a reward_lambda_arn for model version={model.version}"
                    )
            if model.version is Version.TWO:
                if "single_turn" in rl_env_config:
                    if not rl_env_config.get("single_turn", {}).get("lambda_arn"):
                        errors.append(
                            f"rl_env_config.single_turn must contain a lambda_arn for model version={model.version}"
                        )

        if errors:
            error_msg = (
                f"Invalid evaluation configuration for model={model.value}:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )
            raise ValueError(error_msg)
