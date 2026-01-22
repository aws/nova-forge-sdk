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
import enum
import os
from typing import Dict, List

HYPERPOD_RECIPE_PATH = os.path.join(
    "sagemaker_hyperpod_recipes",
    "recipes_collection",
    "recipes",
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
    RUBRIC_LLM_JUDGE = "rubric_llm_judge"
    RFT_EVAL = "rft_eval"


class EvaluationStrategy(enum.Enum):
    """Enum for evaluation strategies."""

    ZERO_SHOT = "zs"
    ZERO_SHOT_COT = "zs_cot"
    FEW_SHOT = "fs"
    FEW_SHOT_COT = "fs_cot"
    GEN_QA = "gen_qa"  # Strategy specific for bring your own dataset.
    JUDGE = "judge"  # Strategy specific for Nova LLM as Judge and rubric_llm_judge.
    RFT_EVAL = "rft_eval"  # Strategy specific for RFT eval


class EvaluationMetric(enum.Enum):
    """Enum for evaluation metrics."""

    ACCURACY = "accuracy"
    EXACT_MATCH = "exact_match"
    DEFLECTION = "deflection"
    ALL = "all"


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
    EvaluationTask.RUBRIC_LLM_JUDGE: EvaluationStrategy.JUDGE,
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
    EvaluationTask.RUBRIC_LLM_JUDGE: EvaluationMetric.ALL,
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


BYOD_AVAILABLE_EVAL_TASKS: List[str] = [
    EvaluationTask.GEN_QA.value,
    EvaluationTask.LLM_JUDGE.value,
    EvaluationTask.RUBRIC_LLM_JUDGE.value,
    EvaluationTask.RFT_EVAL.value,
]


def get_available_subtasks(task: EvaluationTask) -> List[str]:
    """
    Get available subtasks for a given evaluation task.

    Args:
        task: The evaluation task

    Returns:
        List of available subtasks for the task
    """
    return EVAL_AVAILABLE_SUBTASKS.get(task, [])
