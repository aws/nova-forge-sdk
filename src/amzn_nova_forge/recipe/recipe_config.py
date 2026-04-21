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
Recipe configuration - evaluation tasks, strategies, metrics, and subtask mappings.

Re-exported from amzn_nova_forge.core for backward compatibility.
"""

from amzn_nova_forge.core.constants import (  # noqa: F401
    BYOD_AVAILABLE_EVAL_TASKS,
    EVAL_AVAILABLE_SUBTASKS,
    EVAL_TASK_METRIC_MAP,
    EVAL_TASK_STRATEGY_MAP,
    HYPERPOD_RECIPE_PATH,
    SERVERLESS_CUSTOM_SCORER_EVAL_TASKS,
    get_available_subtasks,
)
from amzn_nova_forge.core.enums import (  # noqa: F401
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
)

__all__ = [
    "EvaluationTask",
    "EvaluationStrategy",
    "EvaluationMetric",
    "HYPERPOD_RECIPE_PATH",
    "EVAL_TASK_STRATEGY_MAP",
    "EVAL_TASK_METRIC_MAP",
    "EVAL_AVAILABLE_SUBTASKS",
    "BYOD_AVAILABLE_EVAL_TASKS",
    "SERVERLESS_CUSTOM_SCORER_EVAL_TASKS",
    "get_available_subtasks",
]
