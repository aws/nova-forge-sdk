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
"""Resolve a dataset validator from TrainingMethod using VALIDATOR_CONFIG.

This module consolidates the TrainingMethod → validator mapping that was
previously duplicated in ``invalid_records_filter_operation._get_sample_model``
and ``validate_operation.SchemaValidateOperation.execute``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    EVAL_TASK_VALIDATOR_OVERRIDES,
    VALIDATOR_CONFIG,
)
from amzn_nova_forge.dataset.dataset_validator import (
    CPTDatasetValidator,
    EvalDatasetValidator,
    RFTDatasetValidator,
    RFTMultiturnDatasetValidator,
    SFTDatasetValidator,
)

# These imports are under TYPE_CHECKING to avoid circular import chains
if TYPE_CHECKING:
    from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
        BaseDatasetValidator,
    )
    from amzn_nova_forge.model.model_enums import Model, TrainingMethod
    from amzn_nova_forge.recipe.recipe_config import EvaluationTask

_VALIDATOR_CLASSES = {
    "SFTDatasetValidator": SFTDatasetValidator,
    "RFTDatasetValidator": RFTDatasetValidator,
    "RFTMultiturnDatasetValidator": RFTMultiturnDatasetValidator,
    "CPTDatasetValidator": CPTDatasetValidator,
    "EvalDatasetValidator": EvalDatasetValidator,
}


def resolve_schema_validator(
    training_method: TrainingMethod,
    model: Model,
    eval_task: Optional[EvaluationTask] = None,
) -> Optional[BaseDatasetValidator]:
    """Return a configured schema validator instance, or ``None`` if unavailable.

    Looks up ``VALIDATOR_CONFIG`` by *training_method*, with an override
    check for specific ``eval_task`` values under ``EVALUATION``.
    """
    config = VALIDATOR_CONFIG.get(training_method)
    if config is None:
        return None

    # Check eval_task override
    if eval_task is not None and eval_task.name in EVAL_TASK_VALIDATOR_OVERRIDES:
        config = EVAL_TASK_VALIDATOR_OVERRIDES[eval_task.name]

    validator_name = config.get("validator")
    if validator_name is None:
        return None

    cls = _VALIDATOR_CLASSES[validator_name]
    init_arg = config.get("init_arg", "none")
    if init_arg == "model":
        return cls(model)
    if init_arg == "eval_task":
        return cls(eval_task)
    return cls()
