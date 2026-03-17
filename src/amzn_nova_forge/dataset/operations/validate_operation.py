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
"""Validate operations and the ValidateMethod registry."""

from enum import Enum
from typing import Any, Optional

from ...util.iterator_utils import peek
from ...util.logging import logger
from .base import NovaForgeValidateOperation


class SchemaValidateOperation(NovaForgeValidateOperation):
    """Validate the dataset against schema requirements for a training method and model."""

    def execute(self, loader: Any, **kwargs) -> None:
        from ...model.model_enums import Model, TrainingMethod
        from ...recipe.recipe_config import EvaluationTask
        from ..dataset_validator import (
            CPTDatasetValidator,
            EvalDatasetValidator,
            RFTDatasetValidator,
            RFTMultiturnDatasetValidator,
            SFTDatasetValidator,
        )

        training_method: Optional[TrainingMethod] = kwargs.get("training_method")
        model: Optional[Model] = kwargs.get("model")
        eval_task: Optional[EvaluationTask] = kwargs.get("eval_task")

        if training_method is None or model is None:
            raise ValueError(
                "training_method and model are required for schema validation."
            )

        dataset_iter = loader.dataset()
        peeked_value, dataset_iter = peek(dataset_iter)

        if peeked_value is None:
            logger.info("Dataset is empty. Call load() method to load data first")
            return

        if training_method == TrainingMethod.EVALUATION and eval_task is None:
            logger.warning(
                "`eval_task` not provided for EVALUATION method. "
                "Using default evaluation validation. "
                "For RFT Multiturn evaluation, pass eval_task=EvaluationTask.RFT_MULTITURN_EVAL"
            )

        if training_method in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL):
            SFTDatasetValidator().validate(dataset_iter, model)
        elif training_method == TrainingMethod.EVALUATION:
            if eval_task == EvaluationTask.RFT_MULTITURN_EVAL:
                RFTMultiturnDatasetValidator(model).validate(dataset_iter, model)
            else:
                EvalDatasetValidator(eval_task).validate(dataset_iter, model)
        elif training_method in (TrainingMethod.RFT_FULL, TrainingMethod.RFT_LORA):
            RFTDatasetValidator(model).validate(dataset_iter, model)
        elif training_method in (
            TrainingMethod.RFT_MULTITURN_FULL,
            TrainingMethod.RFT_MULTITURN_LORA,
        ):
            RFTMultiturnDatasetValidator(model).validate(dataset_iter, model)
        elif training_method == TrainingMethod.CPT:
            CPTDatasetValidator().validate(dataset_iter, model)
        else:
            logger.info(
                "Skipping validation. Validation isn't available for that model/method combo right now."
            )


class ValidateMethod(Enum):
    """Supported dataset validation methods."""

    SCHEMA = "schema"


def get_validate_operation(method: ValidateMethod) -> NovaForgeValidateOperation:
    """Factory that returns the operation instance for a given ValidateMethod."""
    registry = {
        ValidateMethod.SCHEMA: SchemaValidateOperation,
    }
    op_class = registry.get(method)
    if op_class is None:
        raise ValueError(
            f"Validate method '{method.value}' is not yet implemented. "
            f"Supported: {[m.value for m in ValidateMethod]}."
        )
    return op_class()
