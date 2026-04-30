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
"""Filter operation that removes samples failing schema validation.

Schema validation uses the same Pydantic models as ``loader.validate(method=ValidateMethod.SCHEMA)``.

Usage via loader::

    loader.filter(
        method=FilterMethod.INVALID_RECORDS,
        training_method=TrainingMethod.SFT_LORA,
        model=Model.NOVA_LITE_2,
        platform=Platform.SMTJ,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import boto3
from pydantic import BaseModel, ValidationError

from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    DATASET_CHECKS,
    DatasetCheckEntry,
)
from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.dataset_validator.dataset_schema_resolver import (
    resolve_schema_validator,
)
from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
    InfrastructureError,
)
from amzn_nova_forge.dataset.operations.base import FilterOperationResult
from amzn_nova_forge.dataset.operations.filter_operation import (
    NovaForgeFilterOperationBase,
)
from amzn_nova_forge.manager.local_runtime_manager import LocalRuntimeManager
from amzn_nova_forge.model.model_enums import Model, Platform, TrainingMethod
from amzn_nova_forge.recipe.recipe_config import EvaluationTask

logger = logging.getLogger(__name__)

# Only include checks that are filterable
FILTER_CHECKS = [c for c in DATASET_CHECKS if c.get("filterable")]


def _get_applicable_checks(
    training_method: TrainingMethod, platform: Platform, model: Model
) -> List[DatasetCheckEntry]:
    """Return checks from config that apply to the given training method and platform."""
    return [
        check
        for check in FILTER_CHECKS
        if training_method in check["applicable_training_methods"]
        and platform in check["applicable_platforms"]
        and model in check["applicable_models"]
    ]


def _get_sample_model(
    training_method: TrainingMethod,
    model: Model,
    eval_task: Any = None,
) -> Optional[type[BaseModel]]:
    """Resolve the Pydantic sample model for a training method.

    Returns ``None`` when no validator is available for the combination.
    """
    validator = resolve_schema_validator(training_method, model, eval_task)
    return validator.get_sample_model() if validator is not None else None


def _sample_fails_schema(
    sample: Dict[str, Any],
    pydantic_model: type[BaseModel],
    nova_model: Model,
    s3_client: Any,
) -> bool:
    """Return True if the sample fails Pydantic schema validation."""
    try:
        context = {"model": nova_model, "s3_client": s3_client}
        pydantic_model.model_validate(sample, context=context)
        return False
    except ValidationError:
        # ValueError raised inside @field_validator is wrapped into ValidationError
        # by Pydantic, so this catches all data validation failures.
        # InfrastructureError is not caught — it propagates to the caller.
        return True


class InvalidRecordsFilterOperation(NovaForgeFilterOperationBase):
    """Removes samples that fail schema validation.

    Schema validation uses the same Pydantic models as
    ``loader.validate(method=ValidateMethod.SCHEMA)``.
    Reserved keyword checks are included in the Pydantic field validators.
    """

    _FILTER_NAME = "Invalid Records"

    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        # Required by NovaForgeFilterOperationBase abstract interface.
        # Post-transform filters bypass _resolve_runtime_manager.
        return (LocalRuntimeManager,)

    def execute(self, loader: Any, **kwargs: Any) -> FilterOperationResult:
        """Execute schema validation, removing failing samples.

        Args:
            loader: The DatasetLoader instance.
            training_method: A ``TrainingMethod`` enum value. Required.
            model: A ``Model`` enum value. Required.
            platform: A ``Platform`` enum value. Required.
            eval_task: Optional ``EvaluationTask`` for EVALUATION method.
            state: DataState (accepted but not used — operates in-memory).
            output_path: str (accepted but not used — operates in-memory).

        Returns:
            A ``FilterOperationResult`` with ``status``, ``filtered_count``,
            ``total_count``, and ``filters_applied``.

        Raises:
            ValueError: If training_method, model, or platform is missing.
        """
        # Accept but ignore output_path — this filter operates in-memory.
        state = kwargs.pop("state", None)
        kwargs.pop("output_path", None)

        training_method = kwargs.get("training_method")
        nova_model = kwargs.get("model")
        platform = kwargs.get("platform")
        eval_task = kwargs.get("eval_task")

        if training_method is None:
            raise ValueError("training_method is required for InvalidRecordsFilterOperation")
        if nova_model is None:
            raise ValueError("model is required for InvalidRecordsFilterOperation")
        if platform is None:
            raise ValueError("platform is required for InvalidRecordsFilterOperation")

        checks = _get_applicable_checks(training_method, platform, nova_model)

        check_names: list[str] = []

        pydantic_model = _get_sample_model(training_method, nova_model, eval_task)
        if pydantic_model is not None:
            # Order should match the order of execution in the filtering loop.
            check_names.insert(0, "schema")

        if not check_names:
            # Build supported list from all filterable checks (including those
            # covered by schema) so the error message is informative.
            all_filterable = [c for c in DATASET_CHECKS if c.get("filterable")]
            supported = sorted(
                {m.value for c in all_filterable for m in c["applicable_training_methods"]}
            )
            raise ValueError(
                f"INVALID_RECORDS filter does not support training method "
                f"'{training_method.value}'. Supported methods: {supported}."
            )

        manager = LocalRuntimeManager()
        self._log_start(manager, "in-memory", "dict", "in-memory")

        original_dataset_fn = loader.dataset

        # Shared mutable result — the generator closure updates counts as
        # records stream through, matching the lazy pattern used by
        # SchemaTransformOperation._wire_transform_generator.
        result = FilterOperationResult(
            status="SUCCEEDED",
            output_state=None,
            filtered_count=0,
            total_count=0,
            filters_applied=check_names,
        )

        filter_op = self

        def filter_generator(
            captured_dataset=original_dataset_fn,
            captured_pydantic_model=pydantic_model,
            captured_nova_model=nova_model,
            captured_result=result,
            captured_filter_op=filter_op,
        ):
            # Reset counts so repeated consumption is idempotent.
            captured_result.total_count = 0
            captured_result.filtered_count = 0

            s3_client = boto3.client("s3")
            for sample in captured_dataset():
                captured_result.total_count += 1
                if captured_pydantic_model is not None and _sample_fails_schema(
                    sample, captured_pydantic_model, captured_nova_model, s3_client
                ):
                    captured_result.filtered_count += 1
                    logger.debug(
                        "Filtered sample %d (schema validation)",
                        captured_result.total_count,
                    )
                else:
                    yield sample

            captured_filter_op._log_complete("in-memory", captured_result)

        loader.dataset = filter_generator

        local_state = DataState(
            path=state.path if state else "",
            format=state.format if state else "unknown",
            location=DataLocation.LOCAL,
            generator=loader.dataset,
        )
        result.output_state = local_state

        return result
