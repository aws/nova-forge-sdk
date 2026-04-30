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
"""Exact dedup operation for data preparation pipelines.

Usage via loader (recommended)::

    loader.load("s3://my-bucket/raw/data.jsonl")
    loader.filter(
        method=FilterMethod.EXACT_DEDUP,
        output_path="s3://my-bucket/deduped/",
    ).execute()

Usage standalone::

    from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import ExactDedupFilterOperation
    from amzn_nova_forge.dataset.data_state import DataLocation, DataState

    op = ExactDedupFilterOperation()
    state = DataState(path="s3://my-bucket/raw/", format="jsonl", location=DataLocation.S3)
    result = op.execute(
        loader=None,
        state=state,
        output_path="s3://my-bucket/deduped/",
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple, Type

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.base import FilterOperationResult
from amzn_nova_forge.dataset.operations.filter_operation import (
    NovaForgeFilterOperationBase,
    _read_summary_json,
    _reload_output_into_loader,
    _resolve_s3_directory_to_jsonl,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig

logger = logging.getLogger(__name__)

_PIPELINE_ID = "exact_dedup_filter"


class ExactDedupFilterOperation(NovaForgeFilterOperationBase):
    """Exact dedup filter pipeline operation.

    Lightweight operation — no AWS calls until ``execute()`` is invoked.
    All configuration is passed via ``execute(**kwargs)``.
    """

    _FILTER_NAME = "Exact Dedup"

    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeManager

        return (GlueRuntimeManager, SMTJRuntimeManager)

    def execute(self, loader: Any, **kwargs: Any) -> FilterOperationResult:
        """Execute the exact_dedup_filter pipeline.

        Args:
            loader: The DatasetLoader instance (or None for standalone use).
            state: DataState describing the current data (path, format, location).
            output_path: Path for deduplicated output (S3 URI).
            input_format: ``"parquet"`` or ``"jsonl"``. Defaults to state format.
            output_format: ``"parquet"`` or ``"jsonl"``. Default ``"jsonl"``.
            text_field: Column/field name containing the text. Default ``"text"``.
            extra_args: Additional kwargs forwarded to the pipeline builder.
            runtime_manager: A ``RuntimeManager`` instance. Defaults to
                ``SMTJRuntimeManager(data_prep=True)``. Glue is still supported
                but customers must pass ``GlueRuntimeManager(...)`` explicitly.
            job_name: Custom job name. Auto-generated if not provided.
            region: AWS region.
            poll_interval: Seconds between status polls.

        Returns:
            FilterOperationResult with output_state and filter counts.
        """
        state = kwargs.pop("state")
        state = self.prepare_input(state, **kwargs)

        # Resolve S3 directory to actual .jsonl file (Ray requirement)
        if state.format == "jsonl" and state.path.startswith("s3://"):
            state = DataState(
                path=_resolve_s3_directory_to_jsonl(state.path),
                format=state.format,
                location=state.location,
                generator=state.generator,
            )

        input_path = state.path
        kwargs.pop("input_path", None)

        output_path = kwargs["output_path"]
        input_format = self._to_glue_format(kwargs.get("input_format", state.format))
        output_format = kwargs.get("output_format", "jsonl")
        text_field = kwargs.get("text_field", "text")
        extra_args = kwargs.get("extra_args")
        job_name = kwargs.get("job_name")

        manager = self._resolve_runtime_manager(input_path, **kwargs)

        self._log_start(manager, input_path, input_format, output_path)
        if "text_field" not in kwargs:
            logger.info("  Using default text_field=%r", text_field)

        merged_extra_args = {**(extra_args or {}), "pipeline_id": _PIPELINE_ID}
        job_config = DataPrepJobConfig(
            job_name=job_name or f"nova-forge-{_PIPELINE_ID.replace('_', '-')}-{int(time.time())}",
            image_uri="",
            recipe_path="",
            data_s3_path=input_path,
            output_s3_path=output_path,
            input_format=input_format,
            output_format=output_format,
            text_field=text_field,
            extra_args=merged_extra_args,
        )

        job_run_id = manager.execute(job_config)

        if loader is not None and output_path:
            _reload_output_into_loader(loader, output_path, output_format)

        total_count, filtered_count = _read_summary_json(
            output_path,
            total_key="input_count",
            filtered_key="duplicates_removed",
        )

        output_state = DataState(
            path=output_path,
            format=output_format,
            location=DataLocation.S3 if output_path.startswith("s3://") else DataLocation.LOCAL,
        )

        result = FilterOperationResult(
            status="SUCCEEDED",
            output_state=output_state,
            total_count=total_count,
            filtered_count=filtered_count,
        )
        self._log_complete(output_path, result)
        return result
