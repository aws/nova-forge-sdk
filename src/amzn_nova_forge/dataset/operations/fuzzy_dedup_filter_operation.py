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
"""Fuzzy dedup (MinHash LSH, CPU-only) operation for data preparation pipelines.

Usage via loader (recommended)::

    loader.load("s3://my-bucket/raw/data.jsonl")
    loader.filter(
        method=FilterMethod.FUZZY_DEDUP,
        output_path="s3://my-bucket/deduped/",
    ).execute()

Usage standalone::

    from amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation import FuzzyDedupFilterOperation
    from amzn_nova_forge.dataset.data_state import DataLocation, DataState

    op = FuzzyDedupFilterOperation()
    state = DataState(path="s3://my-bucket/raw/", format="parquet", location=DataLocation.S3)
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
from amzn_nova_forge.dataset.operations.base import OperationResult
from amzn_nova_forge.dataset.operations.filter_operation import (
    NovaForgeFilterOperationBase,
    _reload_output_into_loader,
    _resolve_s3_directory_to_jsonl,
    _try_import_internal_only,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig

logger = logging.getLogger(__name__)

_PIPELINE_ID = "fuzzy_dedup_cpu"

# Algorithm-tuning parameters forwarded to the runtime job.
# These map 1:1 to run_fuzzy_dedup() kwargs in AGIDataCurator.
_FUZZY_DEDUP_PARAM_KEYS = frozenset(
    {
        "num_perm",
        "ngram_size",
        "jaccard_threshold",
        "num_bands",
        "rows_per_band",
        "bands_per_iteration",
        "seed",
        "lowercase",
    }
)


class FuzzyDedupFilterOperation(NovaForgeFilterOperationBase):
    """CPU-only fuzzy deduplication via similarity detection.

    Removes near-duplicate records using MinHash LSH. Catches paraphrases,
    minor edits, and boilerplate variants that exact dedup misses.

    Delegates to the ``fuzzy_dedup_cpu`` ForgeWorkflows standalone
    pipeline registered in AGIDataCurator.

    Common parameters:

    - ``threshold`` (float): Similarity threshold (0.0-1.0). Default 0.8.
      Records above this are considered duplicates.

    Advanced parameters (all optional, auto-computed when omitted):

    - ``num_perm`` (int): MinHash permutations. Default 256.
    - ``ngram_size`` (int): Character n-gram size. Default 24.
    - ``num_bands`` / ``rows_per_band``: LSH band config.
    - ``bands_per_iteration`` (int): Bands per memory pass. Default 4.
    - ``seed`` (int): Random seed. Default 42.
    - ``lowercase`` (bool): Lowercase before shingling. Default True.

    Lightweight — no AWS calls until ``execute()`` is invoked.
    """

    __slots__ = ()

    _FILTER_NAME = "Fuzzy Dedup"

    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager

        runtimes: Tuple[Type, ...] = (GlueRuntimeManager,)
        internal_only = _try_import_internal_only()
        if internal_only is not None:
            runtimes = runtimes + (internal_only.SMTJDataPrepRuntimeManager,)
        return runtimes

    def execute(self, loader: Any, **kwargs: Any) -> OperationResult:
        """Execute the fuzzy dedup pipeline.

        Args:
            loader: The DatasetLoader instance (or None for standalone use).
            state: DataState describing the current data (path, format, location).
            output_path: Path for deduplicated output (S3 URI).
            input_format: ``"parquet"`` or ``"jsonl"``. Defaults to state format.
            output_format: ``"parquet"`` or ``"jsonl"``. Default ``"parquet"``.
            text_field: Column/field name containing the text. Default ``"text"``.
            threshold: Similarity threshold (0.0-1.0). Default 0.8.
            runtime_manager: A ``RuntimeManager`` instance. Defaults to
                ``GlueRuntimeManager`` with default settings.
            extra_args: Additional kwargs forwarded to the pipeline.

        Returns:
            OperationResult with output_state describing the deduplicated output.
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
        input_format = kwargs.get("input_format", state.format)
        output_format = kwargs.get("output_format", "parquet")
        text_field = kwargs.get("text_field", "text")
        extra_args = kwargs.get("extra_args")
        job_name = kwargs.get("job_name")

        # Map user-facing "threshold" to internal "jaccard_threshold"
        if "threshold" in kwargs:
            kwargs.setdefault("jaccard_threshold", kwargs["threshold"])

        manager = self._resolve_runtime_manager(input_path, **kwargs)

        self._log_start(manager, input_path, input_format, output_path)
        if "text_field" not in kwargs:
            logger.info("  Using default text_field=%r", text_field)

        merged_extra_args: Dict[str, Any] = {
            **(extra_args or {}),
            "pipeline_id": _PIPELINE_ID,
        }

        # Forward algorithm-tuning params so the runtime job script passes
        # them to ForgeWorkflows.execute("fuzzy_dedup_cpu", **kwargs).
        for key in _FUZZY_DEDUP_PARAM_KEYS:
            if key in kwargs:
                merged_extra_args[key] = kwargs[key]

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

        self._log_complete(output_path)

        output_state = DataState(
            path=output_path,
            format=output_format,
            location=DataLocation.S3 if output_path.startswith("s3://") else DataLocation.LOCAL,
        )

        return OperationResult(
            status="SUCCEEDED",
            output_state=output_state,
        )
