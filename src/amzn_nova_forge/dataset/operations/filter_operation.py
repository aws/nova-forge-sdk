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
"""Filter operations and the FilterMethod registry."""

import json
import logging
from abc import abstractmethod
from types import ModuleType
from typing import Any, Optional, Tuple, Type
from urllib.parse import urlparse

import boto3

from amzn_nova_forge.core.enums import FilterMethod
from amzn_nova_forge.dataset.data_state import DataLocation, DataState

from .base import FilterOperationResult, NovaForgeFilterOperation
from .utils import convert_to_s3_parquet, upload_local_file_to_s3

logger = logging.getLogger(__name__)


class NovaForgeFilterOperationBase(NovaForgeFilterOperation):
    """Base class for data filtering operations (default_text_filter, exact_dedup_filter, etc.).

    Subclasses implement ``execute(loader, **kwargs)`` which:
    1. Calls ``prepare_input(state)`` to ensure data is on S3 in a compatible format
    2. Resolves the runtime manager
    3. Runs the filter pipeline
    4. Reloads the filtered output into the loader
    5. Calls ``_log_complete(output_path, result)`` to log the result and warn on 100% drop

    All filters must return ``FilterOperationResult`` with ``filtered_count``
    and ``total_count`` populated.

    Subclasses must also implement ``get_supported_runtimes()`` to declare
    which runtime managers they support.
    """

    _FILTER_NAME: str = "Filter"  # Subclasses override with a human-readable name

    def _log_start(
        self, manager: Any, input_path: str, input_format: str, output_path: str
    ) -> None:
        """Log the start of a filter operation with runtime and I/O details."""
        logger.info("Filter: %s (runtime=%s)", self._FILTER_NAME, manager.runtime_name)
        if manager.runtime_config:
            logger.info("  Runtime config: %s", manager.runtime_config)
        logger.info("  Input: %s (%s)", input_path, input_format)
        logger.info("  Output: %s", output_path)

    def _log_complete(self, output_path: str, result: FilterOperationResult) -> None:
        """Log filter completion and warn on 100% drop."""
        logger.info(
            "Filter complete: %s — %s — output at %s",
            self._FILTER_NAME,
            result,
            output_path,
        )
        if result.total_count > 0 and result.filtered_count == result.total_count:
            logger.warning(
                "All %d records were removed by %s. Verify your data format and filter parameters.",
                result.total_count,
                self._FILTER_NAME,
            )

    # Maps data formats to the Glue/Ray input format string.
    # Formats not in this map need SDK-side conversion to Parquet.
    _GLUE_COMPATIBLE_FORMATS = {"jsonl", "json", "parquet"}

    # Translates data format names to Glue API format strings.
    # Glue only understands "jsonl" and "parquet" — JSON is read as jsonl.
    # CSV and Arrow are converted to Parquet by prepare_input() before
    # reaching this map.
    _GLUE_FORMAT_MAP = {
        "jsonl": "jsonl",
        "json": "jsonl",
        "parquet": "parquet",
    }

    @abstractmethod
    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        """Return a tuple of supported RuntimeManager classes for this operation."""
        pass

    def prepare_input(self, state: "DataState", **kwargs) -> "DataState":
        """Ensure the data is on S3 in a Glue-compatible format.

        - Non-compatible formats (e.g. arrow) → convert to Parquet on S3.
        - Local files in compatible formats → upload to S3 as-is.
        - S3 files in compatible formats → passthrough (no changes).

        Args:
            state: Current data state.
            **kwargs: Must include ``output_path`` (S3 URI) as a fallback
                for deriving the upload/conversion destination.

        Returns:
            Updated DataState (only when data was physically transformed).
        """
        is_glue_compatible = state.format in self._GLUE_COMPATIBLE_FORMATS
        needs_conversion = not is_glue_compatible
        needs_upload = state.location == DataLocation.LOCAL and not needs_conversion

        if needs_conversion:
            s3_base = kwargs.get("output_path", state.path)
            if not s3_base.startswith("s3://"):
                raise ValueError(
                    "Filter operations require an S3 output path when input is local. "
                    "Provide output_path as an S3 URI. "
                    "Example: loader.filter(method=..., output_path='s3://bucket/filtered/')"
                )
            logger.info("  Converting input to Parquet for remote execution...")
            converted_dir = convert_to_s3_parquet(state.generator, s3_base)
            return DataState(
                path=converted_dir,
                format="parquet",
                location=DataLocation.S3,
                generator=state.generator,
            )

        if needs_upload:
            s3_base = kwargs.get("output_path", "")
            if not s3_base.startswith("s3://"):
                raise ValueError(
                    "Filter operations require an S3 output path when input is local. "
                    "Provide output_path as an S3 URI. "
                    "Example: loader.filter(method=..., output_path='s3://bucket/filtered/')"
                )
            logger.info("  Uploading local file to S3 for remote execution...")
            uploaded = upload_local_file_to_s3(state.path, s3_base)
            return DataState(
                path=uploaded,
                format=state.format,
                location=DataLocation.S3,
                generator=state.generator,
            )

        return state

    @classmethod
    def _to_glue_format(cls, fmt: str) -> str:
        """Translate a data format name to the Glue API format string.

        Glue only understands "jsonl" and "parquet". CSV and JSON are
        read by Glue the same way as JSONL.
        """
        return cls._GLUE_FORMAT_MAP.get(fmt, "jsonl")

    def _resolve_runtime_manager(self, input_path: str, **kwargs: Any) -> Any:
        """Build or validate a RuntimeManager from kwargs.

        If ``runtime_manager`` is provided, validates it and the paths.
        Otherwise creates an ``SMTJRuntimeManager(data_prep=True)`` — SMTJ is the
        recommended data-prep runtime since AWS Glue for Ray is deprecated in
        April 2026. Callers who still need Glue can pass a
        ``GlueRuntimeManager`` instance explicitly.

        All imports are deferred so the module can be imported without boto3.

        Args:
            input_path: The resolved input path (from DataState, after prepare_input).
            **kwargs: Must include output_path. Optionally runtime_manager,
                plus runtime-specific params.
        """
        from amzn_nova_forge.dataset.operations.utils import (
            validate_paths_for_remote_execution,
        )
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager
        from amzn_nova_forge.manager.runtime_manager import (
            SMTJRuntimeManager,
            SMTJRuntimeMode,
        )

        remote_runtime_types: Tuple[Type, ...] = (
            GlueRuntimeManager,
            SMTJRuntimeManager,
        )

        output_path = kwargs["output_path"]
        operation_name = kwargs.get("operation_name", type(self).__name__)
        runtime_manager = kwargs.get("runtime_manager")

        if runtime_manager is not None:
            supported_runtimes = self.get_supported_runtimes()
            if not isinstance(runtime_manager, supported_runtimes):
                supported = ", ".join(cls.__name__ for cls in supported_runtimes)
                raise TypeError(
                    f"{operation_name} does not support {type(runtime_manager).__name__}. "
                    f"Supported runtime managers: {supported}"
                )
            if isinstance(runtime_manager, remote_runtime_types):
                validate_paths_for_remote_execution(input_path, output_path)
            if isinstance(runtime_manager, SMTJRuntimeManager):
                runtime_manager.set_mode(SMTJRuntimeMode.DATA_PREP)
            return runtime_manager

        validate_paths_for_remote_execution(input_path, output_path)

        logger.info(
            "No runtime_manager supplied — defaulting to SMTJRuntimeManager "
            "in DATA_PREP mode. AWS Glue is closing onboarding for new Ray "
            "customers after April 30, 2026. Existing Glue-on-Ray customers "
            "can continue by passing GlueRuntimeManager(...) explicitly."
        )
        manager = SMTJRuntimeManager(
            instance_type=kwargs.get("instance_type", "ml.m5.2xlarge"),
            instance_count=kwargs.get("instance_count", 1),
            region=kwargs.get("region"),
            poll_interval=kwargs.get("poll_interval", 30),
        )
        manager.set_mode(SMTJRuntimeMode.DATA_PREP)
        return manager


def get_filter_operation(method: FilterMethod) -> NovaForgeFilterOperationBase:
    """Factory that returns the operation instance for a given FilterMethod."""
    from .default_text_filter_operation import DefaultTextFilterOperation
    from .exact_dedup_filter_operation import ExactDedupFilterOperation
    from .fuzzy_dedup_filter_operation import FuzzyDedupFilterOperation
    from .invalid_records_filter_operation import InvalidRecordsFilterOperation
    from .language_detection_filter_operation import LanguageDetectionFilterOperation

    registry: dict[FilterMethod, type[NovaForgeFilterOperationBase]] = {
        FilterMethod.DEFAULT_TEXT_FILTER: DefaultTextFilterOperation,
        FilterMethod.EXACT_DEDUP: ExactDedupFilterOperation,
        FilterMethod.FUZZY_DEDUP: FuzzyDedupFilterOperation,
        FilterMethod.INVALID_RECORDS: InvalidRecordsFilterOperation,
        FilterMethod.LANGUAGE_DETECTION: LanguageDetectionFilterOperation,
    }
    op_class = registry.get(method)
    if op_class is None:
        raise ValueError(
            f"Filter method '{method.value}' is not yet implemented. "
            f"Supported: {[m.value for m in FilterMethod]}."
        )
    return op_class()  # type: ignore[abstract]


def _read_summary_json(
    output_path: str,
    total_key: str = "input_count",
    filtered_key: str = "duplicates_removed",
) -> Tuple[int, int]:
    """Best-effort read of ``_summary.json`` from an S3 output path.

    Returns ``(total_count, filtered_count)`` parsed from the summary
    file.  Falls back to ``(0, 0)`` on any failure (missing file,
    permissions, malformed JSON).
    """
    summary_path = output_path.rstrip("/") + "/_summary.json"
    try:
        parsed = urlparse(summary_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        summary = json.loads(resp["Body"].read().decode("utf-8"))
        return summary.get(total_key, 0), summary.get(filtered_key, 0)
    except Exception:
        logger.warning("Could not read filter summary. Filter counts will be reported as unknown")
        return 0, 0


def _resolve_s3_directory_to_jsonl(s3_path: str) -> str:
    """If *s3_path* is an S3 directory, resolve it to the single .jsonl file inside.

    Returns the original path unchanged when it already points to a file.
    Raises ``ValueError`` if the directory contains zero or more than one .jsonl file.
    """
    if not s3_path.startswith("s3://"):
        return s3_path

    # Heuristic: a path that already ends with .jsonl is a file, not a directory.
    if s3_path.rstrip("/").endswith(".jsonl"):
        return s3_path

    # Treat as a directory prefix — ensure trailing slash.
    prefix_path = s3_path if s3_path.endswith("/") else s3_path + "/"
    bucket, prefix = prefix_path[len("s3://") :].split("/", 1)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    jsonl_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                jsonl_keys.append(obj["Key"])

    if len(jsonl_keys) == 0:
        raise ValueError(f"No .jsonl files found under S3 directory: {s3_path}")
    if len(jsonl_keys) > 1:
        raise ValueError(
            f"Expected exactly 1 .jsonl file under S3 directory {s3_path}, "
            f"but found {len(jsonl_keys)}: {jsonl_keys}"
        )

    return f"s3://{bucket}/{jsonl_keys[0]}"


def _reload_output_into_loader(loader: Any, output_path: str, output_format: str) -> None:
    """Replace the loader's dataset with a generator over the filtered output."""
    import json as _json

    if output_format == "jsonl":
        from amzn_nova_forge.util.recipe import load_file_content

        resolved_path = _resolve_s3_directory_to_jsonl(output_path)

        def curated_generator():
            for line in load_file_content(
                file_path=resolved_path, extension=".jsonl", encoding="utf-8-sig"
            ):
                line = line.strip()
                if line:
                    yield _json.loads(line)

        loader.dataset = curated_generator
    elif output_format == "parquet":

        def parquet_generator():
            import pyarrow.fs as pafs
            import pyarrow.parquet as pq

            if output_path.startswith("s3://"):
                fs = pafs.S3FileSystem()
                s3_path = output_path[len("s3://") :]
                table = pq.read_table(s3_path, filesystem=fs)
            else:
                table = pq.read_table(output_path)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row

        loader.dataset = parquet_generator
    else:
        raise ValueError(f"Unsupported output_format for reload: {output_format!r}")
