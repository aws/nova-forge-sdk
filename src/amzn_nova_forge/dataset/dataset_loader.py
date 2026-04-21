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
This module provides classes and utilities for loading and transforming
conversation datasets between different formats (Converse, OpenAI).

Classes:
    DatasetLoader: Abstract base class for dataset loading
    JSONDatasetLoader: Loader for JSON files
    JSONLDatasetLoader: Loader for JSONL files
    CSVDatasetLoader: Loader for CSV files
    ParquetDatasetLoader: Loader for Parquet files
    ArrowDatasetLoader: Loader for Arrow IPC/Feather files

Functionality:
    1. Load data from various sources (local files, S3)
    2. Convert to converse and OpenAI conversation formats.
    3. Split a dataset into train/validation/test sets
    4. Save a generated file locally or to a s3 bucket in JSON/JSONL format.

Supported input formats:
    - Local CSV files with conversation columns
    - Local JSON/JSONL files
    - Local Parquet files
    - Local Arrow IPC/Feather files
    - S3 JSON/JSONL files
    - S3 Parquet files
    - S3 Arrow IPC/Feather files
"""

import csv
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.feather
import pyarrow.fs as pafs
import pyarrow.ipc
import pyarrow.parquet as pq

from amzn_nova_forge.core.enums import EvaluationTask, Model, TrainingMethod
from amzn_nova_forge.telemetry import Feature, _telemetry_emitter

from ..util.logging import logger
from ..util.recipe import load_file_content
from .data_state import DataLocation, DataState, OutputPathResolver
from .file_utils import (
    check_extension,
    check_path_exists,
    is_directory,
    resolve_path,
    scan_directory,
)
from .operations.base import DataPrepError, OperationResult
from .operations.filter_operation import (
    FilterMethod,
    get_filter_operation,
)
from .operations.save_operation import SaveOperation
from .operations.show_operation import ShowOperation
from .operations.split_operation import SplitOperation
from .operations.transform_operation import TransformMethod, get_transform_operation
from .operations.validate_operation import ValidateMethod, get_validate_operation


def _extract_model_training_method(**kwargs):
    """Extra-info callback for telemetry: extracts model & training_method."""
    extra = {}
    model = kwargs.get("model")
    if model is not None:
        extra["model"] = model.value if hasattr(model, "value") else str(model)

    training_method = kwargs.get("training_method")
    # Deprecated positional: the `method` param may actually be a TrainingMethod
    if training_method is None:
        method = kwargs.get("method")
        if isinstance(method, TrainingMethod):
            training_method = method
    if training_method is not None:
        extra["method"] = training_method
    return extra or None


def _extract_filter_method(**kwargs):
    """Extra-info callback for telemetry: extracts the filter method name."""
    method = kwargs.get("method")
    if method is not None:
        return {"filterMethod": method.value}
    return None


# Filters that must run after transform() — they inspect the transformed chat template format.
_POST_TRANSFORM_FILTERS = frozenset({FilterMethod.INVALID_RECORDS})


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Example:
        loader = JSONLDatasetLoader()
        loader.load("data.jsonl")
        loader.transform(
            method=TransformMethod.SCHEMA,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            column_mappings={"question": "q", "answer": "a"},
        )
    """

    def __init__(self, **column_mappings):
        if column_mappings:
            logger.warning(
                "Passing column_mappings to the constructor is deprecated. "
                "Pass column_mappings as a dict to transform() instead. "
                "Example: loader.transform(method=TransformMethod.SCHEMA, "
                "training_method=..., model=..., "
                'column_mappings={"question": "q", "answer": "a"})'
            )
        self.column_mappings = column_mappings
        self._dataset: Callable[[], Iterator[Dict]] = lambda: iter([])
        self._load_path: Optional[str] = None
        self._multimodal_image_bucket: Optional[str] = None

        # Operations
        self._show_op = ShowOperation()
        self._save_op = SaveOperation()
        self._split_op = SplitOperation()

        # Pending operations (lazy — executeed on execute()/save()/show()/split())
        # Each entry is ("filter", FilterMethod, kwargs), ("transform", TransformMethod, kwargs),
        # or ("validate", ValidateMethod, kwargs).
        self._pending_operations: list[tuple[str, Union[FilterMethod, TransformMethod], dict]] = []

        # Tracks whether transform() has been queued (guards filter ordering)
        self._has_transforms: bool = False

        # Persisted state from the last execute() call, so subsequent
        # terminal operations don't replay the entire pipeline.
        self._last_state: Optional["DataState"] = None
        self._is_materialized: bool = False

    # --- Unified dataset accessor ---
    @property
    def dataset(self) -> Callable[[], Iterator[Dict]]:
        """The current dataset callable. Each operation reads/writes this."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Callable[[], Iterator[Dict]]) -> None:
        self._dataset = value

    _EXTENSIONS: set[str] = set()

    # Canonical data format name (e.g. "jsonl", "parquet", "arrow").
    # Used by DataState to tell downstream operations (like filter) what
    # format the loaded data is in, so they can decide whether conversion
    # is needed before execution.
    _FORMAT: str = "unknown"

    def _get_format(self) -> str:
        """Return the canonical data format name for this loader."""
        return self._FORMAT

    @abstractmethod
    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single file.

        Subclasses implement this to define format-specific reading logic.
        Must NOT mutate ``self`` — the caller handles wiring.

        Args:
            path: Resolved absolute local path or S3 URI.

        Returns: A zero-arg callable that yields dicts.
        """
        pass

    @_telemetry_emitter(Feature.DATA_PREP, "load")
    def load(self, path: str) -> "DatasetLoader":
        """Load dataset from a file or directory path.

        Resolves relative/tilde paths, detects directories, and delegates
        single-file generator creation to the subclass's ``_make_single_file_generator()``.

        Args:
            path: Local file/directory path or S3 URI.

        Returns: self (for method chaining)

        Raises:
            DataPrepError: If the path does not exist or has an unexpected extension.
        """
        path = resolve_path(path)

        if is_directory(path):
            files = scan_directory(path, self._EXTENSIONS)

            def dir_generator():
                for fp in files:
                    yield from self._make_single_file_generator(fp)()

            self.dataset = dir_generator
        else:
            check_path_exists(path)
            check_extension(path, self._EXTENSIONS)
            self.dataset = self._make_single_file_generator(path)

        self._load_path = path
        self._last_state = None  # Reset persisted state for new pipeline
        self._is_materialized = False
        return self

    @_telemetry_emitter(Feature.DATA_PREP, "show")
    def show(self, n: int = 10) -> None:
        """
        Display the first n rows of the dataset.

        Args:
            n: Number of rows to display (default: 10)
        """
        self._flush_pending()
        self._show_op.execute(self, n=n)

    @_telemetry_emitter(Feature.DATA_PREP, "split")
    def split(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple["DatasetLoader", "DatasetLoader", "DatasetLoader"]:
        """
        Split data into train, validation, and test DatasetLoader objects.

        Args:
            train_ratio: The fraction of data to train on.
            val_ratio: The fraction of data for validation.
            test_ratio: The fraction of data to test on.
            seed: Value used for random generation.

        Returns: Tuple of three DatasetLoader objects (train, val, test)
        """
        self._flush_pending()
        return self._split_op.execute(
            self,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    def split_data(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple["DatasetLoader", "DatasetLoader", "DatasetLoader"]:
        """Deprecated: Use split() instead."""
        logger.warning("split_data() is deprecated, use split() instead.")
        return self.split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    @_telemetry_emitter(
        Feature.DATA_PREP, "transform", extra_info_fn=_extract_model_training_method
    )
    def transform(
        self, method=TransformMethod.SCHEMA, model=None, eval_task=None, **kwargs
    ) -> "DatasetLoader":
        """
        Transform the dataset using the specified method.

        Example:
            loader.transform(
                method=TransformMethod.SCHEMA,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                column_mappings={"question": "q", "answer": "a"},
            )

        Args:
            method: The transform method (default: TransformMethod.SCHEMA).
                Also accepts a TrainingMethod enum for backward compatibility (deprecated).
            model: The Model. Can be passed positionally for backward compatibility.
            eval_task: Optional evaluation task. Can be passed positionally for backward compatibility.
            **kwargs: Method-specific arguments passed to the operation.
                For TransformMethod.SCHEMA:
                    training_method (TrainingMethod): Required. The training method.
                    model (Model): Required. The target model.
                    eval_task (EvaluationTask): Optional. Required when training_method is EVALUATION.
                    column_mappings (dict): Optional. Maps standard column names to your dataset's column names.

        Returns:
            self (for method chaining)
        """
        # Handle deprecated usage: transform(TrainingMethod.X, Model.Y) positionally
        _deprecated_positional = isinstance(method, TrainingMethod)
        if _deprecated_positional:
            logger.warning(
                "transform(method=TrainingMethod) is deprecated. "
                "Use training_method= instead. "
                "Example: loader.transform(method=TransformMethod.SCHEMA, "
                "training_method=TrainingMethod.SFT_LORA, model=...)"
            )
            kwargs.setdefault("training_method", method)
            method = TransformMethod.SCHEMA

        # Handle model passed as explicit param — only warn for old positional API
        if isinstance(model, Model):
            if _deprecated_positional:
                logger.warning(
                    "transform(TrainingMethod, Model) is deprecated. "
                    "Use training_method= and model= instead. "
                    "Example: loader.transform(method=TransformMethod.SCHEMA, "
                    "training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)"
                )
            kwargs.setdefault("model", model)

        # Handle eval_task passed as explicit param — only warn for old positional API
        if isinstance(eval_task, EvaluationTask):
            if _deprecated_positional:
                logger.warning(
                    "Passing eval_task positionally is deprecated. "
                    "Use eval_task= as a keyword argument instead."
                )
            kwargs.setdefault("eval_task", eval_task)

        # Handle deprecated usage: column_mappings supplied via constructor
        if "column_mappings" not in kwargs or kwargs["column_mappings"] is None:
            kwargs["column_mappings"] = self.column_mappings

        self._pending_operations.append(("transform", method, kwargs))
        logger.info("Queued transform: %s", method.value)
        self._has_transforms = True
        return self

    @_telemetry_emitter(Feature.DATA_PREP, "validate", extra_info_fn=_extract_model_training_method)
    def validate(
        self, method=ValidateMethod.INVALID_RECORDS, model=None, eval_task=None, **kwargs
    ) -> "DatasetLoader":
        """
        Validate the dataset using the specified method.

        Example:
            loader.validate(
                method=ValidateMethod.INVALID_RECORDS,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
            )

        Args:
            method: The validation method (default: ValidateMethod.INVALID_RECORDS).
                Also accepts a TrainingMethod enum for backward compatibility (deprecated).
                ValidateMethod.SCHEMA is deprecated; use ValidateMethod.INVALID_RECORDS instead.
            model: The Model. Can be passed positionally for backward compatibility.
            eval_task: Optional evaluation task. Can be passed positionally for backward compatibility.
            **kwargs: Method-specific arguments passed to the operation.
                For ValidateMethod.INVALID_RECORDS (or ValidateMethod.SCHEMA):
                    training_method (TrainingMethod): Required. The training method.
                    model (Model): Required. The target model.
                    eval_task (EvaluationTask): Optional. Required for EVALUATION method.

        Returns:
            self (for method chaining)
        """
        # Handle deprecated usage: validate(TrainingMethod.X, Model.Y) positionally
        _deprecated_positional = isinstance(method, TrainingMethod)
        if _deprecated_positional:
            logger.warning(
                "validate(method=TrainingMethod) is deprecated. "
                "Use training_method= instead. "
                "Example: loader.validate(method=ValidateMethod.INVALID_RECORDS, "
                "training_method=TrainingMethod.SFT_LORA, model=...)"
            )
            kwargs.setdefault("training_method", method)
            method = ValidateMethod.INVALID_RECORDS

        # Deprecation warning for ValidateMethod.SCHEMA
        if method == ValidateMethod.SCHEMA:
            logger.warning(
                "ValidateMethod.SCHEMA is deprecated. "
                "Use ValidateMethod.INVALID_RECORDS instead. "
                "Example: loader.validate(method=ValidateMethod.INVALID_RECORDS, "
                "training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)"
            )

        # Handle model passed as explicit param — only warn for old positional API
        if isinstance(model, Model):
            if _deprecated_positional:
                logger.warning(
                    "validate(TrainingMethod, Model) is deprecated. "
                    "Use training_method= and model= instead. "
                    "Example: loader.validate(method=ValidateMethod.INVALID_RECORDS, "
                    "training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)"
                )
            kwargs.setdefault("model", model)

        # Handle eval_task passed as explicit param — only warn for old positional API
        if isinstance(eval_task, EvaluationTask):
            if _deprecated_positional:
                logger.warning(
                    "Passing eval_task positionally is deprecated. "
                    "Use eval_task= as a keyword argument instead."
                )
            kwargs.setdefault("eval_task", eval_task)

        self._flush_pending()
        op = get_validate_operation(method)
        op.execute(self, **kwargs)
        return self

    @_telemetry_emitter(Feature.DATA_PREP, "filter", extra_info_fn=_extract_filter_method)
    def filter(
        self, method: FilterMethod = FilterMethod.DEFAULT_TEXT_FILTER, **kwargs
    ) -> "DatasetLoader":
        """Queue a data filtering operation (lazy — runs on ``execute()``).

        No AWS calls or pipeline execution happens until ``execute()``
        is called. Multiple ``filter()`` calls can be chained to build
        a multi-step pipeline.

        Example::

            # load() is required before filter():
            loader.load("s3://bucket/raw/data.jsonl")
            loader.filter(
                method=FilterMethod.DEFAULT_TEXT_FILTER,
            ).filter(
                method=FilterMethod.EXACT_DEDUP,
            ).execute()
            # output paths are auto-generated:
            #   s3://bucket/raw/data/2026-04-20_14-30-22/default_text_filter/
            #   s3://bucket/raw/data/2026-04-20_14-30-22/exact_dedup_filter/

        Args:
            method: The filter method (default: ``FilterMethod.DEFAULT_TEXT_FILTER``).
            **kwargs: Method-specific arguments passed to the operation.
                Common kwargs:
                    output_path (str): S3 URI for filtered output. When omitted,
                        derived as ``<parent>/<input_stem>/<session>/<method>/``
                        where ``input_stem`` is the load filename without
                        extension and ``session`` is a UTC timestamp
                        (``YYYY-MM-DD_HH-MM-SS``). Required when loading
                        from a local file.
                    input_format (str): ``"parquet"`` or ``"jsonl"``.
                    output_format (str): ``"parquet"`` or ``"jsonl"``.
                    text_field (str): Column name containing text.
                    runtime_manager: Optional RuntimeManager instance.

        Returns:
            self (for method chaining)
        """
        # Pre-transform filters (DEFAULT_TEXT_FILTER, EXACT_DEDUP) operate on
        # raw flat-text data and must be called before transform().
        # Post-transform filters (INVALID_RECORDS) inspect the chat template
        # and must be called after transform().

        if self._has_transforms and method not in _POST_TRANSFORM_FILTERS:
            raise ValueError(
                "filter() must be called before transform() for pre-transform filters "
                f"(got {method.value}). "
                "Filters like DEFAULT_TEXT_FILTER and EXACT_DEDUP operate on raw data "
                "with a flat text field, but transform() converts data to a nested "
                "format (e.g. Converse) that these filters cannot process. "
                "Use: loader.load(...).filter(...).transform(...) instead."
            )
        if not self._has_transforms and method in _POST_TRANSFORM_FILTERS:
            logger.warning(
                "%s filter expects data in the correct schema format, not raw text. "
                "If you loaded pre-transformed data, verify it is in the "
                "expected schema format. Otherwise, call transform() before "
                "filter(). Samples that fail schema validation will be removed.",
                method.value,
            )
        self._pending_operations.append(("filter", method, kwargs))
        logger.info("Queued filter: %s", method.value)
        return self

    def _flush_pending(self) -> None:
        """Execute pending operations if any exist, then materialize local data.

        Called by terminal operations (show, validate, save, split) before they
        read loader.dataset(). After flushing, we materialize the dataset when
        the final state is LOCAL to prevent subsequent terminal operations from
        re-executing the entire lazy generator chain (load → filter → transform).

        Without materialization, every call to loader.dataset() replays the full
        chain — causing duplicate log messages and redundant computation when
        multiple terminal ops are called sequentially (e.g. show() then validate()).

        We only materialize for LOCAL state because:
        - Remote (S3) state: loader.dataset is already set to a fresh reader
          that loads from the S3 output path — no chain re-execution issue.
        - Local state: loader.dataset is a nested generator factory where each
          layer wraps the previous one. Calling it replays everything.

        Trade-off: This holds the full dataset in memory as a list.
        """
        if self._pending_operations:
            self.execute()

        # Materialize after flush if the data is local/in-memory
        if (
            self._last_state is not None
            and self._last_state.location == DataLocation.LOCAL
            and not self._is_materialized
        ):
            self._materialize_dataset()
            self._is_materialized = True

    @_telemetry_emitter(Feature.DATA_PREP, "execute")
    def execute(self) -> "DatasetLoader":
        """Execute all pending operations (filters and transforms) in order.

        Requires ``load()`` to have been called first for remote filter
        operations that need an S3 path. Runs each queued step
        sequentially, threading ``DataState`` between operations, then
        clears the queue.

        This enables minimal chaining::

            loader.load("s3://bucket/raw/data.jsonl")
            loader.filter(
                method=FilterMethod.DEFAULT_TEXT_FILTER,
            ).filter(
                method=FilterMethod.EXACT_DEDUP,
            ).transform(
                method=TransformMethod.SCHEMA,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
            ).execute()

        After completion the loader's dataset points to the output of
        the last operation.

        Returns:
            self (for method chaining)
        """
        if not self._pending_operations:
            return self

        self._is_materialized = False
        state = self._last_state or DataState.from_loader(self)
        resolver = OutputPathResolver(self._load_path or state.path)

        for op_type, method, kwargs in self._pending_operations:
            op = self._get_operation(op_type, method)
            kwargs["state"] = state
            if "output_path" not in kwargs:
                kwargs["output_path"] = resolver.resolve(method)
            result = op.execute(self, **kwargs)
            state = result.output_state

        self._pending_operations.clear()
        # Persist final pipeline state so subsequent terminal operations
        # (show, validate, save, split) start from where this pipeline
        # ended rather than replaying from the original load path.
        self._last_state = state

        return self

    def _materialize_dataset(self) -> None:
        """Eagerly evaluate the dataset generator chain and cache the result.

        Replaces loader.dataset with a simple factory that returns iter(list),
        so repeated calls no longer re-execute the transform/filter chain.
        Only called from _flush_pending() for LOCAL state — never from execute()
        directly, preserving lazy streaming for manual iteration.
        """
        data = list(self.dataset())

        def _cached_dataset(_m: list = data) -> Iterator[Dict]:
            return iter(_m)

        self.dataset = _cached_dataset

    def _get_operation(self, op_type: str, method: Any) -> Any:
        """Return the operation instance for the given type and method."""
        if op_type == "transform":
            return get_transform_operation(method)
        return get_filter_operation(method)


    @_telemetry_emitter(Feature.DATA_PREP, "save")
    def save(self, save_path: str) -> str:
        """
        Save the dataset to a local or S3 path.

        Args:
            save_path: Path where to save the file (.json or .jsonl).

        Returns: Path where the file was saved.
        """
        self._flush_pending()
        return self._save_op.execute(self, save_path=save_path)

    def save_data(self, save_path: str) -> str:
        """Deprecated: Use save() instead."""
        logger.warning("save_data() is deprecated, use save() instead.")
        return self.save(save_path)


# === DATASET LOADER CLASSES ===
class JSONLDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".jsonl"}
    _FORMAT = "jsonl"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single JSONL file."""

        def jsonl_generator():
            """Generator that yields records from JSONL file line by line."""
            try:
                for line in load_file_content(
                    file_path=path, extension=".jsonl", encoding="utf-8-sig"
                ):
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            preview = line[:120] + ("..." if len(line) > 120 else "")
                            logger.warning("Skipping malformed JSON line in %s: %s", path, preview)
            except Exception as e:
                raise DataPrepError(f"Error loading JSONL file {path}: {e}") from e

        return jsonl_generator


class JSONDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".json"}
    _FORMAT = "json"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single JSON file."""

        def json_generator():
            """Generator that yields records from JSON file."""
            try:
                lines = list(load_file_content(file_path=path, extension=".json", encoding="utf-8"))
                content = "\n".join(lines)
                data = json.loads(content)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data
            except Exception as e:
                raise DataPrepError(f"Error loading JSON file {path}: {e}") from e

        return json_generator


class CSVDatasetLoader(DatasetLoader):
    _EXTENSIONS = {".csv"}
    _FORMAT = "csv"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single CSV file."""

        def csv_generator():
            """Generator that yields records from CSV file row by row."""
            try:
                lines = load_file_content(file_path=path, extension=".csv", encoding="utf-8-sig")
                reader = csv.DictReader(lines)
                yield from reader
            except UnicodeError:
                lines = load_file_content(file_path=path, extension=".csv", encoding="utf-8")
                reader = csv.DictReader(lines)
                yield from reader
            except Exception as e:
                raise DataPrepError(f"Error loading CSV file {path}: {e}") from e

        return csv_generator


class ParquetDatasetLoader(DatasetLoader):
    """Load Parquet files lazily using PyArrow batch iteration."""

    _EXTENSIONS = {".parquet", ".pq"}
    _FORMAT = "parquet"

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single Parquet file."""

        def parquet_generator():
            try:
                if path.startswith("s3://"):
                    fs = pafs.S3FileSystem()
                    s3_path = path[len("s3://") :]
                    pf = pq.ParquetFile(fs.open_input_file(s3_path))
                else:
                    pf = pq.ParquetFile(path)

                for batch in pf.iter_batches():
                    yield from batch.to_pylist()
            except Exception as e:
                raise DataPrepError(f"Error loading Parquet file {path}: {e}") from e

        return parquet_generator


class ArrowDatasetLoader(DatasetLoader):
    """Load Arrow IPC/Feather files lazily using batch iteration."""

    _EXTENSIONS = {".arrow", ".feather", ".ipc"}
    _FORMAT = "arrow"

    @staticmethod
    def _iter_arrow_batches(source) -> Iterator:
        """Yield records from an Arrow IPC source, trying Stream then File format.

        Arrow IPC has two layouts: Stream (sequential, no random access) and
        File (has footer, supports random access). A .arrow/.ipc file could
        be either. We try Stream first (more memory-efficient), then fall
        back to File format.
        """
        try:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                yield from batch.to_pylist()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            # Not a stream — try IPC File format
            if hasattr(source, "seek"):
                source.seek(0)
            reader = pa.ipc.open_file(source)
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()

    def _make_single_file_generator(self, path: str) -> Callable[[], Iterator[Dict]]:
        """Return a generator factory for a single Arrow IPC or Feather file."""

        def arrow_generator():
            try:
                if path.startswith("s3://"):
                    fs = pafs.S3FileSystem()
                    s3_path = path[len("s3://") :]
                    source = fs.open_input_file(s3_path)
                else:
                    source = path

                if path.endswith(".feather"):
                    table = pa.feather.read_table(source)
                    for batch in table.to_batches():
                        yield from batch.to_pylist()
                else:
                    yield from ArrowDatasetLoader._iter_arrow_batches(source)
            except Exception as e:
                raise DataPrepError(f"Error loading Arrow file {path}: {e}") from e

        return arrow_generator
