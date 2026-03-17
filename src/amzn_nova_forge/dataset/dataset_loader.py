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

Functionality:
    1. Load data from various sources (local files, S3)
    2. Convert to converse and OpenAI conversation formats.
    3. Split a dataset into train/validation/test sets
    4. Save a generated file locally or to a s3 bucket in JSON/JSONL format.

Supported input formats:
    - Local CSV files with conversation columns
    - Local JSON/JSONL files
    - S3 JSON/JSONL files
"""

import csv
import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, Optional, Tuple

from amzn_nova_forge.model.model_enums import Model, TrainingMethod
from amzn_nova_forge.recipe.recipe_config import EvaluationTask

from ..util.logging import logger
from ..util.recipe import load_file_content
from .operations.save_operation import SaveOperation
from .operations.show_operation import ShowOperation
from .operations.split_operation import SplitOperation
from .operations.transform_operation import TransformMethod, get_transform_operation
from .operations.validate_operation import ValidateMethod, get_validate_operation


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

        # Operations
        self._show_op = ShowOperation()
        self._save_op = SaveOperation()
        self._split_op = SplitOperation()

    # --- Unified dataset accessor ---
    @property
    def dataset(self) -> Callable[[], Iterator[Dict]]:
        """The current dataset callable. Each operation reads/writes this."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Callable[[], Iterator[Dict]]) -> None:
        self._dataset = value

    @abstractmethod
    def load(self, path: str) -> "DatasetLoader":
        """
        Load dataset from a file path.

        Args:
            path: Local or S3 file path.

        Returns: self (for method chaining)
        """
        pass

    def show(self, n: int = 10) -> None:
        """
        Display the first n rows of the dataset.

        Args:
            n: Number of rows to display (default: 10)
        """
        self._show_op.execute(self, n=n)

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

        op = get_transform_operation(method)
        op.execute(self, **kwargs)
        return self

    def validate(
        self, method=ValidateMethod.SCHEMA, model=None, eval_task=None, **kwargs
    ) -> "DatasetLoader":
        """
        Validate the dataset using the specified method.

        Example:
            loader.validate(
                method=ValidateMethod.SCHEMA,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
            )

        Args:
            method: The validation method (default: ValidateMethod.SCHEMA).
                Also accepts a TrainingMethod enum for backward compatibility (deprecated).
            model: The Model. Can be passed positionally for backward compatibility.
            eval_task: Optional evaluation task. Can be passed positionally for backward compatibility.
            **kwargs: Method-specific arguments passed to the operation.
                For ValidateMethod.SCHEMA:
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
                "Example: loader.validate(method=ValidateMethod.SCHEMA, "
                "training_method=TrainingMethod.SFT_LORA, model=...)"
            )
            kwargs.setdefault("training_method", method)
            method = ValidateMethod.SCHEMA

        # Handle model passed as explicit param — only warn for old positional API
        if isinstance(model, Model):
            if _deprecated_positional:
                logger.warning(
                    "validate(TrainingMethod, Model) is deprecated. "
                    "Use training_method= and model= instead. "
                    "Example: loader.validate(method=ValidateMethod.SCHEMA, "
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

        op = get_validate_operation(method)
        op.execute(self, **kwargs)
        return self

    def save(self, save_path: str) -> str:
        """
        Save the dataset to a local or S3 path.

        Args:
            save_path: Path where to save the file (.json or .jsonl).

        Returns: Path where the file was saved.
        """
        return self._save_op.execute(self, save_path=save_path)

    def save_data(self, save_path: str) -> str:
        """Deprecated: Use save() instead."""
        logger.warning("save_data() is deprecated, use save() instead.")
        return self.save(save_path)


# === DATASET LOADER CLASSES ===
class JSONLDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        """Lazy load JSONL file - creates a generator function."""

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
                            logger.error(f"Error parsing line: {line}. Error: {e}")
            except Exception as e:
                logger.error(f"Error loading JSONL file {path}: {str(e)}")

        self.dataset = jsonl_generator
        return self


class JSONDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        """
        Load JSON file - creates a generator function.
        Note: JSON files must be fully parsed, so this loads the entire file into memory.
        For large datasets, prefer JSONL format which supports true streaming.
        """

        def json_generator():
            """Generator that yields records from JSON file."""
            try:
                lines = list(
                    load_file_content(
                        file_path=path, extension=".json", encoding="utf-8"
                    )
                )
                content = "\n".join(lines)
                data = json.loads(content)
                if isinstance(data, list):
                    yield from data
                else:
                    yield data
            except Exception as e:
                logger.error(f"Error loading JSON file {path}: {str(e)}")

        self.dataset = json_generator
        return self


class CSVDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        """
        Load CSV file - creates a generator function.
        Note: CSV parsing requires reading the header first, but rows are streamed lazily.
        """

        def csv_generator():
            """Generator that yields records from CSV file row by row."""
            try:
                lines = load_file_content(
                    file_path=path, extension=".csv", encoding="utf-8-sig"
                )
                reader = csv.DictReader(lines)
                yield from reader
            except UnicodeError:
                lines = load_file_content(
                    file_path=path, extension=".csv", encoding="utf-8"
                )
                reader = csv.DictReader(lines)
                yield from reader
            except Exception as e:
                logger.error(f"Error loading CSV file {path}: {str(e)}")

        self.dataset = csv_generator
        return self
