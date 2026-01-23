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
import random
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import boto3
import jsonschema
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.dataset.dataset_validator import (
    CPTDatasetValidator,
    EvalDatasetValidator,
    RFTDatasetValidator,
    SFTDatasetValidator,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask

from ..util.logging import logger
from ..util.recipe import load_file_content
from .dataset_transformers import DatasetTransformer
from .transform_format_schema import TRANSFORM_CONFIG


class DataPrepError(Exception):
    """Custom exception for data preparation errors."""

    pass


class DatasetLoader(ABC):
    """
    This abstract class defines the required features across the child classes.

    Args:
        **column_mappings: Keyword arguments where the key is the standard column name,
                            and the value is the actual column name in your dataset.
                            Example: question="input" where "question" is the default name
                                    of the column, and "input" is what you named the column.

    TODO: When we make the README, add the expected columns for certain methods.
    """

    def __init__(self, **column_mappings):
        self.column_mappings = column_mappings
        self.raw_dataset: List[Dict] = []
        self.transformed_dataset: List[Dict] = []
        self.transformer = DatasetTransformer()
        pass

    @abstractmethod
    def load(self, path: str) -> "DatasetLoader":
        """
        Load dataset as its raw format without converting to converse.

        Args:
            path: Dataset path

        Returns: DatasetLoader
        """
        pass

    def show(self, n: int = 10) -> None:
        """
        Display the first n rows of the dataset. Defaults to show the transformed dataset if available.
        Otherwise, it will show the raw dataset.

        Args:
            n: Number of rows to display (default: 10)
        """
        if not self.raw_dataset:
            logger.info("Dataset is empty. Call load() method to load data first")
            return
        if self.transformed_dataset:
            logger.info("Showing transformed dataset:")
            for i, row in enumerate(self.transformed_dataset[:n]):
                logger.info(f"\nRow {i}: {json.dumps(row)}")
        else:
            logger.info("Showing raw dataset:")
            for i, row in enumerate(self.raw_dataset[:n]):
                logger.info(f"\nRow {i}: {json.dumps(row)}")

    def split_data(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple["DatasetLoader", "DatasetLoader", "DatasetLoader"]:
        """
        Split data into train, validation, and test DatasetLoader objects

        Args:
            train_ratio: The % of data to train on
            val_ratio: The % of data for evaluation
            test_ratio: The % of data to test on
            seed: Value used for random generation.

        Returns: Tuple of three DatasetLoader objects (train, val, test)
        """
        # Checks if transformed_dataset has data. If so, this is what we'll split. If not, split raw_data.
        if self.transformed_dataset:
            dataset = self.transformed_dataset
        elif self.raw_dataset:
            dataset = self.raw_dataset
        else:
            raise DataPrepError("Dataset is empty. Call load() method first")

        # Shuffle data
        random.seed(seed)
        shuffled_data = dataset.copy()
        random.shuffle(shuffled_data)

        # Assign default ratio values if none are provided, else ask for all three to be provided.
        if (train_ratio, val_ratio, test_ratio) == (None, None, None):
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
        if train_ratio is None or val_ratio is None or test_ratio is None:
            raise DataPrepError(
                f"Please provide three values for split_data: train_ratio, val_ratio, and test_ratio."
                f"You provided: (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio})"
            )

        if len(shuffled_data) < 10:
            logger.info(
                "The provided dataset is small. Data will be split, but consider adding more data for better results."
            )

        n_total = len(shuffled_data)
        n_train = max(1, round(n_total * train_ratio)) if train_ratio > 0 else 0
        remaining = n_total - n_train

        # Checks if any of the ratios are zero so no data is included under them.
        if val_ratio == 0:
            n_val = 0
            n_test = remaining
        elif test_ratio == 0:
            n_val = remaining
            n_test = 0
        else:
            n_val = max(1, round(n_total * val_ratio)) if val_ratio > 0 else 0
            n_test = remaining - n_val

        # Ensure we haven't exceeded the total length
        if n_train + n_val + n_test > n_total:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        # Validate the ratios
        if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
            raise DataPrepError(
                "Calculated ratio is negative. Provided ratios sum to > 1.0"
            )
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise DataPrepError(
                f"Split ratios must sum to 1.0. Current Ratios: "
                f"(Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio} -> Total: {abs(train_ratio + val_ratio + test_ratio)})"
            )

        # Split the data into train/val/test
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train : n_train + n_val]
        test_data = shuffled_data[n_train + n_val :]

        # Create new DatasetLoaders for each split.
        train_loader = self.__class__(**self.column_mappings)
        train_loader.raw_dataset = train_data

        val_loader = self.__class__(**self.column_mappings)
        val_loader.raw_dataset = val_data

        test_loader = self.__class__(**self.column_mappings)
        test_loader.raw_dataset = test_data

        logger.info(
            f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
        )
        return train_loader, val_loader, test_loader

    def get_transformer_function(self, method_name: str):
        """
        Map transformer method name to actual function.
        """
        transformer_map = {
            "convert_to_converse_sft_nova_one": DatasetTransformer.convert_to_converse_sft_nova_one,
            "convert_to_converse_sft_nova_two": DatasetTransformer.convert_to_converse_sft_nova_two,
            "convert_openai_to_converse_sft_nova_one": DatasetTransformer.convert_openai_to_converse_sft_nova_one,
            "convert_openai_to_converse_sft_nova_two": DatasetTransformer.convert_openai_to_converse_sft_nova_two,
            "convert_to_openai_rft": DatasetTransformer.convert_to_openai_rft,
            "convert_to_evaluation": DatasetTransformer.convert_to_evaluation,
            "convert_to_cpt": DatasetTransformer.convert_to_cpt,
        }

        if method_name not in transformer_map:
            raise ValueError(f"Unknown transformer method: {method_name}")

        return transformer_map[method_name]

    def validate_against_schema(self, schema: dict) -> bool:
        """
        Validate all records in raw_dataset against a schema.

        Args:
            schema: JSON schema to validate against

        Returns:
            True if all records are valid, False otherwise
        """
        try:
            for row in self.raw_dataset:
                jsonschema.validate(instance=row, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False

    def transform(self, method: TrainingMethod, model: Model) -> "DatasetLoader":
        """
        Transform the dataset to the required format for the training method and model.

        Args:
            method: The Training Method that the user wants to run (e.g. SFT_LORA)
            model: The Model (and version) that the user is planning to use (e.g. NOVA_PRO, NOVA_LITE_2)

        Returns:
            self: Updates the value of the transformed_dataset if a change is made.
        """
        if not self.raw_dataset:
            logger.info("Dataset is empty. Call load() method to load data first")
            return self

        # Find the right schema for the training method and model combination.
        transform_config: Optional[Dict[str, Any]] = None
        for (methods, models), config in TRANSFORM_CONFIG.items():
            if (method in methods) and (
                models is None
                or models == model
                or (isinstance(models, tuple) and model in models)
            ):
                transform_config = cast(Dict[str, Any], config)
                break

        if not transform_config:
            raise ValueError(
                f"The combination of training method {method} and model {model} is not yet supported.\n"
                f"Note: RFT is only supported on Nova 2.0."
            )

        target_schema = transform_config["schema"]

        # Step 1: Check if already in target format
        if self.validate_against_schema(target_schema):
            logger.info(transform_config["success_msg"])
            self.transformed_dataset = self.raw_dataset
            return self

        # Step 2: Try each transformer in order (based on source format detection)
        transformers: List[Dict[str, Any]] = transform_config.get("transformers", [])

        for transformer_info in transformers:
            source_schema = transformer_info.get("source_schema")
            method_name = transformer_info["method"]
            transform_msg = transformer_info["msg"]

            # If source_schema is None, this is the generic/fallback transformer
            if source_schema is None:
                logger.info(transform_msg)
                transformer_func = self.get_transformer_function(method_name)
                try:
                    self.transformed_dataset = [
                        transformer_func(rec, self.column_mappings)
                        for rec in self.raw_dataset
                    ]
                    return self
                except Exception as e:
                    raise DataPrepError(
                        f"Error transforming dataset using generic format: {str(e)}\n"
                        f"Make sure to add the correct column mappings when initializing DatasetLoader."
                    )

            # Validate against the source schema
            if self.validate_against_schema(source_schema):
                logger.info(transform_msg)
                transformer_func = self.get_transformer_function(method_name)
                try:
                    self.transformed_dataset = [
                        transformer_func(rec, self.column_mappings)
                        for rec in self.raw_dataset
                    ]
                    return self
                except Exception as e:
                    raise DataPrepError(
                        f"Error transforming dataset from detected format: {str(e)}"
                    )

        # This shouldn't happen if config is set up correctly (should have a fallback)
        raise DataPrepError(
            f"Unable to transform dataset. No suitable transformer found for the given data format."
        )

    def validate(
        self,
        method: TrainingMethod,
        model: Model,
        eval_task: Optional[EvaluationTask] = None,
    ) -> "DatasetLoader":
        # Default to the transformed_dataset if one exists, otherwise use the raw_dataset.
        if self.transformed_dataset:
            dataset = self.transformed_dataset
        elif self.raw_dataset:
            dataset = self.raw_dataset
        else:
            logger.info("Dataset is empty. Call load() method to load data first")
            return self

        # Select the right validator for the provided method and validate.
        if method in (TrainingMethod.SFT_LORA, TrainingMethod.SFT_FULL):
            # Handles SFT 1.0 and 2.0
            sft_validator = SFTDatasetValidator(dataset, model)
            sft_validator.validate(dataset, model)
        elif method == TrainingMethod.EVALUATION:
            # Handles BYOD Eval datasets, NOT LLM-as-judge.
            eval_validator = EvalDatasetValidator(dataset, model, eval_task)
            eval_validator.validate(dataset, model)
        elif method in (TrainingMethod.RFT_FULL, TrainingMethod.RFT_LORA):
            rft_validator = RFTDatasetValidator(dataset, model)
            rft_validator.validate(dataset, model)
        elif method == TrainingMethod.CPT:
            cpt_validator = CPTDatasetValidator(dataset, model)
            cpt_validator.validate(dataset, model)
        else:
            logger.info(
                "Skipping validation. Validation isn't available for that model/method combo right now."
            )
        return self

    def save_data(self, save_path: str) -> str:
        """
        Saves the dataset to a local or S3 directory.

        Args:
            save_path (str): Path where to save the file

        Returns: Path where the file was saved
        """
        # Check if the dataset is empty.
        if self.transformed_dataset:
            dataset = self.transformed_dataset
        elif self.raw_dataset:
            dataset = self.raw_dataset
        else:
            logger.warning("Warning: Dataset is empty. An empty dataset will be saved.")
            dataset = []

        try:
            # Prepare the data based on format
            if save_path.endswith(".jsonl"):
                data = "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in dataset
                )
            elif save_path.endswith(".json"):
                data = json.dumps(dataset, indent=2, ensure_ascii=False)
            else:
                raise DataPrepError(
                    f"Unsupported format: {format}. Use 'json' or 'jsonl'"
                )

            # Try to save to S3 first
            if save_path.startswith("s3://"):
                s3_path = save_path[5:]  # Remove 's3://'
                bucket, key = s3_path.split("/", 1)

                s3_client = boto3.client("s3")
                try:
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=data.encode("utf-8"),
                        ContentType="application/json",
                    )
                except ClientError as e:
                    raise DataPrepError(f"Failed to upload to S3: {e}")
            # Otherwise save to local file
            else:
                local_path = Path(save_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(data, encoding="utf-8")
            logger.info(f"Dataset saved successfully to {save_path}")
            return save_path

        except Exception as e:
            raise DataPrepError(f"Error saving dataset: {str(e)}")


# === DATASET LOADER CLASSES ===
class JSONLDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        self.raw_dataset = []
        try:
            f = load_file_content(
                file_path=path, extension=".jsonl", encoding="utf-8-sig"
            )
            for line in f.splitlines():
                line = line.strip()
                if line:
                    try:
                        self.raw_dataset.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {line}. Error: {e}")
        except Exception as e:
            logger.error(f"Error loading JSONL file {path}: {str(e)}")
        return self


class JSONDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        self.raw_dataset = []
        try:
            f = load_file_content(file_path=path, extension=".json", encoding="utf-8")
            data = json.loads(f)
            if isinstance(data, list):
                self.raw_dataset = data
            else:
                self.raw_dataset = [data]
        except Exception as e:
            logger.error(f"Error loading JSON file {path}: {str(e)}")
        return self


class CSVDatasetLoader(DatasetLoader):
    # Loads the dataset from CSV format and stores it as JSONL.
    def load(self, path: str) -> "DatasetLoader":
        self.raw_dataset = []
        # Try removing BOM from CSV if present.
        try:
            f = load_file_content(
                file_path=path, extension=".csv", encoding="utf-8-sig"
            )
            reader = csv.DictReader(StringIO(f))
            for row in reader:
                self.raw_dataset.append(row)
        except UnicodeError:
            # If that fails, try regular utf-8
            f = load_file_content(file_path=path, extension=".csv", encoding="utf-8")
            reader = csv.DictReader(StringIO(f))
            for row in reader:
                self.raw_dataset.append(row)
        return self
