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
from typing import Dict, List, Optional, Tuple

import fsspec
import jsonschema

from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod

from ..util.logging import logger
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
        transform_config = None
        for (methods, models), config in TRANSFORM_CONFIG.items():
            if (method in methods) and (
                models is None
                or models == model
                or (isinstance(models, tuple) and model in models)
            ):
                transform_config = config
                break

        if not transform_config:
            raise ValueError(
                f"The combination of training method {method} and model {model} is not yet supported.\n"
                f"Note: RFT is only supported on Nova 2.0."
            )

        # Try to validate the dataset against the schema for the selected training method.
        try:
            [
                jsonschema.validate(instance=row, schema=transform_config["schema"])
                for row in self.raw_dataset
            ]
            logger.info(transform_config["success_msg"])
            self.transformed_dataset = self.raw_dataset

        # Attempt to transform the dataset to the required format.
        except jsonschema.exceptions.ValidationError as validate_e:
            logger.info(transform_config["transform_msg"])
            method_name = transform_config["transformer_method"]

            # Map the right transformation function to the method.
            if method_name == "convert_to_converse_sft_nova_one":
                transformer_func = DatasetTransformer.convert_to_converse_sft_nova_one
            elif method_name == "convert_to_converse_sft_nova_two":
                transformer_func = DatasetTransformer.convert_to_converse_sft_nova_two
            elif method_name == "convert_to_openai_rft":
                transformer_func = DatasetTransformer.convert_to_openai_rft
            elif method_name == "convert_to_evaluation":
                transformer_func = DatasetTransformer.convert_to_evaluation
            else:
                raise ValueError(f"Unknown transformer method: {method_name}")

            try:
                self.transformed_dataset = [
                    transformer_func(rec, self.column_mappings)
                    for rec in self.raw_dataset
                ]
            except Exception as transform_e:
                raise DataPrepError(
                    f"These errors were caught when transforming the dataset: \n"
                    f"- {validate_e.message}. Check: {validate_e.json_path}\n"
                    f"- {transform_e}"
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
            logger.warn("Warning: Dataset is empty. An empty dataset will be saved.")
            dataset = []

        try:
            # fsppec handles saving the data file to either s3 or a local directory.
            with fsspec.open(save_path, "w", encoding="utf-8") as f:
                if ".jsonl" in save_path:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                elif ".json" in save_path:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                else:
                    raise DataPrepError(
                        f"Unsupported format: {format}. Use 'json' or 'jsonl'"
                    )
            logger.info(f"Dataset saved successfully to {save_path}")
            return save_path

        except Exception as e:
            raise DataPrepError(f"Error saving dataset: {str(e)}")


# === DATASET LOADER CLASSES ===
class JSONLDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        self.raw_dataset = []
        try:
            with fsspec.open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    try:
                        self.raw_dataset.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {line.strip()}. Error: {e}")
        except Exception as e:
            logger.error(f"Error loading JSONL file {path}: {str(e)}")
        return self


class JSONDatasetLoader(DatasetLoader):
    def load(self, path: str) -> "DatasetLoader":
        self.raw_dataset = []
        try:
            with fsspec.open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
            with fsspec.open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.raw_dataset.append(row)
        except UnicodeError:
            # If that fails, try regular utf-8
            with fsspec.open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.raw_dataset.append(row)
        return self
