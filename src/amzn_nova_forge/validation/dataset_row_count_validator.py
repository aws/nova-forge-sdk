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
"""Shared dataset row-count validation logic.

Used by both:
- ``BaseDatasetValidator._validate_row_counts`` (loader.validate path)
- ``Validator._validate_dataset_row_counts`` (train/evaluate pre-flight path)

Recipe origin:
    The ``recipe`` dict passed to ``validate_row_counts`` originates from HubContent.
    ``get_hub_recipe_metadata`` calls DescribeHubContent on SageMakerPublicHub, returning
    S3 URIs for recipe templates. ``download_templates_from_s3`` fetches and parses them,
    ``RecipeBuilder._build_final_recipe`` applies user overrides, and the resulting dict
    flows through ``Validator.validate`` → ``_validate_dataset_row_counts`` → here.
    ``min_rows_recipe_field`` checks resolve values from this HubContent-derived recipe.
"""

from typing import Any, Dict, List, Optional

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    DATASET_CHECKS,
    DatasetCheckEntry,
)
from amzn_nova_forge.util.logging import logger

CONFIG_TO_INDICATE_PRE_TRAINING_CHECK = "pre_training_only"


def _get_recipe_value(data: Dict[str, Any], key: str) -> Any:
    """Recursively search a nested dict for *key* and return its value.

    Currently only used for ``global_batch_size`` which appears once in
    recipes (under ``training_config``).  If a future recipe contains the
    same key at multiple nesting levels, this might not work as expected
    and will need updating.
    """
    for k, v in data.items():
        if k == key:
            return v
        if isinstance(v, dict):
            try:
                return _get_recipe_value(v, key)
            except KeyError:
                continue
    raise KeyError(key)


def get_applicable_row_count_checks(
    method: TrainingMethod, platform: Platform, model: Model
) -> List[DatasetCheckEntry]:
    """Return DATASET_CHECKS entries of type ``row_count`` matching the given triple."""
    return [
        c
        for c in DATASET_CHECKS
        if c.get("type") == "row_count"
        and method in c["training_methods"]
        and platform in c["platforms"]
        and model in c["models"]
    ]


def validate_row_counts(
    num_samples: int,
    model: Model,
    platform: Platform,
    training_method: TrainingMethod,
    *,
    recipe: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> None:
    """Check *num_samples* against applicable row-count rules.

    When *errors* is ``None`` (loader.validate path), violations raise
    ``ValueError`` immediately.  When *errors* is a list (Validator.validate
    path), messages are appended instead.

    *recipe* is the fully-built recipe dict (train path); when provided,
    ``min_rows_recipe_field`` checks resolve the value from the recipe.
    When *recipe* is ``None`` (loader path), ``min_rows_recipe_field``
    checks are silently skipped.
    """
    applicable = get_applicable_row_count_checks(training_method, platform, model)
    if not applicable:
        return

    def _report(msg: str) -> None:
        if errors is not None:
            errors.append(msg)
        else:
            raise ValueError(msg)

    ctx = f"{platform.value}/{training_method.value}/{model.value}"

    for check in applicable:
        if check.get(CONFIG_TO_INDICATE_PRE_TRAINING_CHECK) and errors is None:
            continue

        max_rows = check.get("max_rows")
        if max_rows is not None and num_samples > max_rows:
            _report(
                f"Dataset has {num_samples} samples, which exceeds the "
                f"maximum of {max_rows} for {ctx}."
            )

        min_rows = check.get("min_rows")
        if min_rows is not None and num_samples < min_rows:
            _report(
                f"Dataset has {num_samples} samples, which is below the "
                f"minimum of {min_rows} for {ctx}."
            )

        recipe_field = check.get("min_rows_recipe_field")
        if recipe_field and recipe is not None:
            try:
                effective_min = _get_recipe_value(recipe, recipe_field)
                if isinstance(effective_min, int) and num_samples < effective_min:
                    _report(
                        f"Dataset has {num_samples} samples, which is below "
                        f"the minimum of {effective_min} ({recipe_field}) for {ctx}."
                    )
            except KeyError:
                logger.debug(
                    "Recipe field '%s' not found; skipping min-rows check.",
                    recipe_field,
                )


def count_s3_dataset_rows(data_s3_path: str, region: str) -> int:
    """Count non-empty lines in an S3 JSONL file by streaming."""
    # Lazy imports: only this function needs boto3 and _parse_s3_uri.
    # Keeping them here avoids making boto3 a hard dependency for modules
    # that import validate_row_counts (e.g. BaseDatasetValidator in the
    # local loader.validate path) and breaks the circular import chain:
    # dataset_validator → dataset_row_count_validator → util.recipe → util.sagemaker → manager.runtime_manager
    import boto3

    from amzn_nova_forge.util.recipe import _parse_s3_uri

    if not region:
        raise ValueError("A valid AWS region string is required.")

    s3_parts = _parse_s3_uri(data_s3_path)
    if not s3_parts:
        raise ValueError(
            f"Invalid S3 path: {data_s3_path}. Expected format: s3://bucket-name/path/to/data.jsonl"
        )
    bucket, key = s3_parts
    s3_client = boto3.client("s3", region_name=region)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    count = 0
    for line in response["Body"].iter_lines():
        if line.strip():
            count += 1
    return count
