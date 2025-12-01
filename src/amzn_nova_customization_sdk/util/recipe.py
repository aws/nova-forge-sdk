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
import io
import re
import shutil
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import boto3
import yaml
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.util.logging import logger

T = TypeVar("T")
DataclassLike = Union[Type[Any], Any]
S3_URI_REGEX = re.compile(r"^s3://([a-zA-Z0-9.\-_]+)/(.+)$")


class RecipeLoadError(Exception):
    """Custom exception for recipe loading errors."""

    pass


class RecipePath:
    """Container for recipe paths. Allows automatically deleting temporary recipe directories."""

    roots: List[str] = []

    def __init__(self, path: str, root: Optional[str] = None, temp: bool = False):
        self.path = path
        self.root = root
        self.temp = temp

        if temp and root is not None:
            self.roots.append(root)

    @staticmethod
    def delete_temp_dir(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            logger.warn(f"Failed to delete temporary directory {directory}\nError: {e}")

    def close(self):
        if self.temp:
            RecipePath.delete_temp_dir(self.root)

    def close_all(self):
        for path in RecipePath.roots:
            RecipePath.delete_temp_dir(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    """Parse S3 URI into (bucket, key) tuple, or None if the URI is invalid."""
    match = S3_URI_REGEX.match(uri)
    if not match:
        return None
    bucket, key = match.groups()
    return (bucket, key)


def _validate_extension(path: str, extension: str) -> None:
    """
    Validate that the given path has the required file extension.

    Args:
        path: File path or S3 URI
        extension: Extension (e.g., '.yaml')

    Raises:
        RecipeLoadError: If extension doesn't match
    """
    if not path.lower().endswith(extension.lower()):
        raise RecipeLoadError(f"File must have {extension} extension: {path}")


def _load_file_content(recipe_path: str) -> str:
    """
    Load file content from S3 or local filesystem.

    Args:
        recipe_path: Path to YAML file (either local path or S3 URI)

    Returns:
        File content as string

    Raises:
        RecipeLoadError: If file cannot be loaded
    """
    # Validate extension
    _validate_extension(recipe_path, ".yaml")

    # Try S3 first
    s3_parts = _parse_s3_uri(recipe_path)
    if s3_parts:
        bucket, key = s3_parts
        try:
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except ClientError as e:
            raise RecipeLoadError(f"Failed to load S3 file {recipe_path}: {e}")

    # Try local filesystem second
    try:
        path = Path(recipe_path)
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RecipeLoadError(f"File not found: {recipe_path}")
    except OSError as e:
        raise RecipeLoadError(f"Failed to read file {recipe_path}: {e}")


def merge_overrides_with_input_recipe(
    recipe_path: str,
    recipe_class: type,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load a recipe YAML file and merge it with overrides, prioritizing override values.

    Args:
        recipe_path: Path to YAML recipe file (local path or s3:// URI)
        recipe_class: The type of recipe dataclass that we are parsing (i.e. EvalRecipeConfig)
        overrides: Dictionary of override values

    Returns:
        Dictionary with merged override values
    """
    # Load YAML recipe
    yaml_str = _load_file_content(recipe_path)

    try:
        recipe_dict = yaml.safe_load(io.StringIO(yaml_str))
    except yaml.YAMLError as e:
        raise RecipeLoadError(f"Invalid YAML in {recipe_path}: {e}")

    if not isinstance(recipe_dict, dict):
        raise RecipeLoadError(
            f"YAML must be a dictionary, got {type(recipe_dict).__name__}"
        )

    # Recursively flatten the recipe_dict such that only leaf values are appended to overrides
    def add_leaf_values(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                add_leaf_values(value)
            elif (
                key not in overrides
            ):  # Values in "overrides" take priority over values within the input recipe
                overrides[key] = value

    add_leaf_values(recipe_dict)

    # Convert str values within "overrides" into Enum values (where applicable)
    enums: Dict[str, Any] = {
        k: v
        for k, v in get_all_type_hints(recipe_class).items()
        if isinstance(v, type) and issubclass(v, Enum)
    }
    errors: List[str] = []
    for key, value in overrides.items():
        if key in enums:
            enum_type = enums[key]
            try:
                overrides[key] = enum_type(value)
            except ValueError:
                errors.append(
                    f"Invalid override '{key}' with value '{value}'. "
                    f"Valid options are: {[e.value for e in enum_type]}"
                )
    if errors:
        error_msg = f"\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)

    return overrides


def resolve_overrides(
    recipe_class: type,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entry helper method for combining an input recipe file (if it exists) with overrides (if they exist).
    Fields within the input recipe will be considered "overrides" - relative to default values that would be generated if the SDK user doesn't provide any input recipe.
    Values that are explicitly provided in the "overrides" parameter will take priority over anything that is in the input recipe.

    Args:
        recipe_class: The type of recipe dataclass that we are parsing (i.e. EvalRecipeConfig)
        recipe_path: (optional) Path to YAML recipe file (local path or s3:// URI)
        overrides: (optional) Dictionary of override values

    Returns:
        Dictionary with merged override values
    """
    overrides = overrides or {}

    if recipe_path:
        if overrides:
            logger.info(
                f"Recipe provided at {recipe_path}. Applying override values into recipe, and ignoring other user input in favor of the recipe content."
            )
        else:
            logger.info(
                f"Recipe provided at {recipe_path}. Ignoring other user input in favor of the recipe content."
            )
        overrides = merge_overrides_with_input_recipe(
            recipe_path=recipe_path,
            recipe_class=recipe_class,
            overrides=overrides,
        )
    return overrides


def get_all_type_hints(object: DataclassLike) -> Dict[str, Any]:
    """
    Recursively gather type hints from a dataclass (class or instance)
    and all of its nested dataclass fields. Returns a flattened mapping.
    """
    cls = object if isinstance(object, type) else type(object)
    all_type_hints: Dict[str, Any] = {}

    class_types = get_type_hints(cls)
    for field in fields(cls):
        field_type = class_types.get(field.name)

        origin = get_origin(field_type)
        if origin is Union:
            args = [t for t in get_args(field_type) if t is not type(None)]
            for arg in args:
                if is_dataclass(arg):
                    nested_hints = get_all_type_hints(arg)
                    all_type_hints.update(nested_hints)
                    break
            else:
                all_type_hints[field.name] = field_type
        elif is_dataclass(field_type):
            nested_hints = get_all_type_hints(field_type)
            all_type_hints.update(nested_hints)
        else:
            all_type_hints[field.name] = field_type

    return all_type_hints


def get_all_key_names(object: DataclassLike) -> Set[str]:
    """
    Recursively gather all key names from a dataclass (class or instance)
    and its nested dataclasses. Returns a flattened set of key names.
    """
    cls = object if isinstance(object, type) else type(object)
    all_names: Set[str] = set()

    for field in fields(cls):
        field_type = field.type

        origin = get_origin(field_type)
        if origin is Union:
            args = [t for t in get_args(field_type) if t is not type(None)]
            for arg in args:
                if is_dataclass(arg):
                    all_names |= get_all_key_names(arg)
                    break
            else:
                all_names.add(field.name)
        elif is_dataclass(field_type):
            all_names |= get_all_key_names(field_type)
        else:
            all_names.add(field.name)

    return all_names
