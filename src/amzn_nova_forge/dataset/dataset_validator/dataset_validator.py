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
"""
Base validator module for dataset validation across different training methods.

This module provides the abstract base class that all specific validators
must inherit from to ensure consistent validation interface.
"""

import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterator, List, Optional

import boto3
import filetype
from pydantic import BaseModel, ValidationError, ValidationInfo

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    DATASET_CHECKS,
    DatasetCheckEntry,
)
from amzn_nova_forge.validation.dataset_row_count_validator import (
    validate_row_counts,
)

from ...util.iterator_utils import peek
from ...util.logging import logger


class InfrastructureError(Exception):
    """Raised when a validation check fails due to infrastructure issues (S3, network, etc.),
    not due to invalid data."""

    pass


class BaseDatasetValidator(ABC):
    """
    Abstract base class for dataset validators.

    All training method-specific validators should inherit
    from this class and implement the validate() method.
    """

    def __init__(self):
        """
        Initialize the base dataset validator.
        """
        self.num_samples = 0

    @abstractmethod
    def get_sample_model(self) -> type[BaseModel]:
        """Return the Pydantic model class used to validate individual samples."""
        pass

    @abstractmethod
    def get_success_message(self) -> str:
        """Return the success message for this validator type."""
        pass

    @abstractmethod
    def get_optional_fields(self) -> List[str]:
        """Return a list of optional fields for this validator type."""
        pass

    def validate(
        self,
        dataset: Iterator[Dict],
        model: Model,
        **kwargs: Any,
    ) -> None:
        """
        Validates the entire conversation dataset against Nova format requirements.

        Args:
            dataset: Iterator of dataset samples.
            model: The target Nova model.
            **kwargs: Optional row-count check params:
                platform (Platform): Platform for row-count checks.
                training_method (TrainingMethod): Training method for row-count checks.
        """
        error_message = ""
        failed_samples_id_list = []

        # Track optional field consistency with minimal memory
        optional_fields = self.get_optional_fields()
        field_consistency: Dict[str, bool | None] = {field: None for field in optional_fields}
        first_sample_with_field: Dict[str, int] = {}

        # Checks the first line of the dataset to quickly validate that required fields are there.
        first_item, dataset = peek(dataset)
        if first_item:
            sample_keys = set(first_item.keys())
            if "messages" not in sample_keys and (
                "question" in sample_keys or "answer" in sample_keys
            ):
                raise ValueError(
                    "Dataset appears to be in a generic format (CSV, plain JSON, etc). "
                    "Please use the loader.transform() method to transform your data to Converse format first."
                )

        s3_client = boto3.client("s3")
        # Validate each data entry
        for i, sample in enumerate(dataset):
            try:
                sample_model = self.get_sample_model()
                sample_model.model_validate(
                    sample, context={"model": model, "s3_client": s3_client}
                )

                # Check optional field consistency
                self._check_optional_field_consistency(
                    sample,
                    i,
                    optional_fields,
                    field_consistency,
                    first_sample_with_field,
                )

                self.num_samples += 1
            except ValidationError as e:
                failed_samples_id_list.append(i)
                error_message += f"\nSample {i}:\n"
                for err in e.errors():
                    err["msg"] = err["msg"].replace("Value error, ", "")
                    sample_error_message = (
                        f"  - Location {err['loc']}: {err['msg']} (type={err['type']})\n"
                    )
                    error_message += sample_error_message
            except InfrastructureError:
                raise
            except Exception as e:
                raise ValueError(f"Unexpected error in sample {i}: {e}")

        # Report any failed validation results
        if error_message:
            failed_samples_str = _format_failed_samples(failed_samples_id_list)
            final_err_msg = f"Validation failed for samples: {failed_samples_str}\n{error_message}"
            raise ValueError(final_err_msg)

        # Row count checks driven by DATASET_CHECKS config
        self._validate_row_counts(
            model,
            kwargs.get("platform"),
            kwargs.get("training_method"),
        )

        print(f"{self.get_success_message()}")

    def _validate_row_counts(
        self,
        model: Model,
        platform: Optional[Platform],
        training_method: Optional[TrainingMethod],
    ) -> None:
        """Enforce max/min row count checks from DATASET_CHECKS config.

        Delegates to the shared ``validate_row_counts`` helper.
        Checks are skipped when ``platform`` or ``training_method`` is not provided.
        ``min_rows_recipe_field`` checks are skipped here (no recipe available).
        """
        if platform is None or training_method is None:
            logger.warning(
                "platform and/or training_method not provided — skipping row-count "
                "checks. Pass both to enable dataset size validation."
            )
            return

        validate_row_counts(self.num_samples, model, platform, training_method)

    def _check_optional_field_consistency(
        self,
        sample: Dict,
        sample_index: int,
        optional_fields: List[str],
        field_consistency: Dict[str, bool | None],
        first_sample_with_field: Dict[str, int],
    ) -> None:
        """
        Check that optional fields are used consistently across all samples.

        If an optional field appears in any sample, it must appear in all samples.

        Args:
            sample: The current sample being validated
            sample_index: The index of the current sample
            optional_fields: List of field names to check for consistency
            field_consistency: Dict tracking field state (None=never seen, True=always present)
            first_sample_with_field: Dict tracking which sample first introduced each field

        Raises:
            ValueError: If an optional field is present in some samples but not others
        """
        for field_name in optional_fields:
            has_field = self._has_nested_field(sample, field_name)

            if field_consistency[field_name] is None:
                # First time seeing this field - record its state
                field_consistency[field_name] = has_field
                if has_field:
                    first_sample_with_field[field_name] = sample_index
            elif field_consistency[field_name] != has_field:
                # Inconsistency detected - fail fast with detailed error
                if has_field:
                    raise ValueError(
                        f"Dataset consistency error: If any sample contains '{field_name}', "
                        f"all samples must contain '{field_name}'. Field first appeared in sample "
                        f"{sample_index} but is missing in earlier samples."
                    )
                else:
                    raise ValueError(
                        f"Dataset consistency error: If any sample contains '{field_name}', "
                        f"all samples must contain '{field_name}'. Field present in sample "
                        f"{first_sample_with_field[field_name]} but missing in sample {sample_index}."
                    )

    def _has_nested_field(self, sample: Dict, field_path: str) -> bool:
        """Check if a top-level field or nested field exists."""
        keys = field_path.split(".")
        current = sample

        for i, key in enumerate(keys):
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                # Check if any item in the list has the key
                return any(
                    self._has_nested_field(item, ".".join(keys[i:]))
                    for item in current
                    if isinstance(item, dict)
                )
            else:
                return False
        return True


def _is_valid_path(file_path: str) -> None:
    """Validates that file path contains only alphanumeric characters, underscores, hyphens, slashes, and dots."""
    pattern = r"^[\w\-/\.]+$"
    if not re.match(pattern, file_path):
        raise ValueError(
            f"Invalid characters in 'uri'. Only alphanumeric, underscores, hyphens, slashes, and dots are allowed"
        )


def _format_failed_samples(failed_samples_id_list: List[int]) -> str:
    """Format the list of failed sample IDs for error messages."""
    if len(failed_samples_id_list) > 3:
        first_sample_id = failed_samples_id_list[0]
        second_sample_id = failed_samples_id_list[1]
        last_sample_id = failed_samples_id_list[-1]
        return f"[{first_sample_id}, {second_sample_id}, ...{last_sample_id}]"
    else:
        return f"{failed_samples_id_list}"


# ---------------------------------------------------------------------------
# Scoped validation executors driven by DATASET_CHECKS config
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> List[str]:
    """Parse an S3 URI into (bucket, key)."""
    return uri[len("s3://") :].split("/", 1)


def _validate_keywords(instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]) -> None:
    text = getattr(instance, "text", None)
    if text is None:
        return
    keywords: List[str] = validation.get("invalid_keywords", [])
    # Reserved keyword matching is case-sensitive.
    found = [kw for kw in keywords if kw in text]
    if found:
        raise ValueError(f"Please do not use these keywords: {', '.join(found)}")


def _validate_file_size(instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]) -> None:
    uri = instance.source.s3Location.uri
    s3_client = ctx.get("s3_client")
    if s3_client is None:
        raise InfrastructureError("Invalid s3_client: s3_client cannot be None.")
    limit = validation["valid_max_limit"]
    try:
        bucket, key = _parse_s3_uri(uri)
        size = s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except Exception as e:
        raise InfrastructureError(f"Failed to validate file size for {uri}: {e}")
    if size > limit:
        raise ValueError(
            f"{uri} exceeds {limit // (1024 * 1024)} MB limit ({size / (1024 * 1024):.1f} MB)"
        )


def _validate_content_count(
    instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]
) -> None:
    field = validation["name"].replace("_count", "")
    limit = validation["valid_max_limit"]
    count = sum(getattr(item, field, None) is not None for item in instance.content)
    if count > limit:
        raise ValueError(f"Only {limit} {field}(s) allowed per message, found {count}")


def _validate_video_duration(
    instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]
) -> None:

    uri = instance.source.s3Location.uri
    s3_client = ctx.get("s3_client")
    if s3_client is None:
        raise InfrastructureError("Invalid s3_client: s3_client cannot be None.")
    limit = validation["valid_max_limit"]
    try:
        from pymediainfo import MediaInfo
    except ImportError:
        raise InfrastructureError(
            "pymediainfo and/or its system dependency libmediainfo not installed. "
            "See 'Image/Video validation' in docs/data_prep.md for setup instructions."
        )

    try:
        bucket, key = _parse_s3_uri(uri)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=tmp)
            tmp.flush()
            info = MediaInfo.parse(tmp.name)
            video_tracks = [t for t in info.tracks if t.track_type == "Video"]
            if not video_tracks or video_tracks[0].duration is None:
                raise InfrastructureError(f"Unable to determine video duration for {uri}")
            duration_ms = video_tracks[0].duration
            duration_s = float(duration_ms) / 1000
            if duration_s > limit:
                raise ValueError(f"Video {uri} duration {duration_s:.0f}s exceeds {limit}s limit")
    except ValueError:
        raise
    except Exception as e:
        raise InfrastructureError(f"Failed to validate video duration for {uri}: {e}")


_ExecutorFn = Callable[[Any, DatasetCheckEntry, Dict[str, Any]], None]


def _validate_format_allowlist(
    instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]
) -> None:
    supported_models = validation["valid_models"]
    supported_formats = validation["valid_formats"]

    model = ctx.get("model")
    fmt = getattr(instance, "format", None)
    if fmt is None:
        raise ValueError("Invalid format: multimodal content format cannot be None.")
    if model not in supported_models or fmt.lower() not in supported_formats:
        raise ValueError(
            f"Invalid format {fmt} on model {model}. "
            f"Supported formats are {supported_formats} on {supported_models}."
        )

    # Magic number consistency check using filetype library

    uri = instance.source.s3Location.uri
    s3_client = ctx.get("s3_client")
    if s3_client is None:
        raise InfrastructureError("Invalid s3_client: s3_client cannot be None.")
    try:
        bucket, key = _parse_s3_uri(uri)
        resp = s3_client.get_object(Bucket=bucket, Key=key, Range="bytes=0-261")
        header = resp["Body"].read()
    except Exception as e:
        raise InfrastructureError(f"Failed to read file header for {uri}: {e}")
    kind = filetype.match(header)
    if kind is None:
        raise ValueError(f"File {uri} declared format '{fmt}' does not match file contents")
    detected = kind.extension
    declared = "jpg" if fmt.lower() == "jpeg" else fmt.lower()
    if detected != declared:
        raise ValueError(
            f"File {uri} declared format '{fmt}' does not match detected format '{detected}'"
        )


def _validate_content_allowlist(
    instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]
) -> None:
    supported_models = validation["valid_models"]
    supported_contents = validation["valid_contents"]

    model = ctx.get("model")
    has_content = any(
        getattr(item, content_type, None) is not None
        for item in instance.content
        for content_type in supported_contents
    )

    if has_content and model not in supported_models:
        contents = [
            getattr(item, content_type, None)
            for item in instance.content
            for content_type in supported_contents
            if getattr(item, content_type, None) is not None
        ]
        raise ValueError(
            f"Invalid content {contents} on model {model}. "
            f"Supported contents are {supported_contents} on {supported_models}."
        )


def _validate_image_dimensions(
    instance: Any, validation: DatasetCheckEntry, ctx: Dict[str, Any]
) -> None:

    uri = instance.source.s3Location.uri
    s3_client = ctx.get("s3_client")
    if s3_client is None:
        raise InfrastructureError("Invalid s3_client: s3_client cannot be None.")
    min_dim = validation["valid_min_limit"]
    try:
        from pymediainfo import MediaInfo
    except ImportError:
        raise InfrastructureError(
            "pymediainfo and/or its system dependency libmediainfo not installed. "
            "See 'Image/Video validation' in docs/data_prep.md for setup instructions."
        )
    try:
        bucket, key = _parse_s3_uri(uri)
        with tempfile.NamedTemporaryFile() as tmp:
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=tmp)
            tmp.flush()
            info = MediaInfo.parse(tmp.name)
            image_tracks = [t for t in info.tracks if t.track_type == "Image"]
            track = image_tracks[0] if image_tracks else None
            if track is None:
                general = [t for t in info.tracks if t.track_type == "General"]
                track = general[0] if general else None
            if track is None or track.width is None or track.height is None:
                raise InfrastructureError(f"Unable to determine image dimensions for {uri}")
            width, height = int(track.width), int(track.height)
            if width < min_dim or height < min_dim:
                raise ValueError(
                    f"Image {uri} dimensions {width}x{height} below minimum {min_dim}x{min_dim}"
                )
    except ValueError:
        raise
    except Exception as e:
        raise InfrastructureError(f"Failed to validate image dimensions for {uri}: {e}")


_EXECUTORS: Dict[str, _ExecutorFn] = {
    "keyword": _validate_keywords,
    "file_size": _validate_file_size,
    "content_count": _validate_content_count,
    "video_duration": _validate_video_duration,
    "format_allowlist": _validate_format_allowlist,
    "content_allowlist": _validate_content_allowlist,
    "image_dimensions": _validate_image_dimensions,
}


def _run_validations_in_scope(
    instance: BaseModel, info: ValidationInfo, training_method: TrainingMethod
) -> None:
    """Run all DATASET_CHECKS scoped to this model's class name."""
    scope = type(instance).__name__
    ctx = info.context or {}
    model = ctx.get("model")
    if model is None:
        raise ValueError("Invalid model: model type cannot be None.")

    for check in DATASET_CHECKS:
        if not check["filterable"]:
            continue
        if "scope" not in check or scope not in check["scope"]:
            continue
        if training_method not in check["applicable_training_methods"]:
            continue
        if model not in check["applicable_models"]:
            continue

        executor = _EXECUTORS.get(check["type"])
        if executor:
            executor(instance, check, ctx)
