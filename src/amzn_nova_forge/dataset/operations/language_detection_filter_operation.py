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
"""Language detection operation for data preparation pipelines.

Filters records by detected language using a FastText ``lid`` model. Keeps
only rows whose detected language is in the ``languages`` allowlist and
(optionally) whose confidence meets a ``min_score`` threshold. Output
schema matches the input — no extra columns are written.

Usage via loader (recommended)::

    loader.load("s3://my-bucket/raw/data.jsonl")
    loader.filter(
        method=FilterMethod.LANGUAGE_DETECTION,
        output_path="s3://my-bucket/filtered/",
        model_path="s3://my-bucket/models/lid.176.bin",
        languages=["en", "fr"],
        min_score=0.5,
    ).execute()

Usage standalone::

    from amzn_nova_forge.dataset.operations.language_detection_filter_operation import (
        LanguageDetectionFilterOperation,
    )
    from amzn_nova_forge.dataset.data_state import DataLocation, DataState

    op = LanguageDetectionFilterOperation()
    state = DataState(path="s3://my-bucket/raw/", format="jsonl", location=DataLocation.S3)
    result = op.execute(
        loader=None,
        state=state,
        output_path="s3://my-bucket/filtered/",
        model_path="s3://my-bucket/models/lid.176.bin",
        languages=["en"],
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from typing import Any, Dict, Tuple, Type

import boto3
import requests
from botocore.exceptions import ClientError

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.base import FilterOperationResult
from amzn_nova_forge.dataset.operations.filter_operation import (
    NovaForgeFilterOperationBase,
    _read_summary_json,
    _reload_output_into_loader,
    _resolve_s3_directory_to_jsonl,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig
from amzn_nova_forge.util.s3_utils import ensure_bucket_exists, get_dataprep_bucket_name

logger = logging.getLogger(__name__)

_PIPELINE_ID = "language_detection"

# When the caller omits ``model_path``, the SDK downloads the canonical
# FastText language-identification model from Meta's public hosted copy
# (MIT-licensed) and stages it once per account in the data-prep bucket.
# Subsequent runs skip straight to the cached copy.
_DEFAULT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_DEFAULT_MODEL_S3_KEY = "nova-forge/models/language_detection/lid.176.bin"
_DEFAULT_MODEL_FILENAME = "lid.176.bin"
_DEFAULT_MODEL_SIZE_MB = 126
# SHA-256 of the canonical lid.176.bin hosted by Meta. Verified after
# download; a mismatch aborts the stage before upload. Protects against
# CDN compromise, MITM, and partial/truncated downloads.
_DEFAULT_MODEL_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"

# Parameters forwarded to the runtime job. These map 1:1 to
# ``LanguageDetectionStage.__init__`` kwargs in AGIDataCurator.
_LANGUAGE_DETECTION_PARAM_KEYS = frozenset(
    {
        "model_path",
        "lang_field",
        "score_field",
        "languages",
        "min_score",
        "keep_undetected",
    }
)


class LanguageDetectionFilterOperation(NovaForgeFilterOperationBase):
    """Filter records by detected language (FastText lid).

    Drops rows whose detected language is outside the ``languages``
    allowlist or whose confidence falls below ``min_score``. The output
    schema matches the input — no extra columns are written.

    Delegates to the ``language_detection`` ForgeWorkflows pipeline
    registered in AGIDataCurator.

    Required parameters:

    - ``languages`` (list[str]): Non-empty ISO 639-1 allowlist
      (e.g. ``["en", "fr"]``).

    Optional parameters:

    - ``model_path`` (str): Path to a FastText lid model file (local or
      ``s3://`` URI). If omitted, the SDK auto-stages the canonical
      ``lid.176.bin`` (~126 MB, MIT-licensed) from Meta's public copy
      into ``s3://<dataprep-bucket>/nova-forge/models/language_detection/``
      on first use, then reuses the cached copy on subsequent runs.
    - ``min_score`` (float): Minimum confidence in [0.0, 1.0]. Default
      ``0.0`` (no score threshold).
    - ``keep_undetected`` (bool): Keep records where detection failed
      (e.g. empty text). Default ``False``.

    Lightweight — no AWS calls until ``execute()`` is invoked.

    Note: Annotation-only language tagging (adds ``lang`` / ``lang_score``
    columns instead of filtering) will land as a separate transform
    operation in a follow-up.
    """

    __slots__ = ()

    _FILTER_NAME = "Language Detection"

    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeManager

        return (GlueRuntimeManager, SMTJRuntimeManager)

    def execute(self, loader: Any, **kwargs: Any) -> FilterOperationResult:
        """Execute the language_detection pipeline.

        Args:
            loader: The DatasetLoader instance (or None for standalone use).
            state: DataState describing the current data (path, format, location).
            output_path: Path for filtered output (S3 URI).
            model_path: Optional path to a FastText lid model. If omitted,
                the SDK auto-stages ``lid.176.bin`` (~126 MB) once per
                account under the data-prep bucket.
            input_format: ``"parquet"`` or ``"jsonl"``. Defaults to state format.
            output_format: ``"parquet"`` or ``"jsonl"``. Default ``"parquet"``.
            text_field: Column/field name containing the text. Default ``"text"``.
            languages: Non-empty list of ISO 639-1 language codes to keep (required).
            min_score: Minimum confidence score threshold in [0.0, 1.0]. Default 0.0.
            keep_undetected: Keep rows where detection failed. Default False.
            lang_field: Output column for language code. Default ``"lang"``.
            score_field: Output column for confidence. Default ``"lang_score"``.
            runtime_manager: A ``RuntimeManager`` instance. Defaults to an
                ``SMTJRuntimeManager`` flipped into data-prep mode. Pass a
                ``GlueRuntimeManager`` to use the legacy runtime (closed to
                new customers after April 30, 2026).
            extra_args: Additional kwargs forwarded to the pipeline.

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
        output_format = kwargs.get("output_format", "parquet")
        text_field = kwargs.get("text_field", "text")
        extra_args = kwargs.get("extra_args")
        job_name = kwargs.get("job_name")

        languages = kwargs.get("languages")
        if not languages:
            raise ValueError(
                "LanguageDetectionFilterOperation requires a non-empty `languages` "
                "list (ISO 639-1 codes, e.g. ['en', 'fr']). "
                "Annotation-only mode is not supported."
            )

        min_score = kwargs.get("min_score")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError(
                f"`min_score` must be in [0.0, 1.0], got {min_score}. "
                "FastText confidence is a probability in [0, 1]."
            )

        if "model_path" not in kwargs:
            kwargs["model_path"] = self._ensure_default_model()

        manager = self._resolve_runtime_manager(input_path, **kwargs)

        self._log_start(manager, input_path, input_format, output_path)
        if "text_field" not in kwargs:
            logger.info("  Using default text_field=%r", text_field)

        merged_extra_args: Dict[str, Any] = {
            **(extra_args or {}),
            "pipeline_id": _PIPELINE_ID,
        }

        # Forward stage params so the runtime job passes them to
        # ForgeWorkflows.execute("language_detection", **kwargs).
        for key in _LANGUAGE_DETECTION_PARAM_KEYS:
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
            # Only language detection needs fasttext at runtime — other
            # pipelines should not pay this startup cost.
            extra_pip_packages=["fasttext-wheel"],
        )

        manager.execute(job_config)

        if loader is not None and output_path:
            _reload_output_into_loader(loader, output_path, output_format)

        # LanguageDetectionStage currently returns a ray.data.Dataset from
        # `pipeline.run()`, so the Glue entry script has no dict to
        # serialize into `_summary.json`. `_read_summary_json` will fall
        # back to (0, 0) and `FilterOperationResult.__str__` will log
        # "counts unknown". When AGIDataCurator surfaces input/filtered
        # counts for stage-based workflows, this will start reporting real
        # numbers without any SDK change.
        total_count, filtered_count = _read_summary_json(
            output_path,
            total_key="input_count",
            filtered_key="num_filtered",
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

    @staticmethod
    def _ensure_default_model() -> str:
        """Return the S3 URI of the default FastText lid.176.bin model.

        Called only when the caller omits ``model_path``. The model is
        cached in the auto-created data-prep bucket under
        ``nova-forge/models/language_detection/lid.176.bin``:

        - First call per account: downloads ~126 MB from Meta's public
          MIT-licensed copy, uploads to S3, and returns the S3 URI.
        - Subsequent calls: HEADs the S3 key and returns immediately.

        Users who want to pin a specific model (air-gapped accounts,
        custom training, etc.) can still pass ``model_path=...``
        explicitly to skip this entirely.
        """
        bucket = get_dataprep_bucket_name()
        ensure_bucket_exists(bucket)
        s3_uri = f"s3://{bucket}/{_DEFAULT_MODEL_S3_KEY}"

        s3 = boto3.client("s3")
        try:
            s3.head_object(Bucket=bucket, Key=_DEFAULT_MODEL_S3_KEY)
            logger.info("Using cached FastText lid model at %s", s3_uri)
            return s3_uri
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in ("404", "NoSuchKey", "NotFound"):
                raise

        # First-time stage — download from upstream, upload to S3.
        logger.info(
            "FastText lid model not found in S3. Staging %s (~%d MB) from %s",
            _DEFAULT_MODEL_FILENAME,
            _DEFAULT_MODEL_SIZE_MB,
            _DEFAULT_MODEL_URL,
        )
        logger.info(
            "This is a one-time download per account. "
            "Subsequent runs will reuse the cached copy at %s",
            s3_uri,
        )

        # Download to a temp file (NamedTemporaryFile so boto can upload
        # by path). Cleaned up in the finally block.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".bin", prefix="lid176-", delete=False) as tmp:
                tmp_path = tmp.name
            logger.info("Downloading %s -> %s", _DEFAULT_MODEL_URL, tmp_path)
            # Stream-hash while writing so we don't need a second pass
            # over the 126 MB file on disk.
            sha = hashlib.sha256()
            with requests.get(_DEFAULT_MODEL_URL, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            sha.update(chunk)
                            out.write(chunk)
            digest = sha.hexdigest()
            if digest != _DEFAULT_MODEL_SHA256:
                raise RuntimeError(
                    f"Integrity check failed for {_DEFAULT_MODEL_FILENAME} "
                    f"downloaded from {_DEFAULT_MODEL_URL}: expected SHA-256 "
                    f"{_DEFAULT_MODEL_SHA256}, got {digest}. The file may be "
                    "truncated or the upstream copy may have changed. "
                    "Staging aborted — nothing uploaded to S3."
                )
            logger.info("SHA-256 verified. Uploading to %s", s3_uri)
            s3.upload_file(tmp_path, bucket, _DEFAULT_MODEL_S3_KEY)
            logger.info("FastText lid model staged at %s", s3_uri)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("Could not remove temp file %s", tmp_path)

        return s3_uri
