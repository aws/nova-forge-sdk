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
"""Shared job caching utility for Nova Forge SDK.

Provides ``JobCacheContext`` and pure functions for persisting / loading
job results to/from the local filesystem.  Each service class builds a
``JobCacheContext`` from its ``ForgeConfig`` and constructor arguments,
then calls ``load_existing_result`` / ``persist_result`` around its main
method.

Rule: this module imports nothing outside ``core/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from amzn_nova_forge.core.constants import DEFAULT_JOB_CACHE_DIR
from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.core.result.job_result import BaseJobResult, JobStatus

if TYPE_CHECKING:
    from amzn_nova_forge.core.types import ForgeConfig

logger = logging.getLogger(__name__)


def _default_job_caching_config() -> Dict[str, Any]:
    """Return the default job caching configuration.

    Matches the defaults historically set in ``NovaModelCustomizer.__init__``.
    """
    return {
        "include_core": True,
        "include_recipe": True,
        "include_infra": False,
        "include_params": [],
        "exclude_params": [],
        "allowed_statuses": [JobStatus.COMPLETED, JobStatus.IN_PROGRESS],
    }


@dataclass
class JobCacheContext:
    """Flat context capturing everything the caching utility needs.

    Built by service classes from ``ForgeConfig`` + constructor args.
    Eliminates the need to pass a raw ``customizer`` object.
    """

    enable_job_caching: bool
    job_cache_dir: str = DEFAULT_JOB_CACHE_DIR
    job_caching_config: Optional[Dict[str, Any]] = None
    model: Optional[Model] = None
    method: Optional[TrainingMethod] = None
    data_s3_path: Optional[str] = None
    model_path: Optional[str] = None
    output_s3_path: Optional[str] = None
    instance_type: Optional[str] = None
    instance_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.job_caching_config is None:
            self.job_caching_config = _default_job_caching_config()


# ---------------------------------------------------------------------------
# Helper: build a context from ForgeConfig
# ---------------------------------------------------------------------------


def build_cache_context(
    config: ForgeConfig,
    *,
    model: Optional[Model] = None,
    method: Optional[TrainingMethod] = None,
    data_s3_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_s3_path: Optional[str] = None,
    instance_type: Optional[str] = None,
    instance_count: Optional[int] = None,
) -> JobCacheContext:
    """Build a ``JobCacheContext`` from a ``ForgeConfig`` and service-class attributes."""
    return JobCacheContext(
        enable_job_caching=config.enable_job_caching,
        job_cache_dir=config.job_cache_dir,
        job_caching_config=config.job_caching_config,
        model=model,
        method=method,
        data_s3_path=data_s3_path,
        model_path=model_path,
        output_s3_path=output_s3_path,
        instance_type=instance_type,
        instance_count=instance_count,
    )


# ---------------------------------------------------------------------------
# Caching functions (operate on JobCacheContext, not a raw customizer)
# ---------------------------------------------------------------------------


def generate_job_hash(ctx: JobCacheContext, job_name: str, job_type: str, **job_params: Any) -> str:
    """Generate a segmented hash where each parameter gets its own labeled segment.

    This allows flexible cache matching by comparing only relevant segments.
    """
    segments: Dict[str, str] = {}

    segments["model"] = hashlib.sha256(
        str(ctx.model.value if ctx.model else "").encode()
    ).hexdigest()[:8]
    segments["method"] = hashlib.sha256(
        str(ctx.method.value if ctx.method else "").encode()
    ).hexdigest()[:8]
    segments["data_s3_path"] = hashlib.sha256((ctx.data_s3_path or "").encode()).hexdigest()[:8]
    segments["job_type"] = hashlib.sha256(job_type.encode()).hexdigest()[:8]
    segments["model_path"] = hashlib.sha256(str(ctx.model_path).encode()).hexdigest()[:8]

    if "recipe_path" in job_params:
        segments["recipe_path"] = hashlib.sha256(
            str(job_params["recipe_path"]).encode()
        ).hexdigest()[:8]

    overrides = job_params.get("overrides", {})
    if isinstance(overrides, dict):
        for param, value in overrides.items():
            segments[f"override_{param}"] = hashlib.sha256(str(value).encode()).hexdigest()[:8]

    if ctx.instance_type is not None:
        segments["instance_type"] = hashlib.sha256(str(ctx.instance_type).encode()).hexdigest()[:8]
    if ctx.instance_count is not None:
        segments["instance_count"] = hashlib.sha256(str(ctx.instance_count).encode()).hexdigest()[
            :8
        ]

    for key, value in job_params.items():
        if key not in ["recipe_path", "overrides"]:
            segments[key] = hashlib.sha256(str(value).encode()).hexdigest()[:8]

    segment_pairs = [f"{k}:{v}" for k, v in sorted(segments.items())]
    return ",".join(segment_pairs)


def should_persist_results(ctx: JobCacheContext) -> bool:
    """Return True if caching is enabled and the cache directory is usable."""
    if not ctx.enable_job_caching:
        return False
    if not ctx.job_cache_dir:
        logger.warning("Job caching enabled but job_cache_dir is not set")
        return False
    cache_path = Path(ctx.job_cache_dir)
    if not cache_path.exists():
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create job cache directory '{ctx.job_cache_dir}': {e}")
            return False
    return True


def matches_job_cache_criteria(
    job_caching_config: dict, stored_hash: str, current_hash: str
) -> bool:
    """Check if a stored segmented hash matches the current hash per config rules."""

    def parse_segments(hash_str: str) -> Dict[str, str]:
        segments: Dict[str, str] = {}
        for pair in hash_str.split(","):
            if ":" in pair:
                key, value = pair.split(":", 1)
                segments[key] = value
        return segments

    stored_segments = parse_segments(stored_hash)
    current_segments = parse_segments(current_hash)

    config = job_caching_config

    exclude_params = config.get("exclude_params", [])
    if isinstance(exclude_params, list):
        for param in exclude_params:
            stored_segments.pop(param, None)
            current_segments.pop(param, None)

    include_params = config.get("include_params", [])
    if isinstance(include_params, list):
        for param in include_params:
            if stored_segments.get(param) != current_segments.get(param):
                return False

    exclude_params = config.get("exclude_params", [])
    if isinstance(exclude_params, list) and "*" in exclude_params:
        return True

    if config.get("include_core", True):
        core_fields = ["model", "method", "data_s3_path", "job_type", "model_path"]
        for field in core_fields:
            if stored_segments.get(field) != current_segments.get(field):
                return False

    if config.get("include_recipe", True):
        if stored_segments.get("recipe_path") != current_segments.get("recipe_path"):
            return False
        all_override_keys: set[str] = set()
        for segments in [stored_segments, current_segments]:
            all_override_keys.update(k for k in segments.keys() if k.startswith("override_"))
        for override_key in all_override_keys:
            if stored_segments.get(override_key) != current_segments.get(override_key):
                return False

    if config.get("include_infra", False):
        infra_fields = ["instance_type", "instance_count"]
        for field in infra_fields:
            if stored_segments.get(field) != current_segments.get(field):
                return False

    return True


def get_result_file_path(
    ctx: JobCacheContext, job_name: str, job_type: str, **job_params: Any
) -> Path:
    """Return the path for a new persisted result file."""
    if not should_persist_results(ctx):
        raise ValueError("Cannot get result file path when persistence is disabled")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
    filename = f"{job_name}_{job_type}_{timestamp}.json"
    return Path(ctx.job_cache_dir) / filename


def load_existing_result(
    ctx: JobCacheContext, job_name: str, job_type: str, **job_params: Any
) -> Optional[BaseJobResult]:
    """Load a cached result matching the given parameters, or return None."""
    if not should_persist_results(ctx):
        return None

    caching_config = ctx.job_caching_config or _default_job_caching_config()

    allowed_statuses = caching_config.get("allowed_statuses", None)
    if not isinstance(allowed_statuses, list):
        logger.error(
            "Invalid allowed_statuses configuration: expected list, "
            f"got {type(allowed_statuses).__name__} with value {allowed_statuses}. "
            "Skipping job cache lookup."
        )
        return None

    try:
        current_hash = generate_job_hash(ctx, job_name, job_type, **job_params)
        results_dir = Path(ctx.job_cache_dir)

        if not results_dir.exists():
            return None

        pattern = f"{job_name}_{job_type}_*.json"
        for result_file in results_dir.glob(pattern):
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)

                stored_hash = data.get("_job_cache_hash")
                if stored_hash and matches_job_cache_criteria(
                    caching_config, stored_hash, current_hash
                ):
                    result = BaseJobResult.load(str(result_file))
                    result._job_cache_hash = stored_hash  # type: ignore[attr-defined]

                    job_status, raw_status = result.get_job_status()
                    if job_status in allowed_statuses:
                        logger.info(
                            f"Reusing existing {job_type} result for {job_name} "
                            f"with status {job_status} from {result_file.absolute()}"
                        )
                        return result
                    else:
                        logger.info(
                            f"Found matching {job_type} result for {job_name} "
                            f"but job status {job_status} not in allowed statuses "
                            f"{[s.value for s in allowed_statuses]}"
                        )
            except Exception as e:
                logger.debug(f"Skipping corrupted result file {result_file}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Failed to search for existing results: {e}")

    return None


def collect_all_parameters(
    ctx: JobCacheContext, job_name: str, job_type: str, **job_params: Any
) -> dict:
    """Collect all relevant parameters from the cache context and job params."""
    all_params: Dict[str, Any] = {}

    if ctx.instance_type is not None:
        all_params["infra_instance_type"] = ctx.instance_type
    if ctx.instance_count is not None:
        all_params["infra_instance_count"] = ctx.instance_count

    all_params["model"] = (
        ctx.model.value if ctx.model and hasattr(ctx.model, "value") else str(ctx.model)
    )
    all_params["method"] = (
        ctx.method.value if ctx.method and hasattr(ctx.method, "value") else str(ctx.method)
    )
    all_params["data_s3_path"] = ctx.data_s3_path
    all_params["output_s3_path"] = ctx.output_s3_path
    all_params["model_path"] = ctx.model_path

    all_params.update(job_params)

    return all_params


def persist_result(
    ctx: JobCacheContext,
    result: BaseJobResult,
    job_name: str,
    job_type: str,
    **job_params: Any,
) -> None:
    """Persist a job result to the cache directory if caching is enabled."""
    if not should_persist_results(ctx):
        return

    try:
        result_file = get_result_file_path(ctx, job_name, job_type, **job_params)
        result_file.parent.mkdir(parents=True, exist_ok=True)

        data = result._to_dict()
        data["__class_name__"] = result.__class__.__name__

        if ctx.enable_job_caching:
            all_params = collect_all_parameters(ctx, job_name, job_type, **job_params)
            segmented_hash = generate_job_hash(ctx, job_name, job_type, **all_params)
            data["_job_cache_hash"] = segmented_hash
            data["_all_parameters"] = all_params

        with open(result_file, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Job result saved to {result_file}")
    except Exception as e:
        logger.warning(f"Failed to persist {job_type} result for {job_name}: {e}")
