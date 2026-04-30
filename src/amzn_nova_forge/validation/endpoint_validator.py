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
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from amzn_nova_forge.core.constants import SUPPORTED_SMI_CONFIGS
from amzn_nova_forge.core.enums import Model
from amzn_nova_forge.util.logging import logger

S3_URI_PREFIX_REGEX = re.compile(r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9_.-]+)*/$")

BEDROCK_DEPLOYMENT_ARN_REGEX = re.compile(
    r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:custom-model-deployment/[A-Za-z0-9-_]+$"
)

SAGEMAKER_ENDPOINT_ARN_REGEX = re.compile(
    r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:endpoint/[A-Za-z0-9-_]+$"
)

# Matches SageMaker ARNs across all AWS partitions (standard, GovCloud, China, etc.)
SAGEMAKER_ARN_RE = re.compile(r"^arn:aws[\w-]*:sagemaker:")

# --- SageMaker inference environment variable keys ---
# Required
ENV_CONTEXT_LENGTH = "CONTEXT_LENGTH"
ENV_MAX_CONCURRENCY = "MAX_CONCURRENCY"

# Optional
ENV_DEFAULT_TEMPERATURE = "DEFAULT_TEMPERATURE"
ENV_DEFAULT_TOP_P = "DEFAULT_TOP_P"
ENV_DEFAULT_TOP_K = "DEFAULT_TOP_K"
ENV_DEFAULT_MAX_NEW_TOKENS = "DEFAULT_MAX_NEW_TOKENS"
ENV_DEFAULT_LOGPROBS = "DEFAULT_LOGPROBS"

ENV_SPECULATIVE_DECODING_METHOD = "SPECULATIVE_DECODING_METHOD"
ENV_DISABLE_SPECULATIVE_DECODING = "DISABLE_SPECULATIVE_DECODING"
ENV_NUM_SPECULATIVE_TOKENS = "NUM_SPECULATIVE_TOKENS"
ENV_SUFFIX_DECODING_MAX_TREE_DEPTH = "SUFFIX_DECODING_MAX_TREE_DEPTH"
ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS = "SUFFIX_DECODING_MAX_CACHED_REQUESTS"
ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR = "SUFFIX_DECODING_MAX_SPEC_FACTOR"
ENV_SUFFIX_DECODING_MIN_TOKEN_PROB = "SUFFIX_DECODING_MIN_TOKEN_PROB"
ENV_KV_CACHE_DTYPE = "KV_CACHE_DTYPE"
ENV_QUANTIZATION_DTYPE = "QUANTIZATION_DTYPE"

VALID_SPECULATIVE_DECODING_METHODS = {"eagle3", "suffix"}
VALID_DISABLE_SPECULATIVE_DECODING_VALUES = {"true", "false"}
VALID_KV_CACHE_DTYPES = {"fp8"}
VALID_QUANTIZATION_DTYPES = {"fp8"}

_FIELD_TO_ENV_KEY = {
    "context_length": ENV_CONTEXT_LENGTH,
    "max_concurrency": ENV_MAX_CONCURRENCY,
    "default_temperature": ENV_DEFAULT_TEMPERATURE,
    "default_top_p": ENV_DEFAULT_TOP_P,
    "default_top_k": ENV_DEFAULT_TOP_K,
    "default_max_new_tokens": ENV_DEFAULT_MAX_NEW_TOKENS,
    "default_logprobs": ENV_DEFAULT_LOGPROBS,
    "speculative_decoding_method": ENV_SPECULATIVE_DECODING_METHOD,
    "disable_speculative_decoding": ENV_DISABLE_SPECULATIVE_DECODING,
    "num_speculative_tokens": ENV_NUM_SPECULATIVE_TOKENS,
    "suffix_decoding_max_tree_depth": ENV_SUFFIX_DECODING_MAX_TREE_DEPTH,
    "suffix_decoding_max_cached_requests": ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS,
    "suffix_decoding_max_spec_factor": ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR,
    "suffix_decoding_min_token_prob": ENV_SUFFIX_DECODING_MIN_TOKEN_PROB,
    "kv_cache_dtype": ENV_KV_CACHE_DTYPE,
    "quantization_dtype": ENV_QUANTIZATION_DTYPE,
}

_ENV_KEY_TO_FIELD = {v: k for k, v in _FIELD_TO_ENV_KEY.items()}


class SageMakerEndpointEnvironment(BaseModel):
    """Validated environment variables for a SageMaker inference endpoint.

    Construct with typed Python values. Call ``to_env_dict()`` to get the
    ``Dict[str, str]`` expected by the SageMaker ``CreateModel`` API.

    Example::

        env = SageMakerEndpointEnvironment(
            context_length=4000,
            max_concurrency=1,
            default_temperature=0.7,
            kv_cache_dtype="fp8",
        )
        create_model(..., environment=env.to_env_dict())
    """

    # --- Required ---
    context_length: int = 4000
    max_concurrency: int = 1

    # --- Sampling defaults (optional) ---
    default_temperature: Optional[float] = None
    default_top_p: Optional[float] = None
    default_top_k: Optional[int] = None
    default_max_new_tokens: Optional[int] = None
    default_logprobs: Optional[int] = None

    # --- Speculative decoding (optional) ---
    speculative_decoding_method: Optional[Literal["eagle3", "suffix"]] = None
    disable_speculative_decoding: Optional[Literal["true", "false"]] = None
    num_speculative_tokens: Optional[int] = None

    @field_validator(
        "speculative_decoding_method",
        "disable_speculative_decoding",
        "kv_cache_dtype",
        "quantization_dtype",
        mode="before",
    )
    @classmethod
    def _normalize_literal_str(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            return v.lower()
        return v

    # --- Suffix decoding tuning (optional) ---
    suffix_decoding_max_tree_depth: Optional[int] = None
    suffix_decoding_max_cached_requests: Optional[int] = None
    suffix_decoding_max_spec_factor: Optional[float] = None
    suffix_decoding_min_token_prob: Optional[float] = None

    # --- Memory / quantization (optional) ---
    kv_cache_dtype: Optional[Literal["fp8"]] = None
    quantization_dtype: Optional[Literal["fp8"]] = None

    @field_validator("context_length", "max_concurrency")
    @classmethod
    def _positive_int(cls, v: int, info) -> int:  # type: ignore[override]
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("default_temperature")
    @classmethod
    def _temperature_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 2.0):
            raise ValueError("default_temperature must be between 0 and 2.0")
        return v

    @field_validator("default_top_p")
    @classmethod
    def _top_p_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 1e-10 or v > 1.0):
            raise ValueError("default_top_p must be between 1e-10 and 1.0")
        return v

    @field_validator("default_top_k")
    @classmethod
    def _top_k_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < -1 or v == 0):
            raise ValueError("default_top_k must be -1 (disabled) or a positive integer")
        return v

    @field_validator("default_max_new_tokens")
    @classmethod
    def _max_new_tokens_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("default_max_new_tokens must be a positive integer")
        return v

    @field_validator("default_logprobs")
    @classmethod
    def _logprobs_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 20):
            raise ValueError("default_logprobs must be an integer between 1 and 20")
        return v

    @field_validator("num_speculative_tokens")
    @classmethod
    def _num_spec_tokens_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 10):
            raise ValueError("num_speculative_tokens must be an integer between 1 and 10")
        return v

    @field_validator("suffix_decoding_max_tree_depth")
    @classmethod
    def _tree_depth_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("suffix_decoding_max_tree_depth must be a positive integer")
        return v

    @field_validator("suffix_decoding_max_cached_requests")
    @classmethod
    def _cached_requests_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("suffix_decoding_max_cached_requests must be a non-negative integer")
        return v

    @field_validator("suffix_decoding_max_spec_factor")
    @classmethod
    def _spec_factor_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("suffix_decoding_max_spec_factor must be a non-negative number")
        return v

    @field_validator("suffix_decoding_min_token_prob")
    @classmethod
    def _min_token_prob_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1.0):
            raise ValueError("suffix_decoding_min_token_prob must be between 0 and 1.0")
        return v

    def to_env_dict(self) -> Dict[str, str]:
        """Return a ``Dict[str, str]`` suitable for the SageMaker ``CreateModel``
        ``Environment`` parameter.  Only non-``None`` fields are included."""
        result: Dict[str, str] = {}
        for field_name, env_key in _FIELD_TO_ENV_KEY.items():
            value = getattr(self, field_name)
            if value is not None:
                result[env_key] = str(value)
        return result

    @classmethod
    def from_env_dict(cls, env_vars: Dict[str, str]) -> "SageMakerEndpointEnvironment":
        """Construct from a raw ``Dict[str, str]`` of environment variables.

        Raises:
            ValueError: For unrecognised environment variable keys.
            ValidationError: For missing required keys or invalid values.
        """
        kwargs = {}
        for env_key, raw_value in env_vars.items():
            field_name = _ENV_KEY_TO_FIELD.get(env_key)
            if field_name is None:
                raise ValueError(f"Invalid environment variable: {env_key}")
            kwargs[field_name] = raw_value
        return cls.model_validate(kwargs)

    def validate_smi_config_bounds(self, model: Model, instance_type: str) -> None:
        """Validate ``context_length`` and ``max_concurrency`` against the
        supported SMI configuration table for the given model and instance type.

        Raises ``ValueError`` if the values exceed the supported tier limits.
        Logs a warning and returns silently for unknown model/instance combos.
        """
        config_key = (model, instance_type)
        tiers = SUPPORTED_SMI_CONFIGS.get(config_key)

        if tiers is None:
            logger.warning(
                f"No SMI configuration found for ({model.name}, {instance_type}). "
                f"Skipping CONTEXT_LENGTH/MAX_CONCURRENCY bounds validation."
            )
            return

        sorted_tiers = sorted(tiers, key=lambda t: t[0])

        max_supported_context = sorted_tiers[-1][0]
        if self.context_length > max_supported_context:
            raise ValueError(
                f"CONTEXT_LENGTH={self.context_length} exceeds maximum supported "
                f"value of {max_supported_context} for {model.name} on "
                f"{instance_type}."
            )

        for tier_context, tier_concurrency in sorted_tiers:
            if self.context_length <= tier_context:
                if self.max_concurrency > tier_concurrency:
                    raise ValueError(
                        f"MAX_CONCURRENCY={self.max_concurrency} exceeds maximum "
                        f"supported value of {tier_concurrency} for {model.name} "
                        f"on {instance_type} at CONTEXT_LENGTH="
                        f"{self.context_length} (tier <={tier_context})."
                    )
                return


def is_sagemaker_arn(value: str) -> bool:
    """Return True if value looks like a SageMaker ARN (any partition)."""
    return bool(SAGEMAKER_ARN_RE.match(value))


def validate_s3_uri_prefix(s3_uri: str) -> None:
    """Validation method that checks string is an S3 URI that is a prefix.

    Raises:
        ValueError: If validation fails
    """
    if not S3_URI_PREFIX_REGEX.match(s3_uri):
        raise ValueError(f"S3 URI must fit pattern {S3_URI_PREFIX_REGEX.pattern}")


def validate_endpoint_arn(endpoint_arn: str) -> None:
    """Validation method that checks endpoint arn is either a bedrock or sagemaker endpoint.

    Raises:
        ValueError: If validation fails
    """
    if not (
        SAGEMAKER_ENDPOINT_ARN_REGEX.match(endpoint_arn)
        or BEDROCK_DEPLOYMENT_ARN_REGEX.match(endpoint_arn)
    ):
        raise ValueError(
            f"Endpoint must fit either SageMaker Endpoint pattern "
            f"{SAGEMAKER_ENDPOINT_ARN_REGEX.pattern} or Bedrock Deployment "
            f"pattern {BEDROCK_DEPLOYMENT_ARN_REGEX.pattern}"
        )


def validate_unit_count(unit_count: Optional[int]) -> None:
    """Validation method that checks unit count is not None and is >= 1.

    Raises:
        ValueError: If validation fails
    """
    if unit_count is None or unit_count < 1:
        raise ValueError("unit_count must be a positive integer value")
