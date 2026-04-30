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
import unittest

from pydantic import ValidationError

from amzn_nova_forge.core.enums import Model
from amzn_nova_forge.validation.endpoint_validator import (
    ENV_CONTEXT_LENGTH,
    ENV_DEFAULT_LOGPROBS,
    ENV_DEFAULT_MAX_NEW_TOKENS,
    ENV_DEFAULT_TEMPERATURE,
    ENV_DEFAULT_TOP_K,
    ENV_DEFAULT_TOP_P,
    ENV_DISABLE_SPECULATIVE_DECODING,
    ENV_KV_CACHE_DTYPE,
    ENV_MAX_CONCURRENCY,
    ENV_NUM_SPECULATIVE_TOKENS,
    ENV_QUANTIZATION_DTYPE,
    ENV_SPECULATIVE_DECODING_METHOD,
    ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS,
    ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR,
    ENV_SUFFIX_DECODING_MAX_TREE_DEPTH,
    ENV_SUFFIX_DECODING_MIN_TOKEN_PROB,
    SageMakerEndpointEnvironment,
    is_sagemaker_arn,
    validate_endpoint_arn,
    validate_s3_uri_prefix,
    validate_unit_count,
)


class TestEndpointValidator(unittest.TestCase):
    def test_validate_s3_uri_prefix_raise_exception(self):
        with self.assertRaises(ValueError):
            validate_s3_uri_prefix("s3://sthree-doesntendwithadash")

    def test_validate_s3_uri_prefix_does_not_raise_exception(self):
        validate_s3_uri_prefix("s3://sthree-validname/")

    def test_valid_environment_variables(self):
        """Valid env vars including all optional fields."""
        valid_env_vars = {
            ENV_CONTEXT_LENGTH: "100",
            ENV_MAX_CONCURRENCY: "10",
            ENV_DEFAULT_TEMPERATURE: "0.7",
            ENV_DEFAULT_TOP_P: "0.9",
            ENV_SPECULATIVE_DECODING_METHOD: "suffix",
            ENV_DISABLE_SPECULATIVE_DECODING: "false",
            ENV_NUM_SPECULATIVE_TOKENS: "5",
            ENV_SUFFIX_DECODING_MAX_TREE_DEPTH: "24",
            ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS: "10000",
            ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR: "1.0",
            ENV_SUFFIX_DECODING_MIN_TOKEN_PROB: "0.1",
            ENV_KV_CACHE_DTYPE: "fp8",
            ENV_QUANTIZATION_DTYPE: "fp8",
        }

        SageMakerEndpointEnvironment.from_env_dict(valid_env_vars)

    def test_missing_required_keys_uses_defaults(self):
        """Omitting context_length and max_concurrency uses defaults."""
        env = SageMakerEndpointEnvironment.from_env_dict({ENV_DEFAULT_TEMPERATURE: "0.7"})
        self.assertEqual(env.context_length, 4000)
        self.assertEqual(env.max_concurrency, 1)
        self.assertEqual(env.default_temperature, 0.7)

    def test_unknown_key_raises_value_error(self):
        """Unknown environment variable keys raise ValueError."""
        env = {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", "BOGUS_KEY": "x"}
        with self.assertRaises(ValueError):
            SageMakerEndpointEnvironment.from_env_dict(env)

    def test_invalid_value_ranges(self):
        """Various invalid values raise ValidationError."""
        test_cases = [
            {ENV_CONTEXT_LENGTH: "-10", ENV_MAX_CONCURRENCY: "10"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10.5"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_TEMPERATURE: "101"},
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_DEFAULT_TOP_P: "0.00000000001",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SPECULATIVE_DECODING_METHOD: "invalid_method",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_DISABLE_SPECULATIVE_DECODING: "yes",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_TREE_DEPTH: "0",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS: "-1",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR: "-0.5",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MIN_TOKEN_PROB: "1.5",
            },
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_KV_CACHE_DTYPE: "bf16"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_QUANTIZATION_DTYPE: "int8"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_NUM_SPECULATIVE_TOKENS: "0"},
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_NUM_SPECULATIVE_TOKENS: "11",
            },
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_NUM_SPECULATIVE_TOKENS: "3.5",
            },
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_LOGPROBS: "0"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_LOGPROBS: "21"},
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_TOP_K: "-2"},
        ]

        for invalid_env_vars in test_cases:
            with self.assertRaises(
                ValidationError,
                msg=f"Expected ValidationError for {invalid_env_vars}",
            ):
                SageMakerEndpointEnvironment.from_env_dict(invalid_env_vars)

    def test_validate_endpoint_arn_invalid_arn(self):
        with self.assertRaises(ValueError):
            validate_endpoint_arn("bad_arn")

    def test_validate_endpoint_arn_valid_arn(self):
        validate_endpoint_arn("arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint")
        validate_endpoint_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/model"
        )

    def test_validate_unit_count_good_value(self):
        validate_unit_count(1)

    def test_validate_unit_count_bad_values(self):
        with self.assertRaises(ValueError):
            validate_unit_count(None)
        with self.assertRaises(ValueError):
            validate_unit_count(0)

    def test_valid_top_k_minus_one(self):
        """top_k=-1 disables top-k filtering."""
        env = SageMakerEndpointEnvironment.from_env_dict(
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_TOP_K: "-1"}
        )
        self.assertEqual(env.default_top_k, -1)

    def test_valid_top_k_zero_rejected(self):
        """top_k=0 is not allowed (must be -1 or >= 1)."""
        with self.assertRaises(ValidationError):
            SageMakerEndpointEnvironment.from_env_dict(
                {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_TOP_K: "0"}
            )

    def test_valid_logprobs_boundaries(self):
        for val in ["1", "20"]:
            env = SageMakerEndpointEnvironment.from_env_dict(
                {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_LOGPROBS: val}
            )
            self.assertEqual(env.default_logprobs, int(val))

    def test_valid_num_speculative_tokens_boundaries(self):
        for val in ["1", "10"]:
            env = SageMakerEndpointEnvironment.from_env_dict(
                {
                    ENV_CONTEXT_LENGTH: "100",
                    ENV_MAX_CONCURRENCY: "10",
                    ENV_NUM_SPECULATIVE_TOKENS: val,
                }
            )
            self.assertEqual(env.num_speculative_tokens, int(val))

    def test_valid_kv_cache_dtype_fp8(self):
        env = SageMakerEndpointEnvironment.from_env_dict(
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_KV_CACHE_DTYPE: "fp8"}
        )
        self.assertEqual(env.kv_cache_dtype, "fp8")

    def test_valid_quantization_dtype_fp8(self):
        env = SageMakerEndpointEnvironment.from_env_dict(
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_QUANTIZATION_DTYPE: "fp8"}
        )
        self.assertEqual(env.quantization_dtype, "fp8")

    def test_literal_string_fields_case_insensitive(self):
        """Mixed-case values are normalised to lowercase for all Literal[str] fields."""
        for raw, field, env_key in [
            ("True", "disable_speculative_decoding", ENV_DISABLE_SPECULATIVE_DECODING),
            ("FALSE", "disable_speculative_decoding", ENV_DISABLE_SPECULATIVE_DECODING),
            ("Eagle3", "speculative_decoding_method", ENV_SPECULATIVE_DECODING_METHOD),
            ("SUFFIX", "speculative_decoding_method", ENV_SPECULATIVE_DECODING_METHOD),
            ("FP8", "kv_cache_dtype", ENV_KV_CACHE_DTYPE),
            ("Fp8", "quantization_dtype", ENV_QUANTIZATION_DTYPE),
        ]:
            env = SageMakerEndpointEnvironment.from_env_dict(
                {
                    ENV_CONTEXT_LENGTH: "100",
                    ENV_MAX_CONCURRENCY: "10",
                    env_key: raw,
                }
            )
            self.assertEqual(getattr(env, field), raw.lower(), f"Failed for {env_key}={raw!r}")

    def test_pydantic_model_basic_construction(self):
        env = SageMakerEndpointEnvironment(context_length=8000, max_concurrency=4)
        self.assertEqual(env.context_length, 8000)
        self.assertEqual(env.max_concurrency, 4)
        self.assertIsNone(env.default_temperature)

    def test_pydantic_model_to_env_dict(self):
        env = SageMakerEndpointEnvironment(
            context_length=8000, max_concurrency=4, default_temperature=0.7, kv_cache_dtype="fp8"
        )
        d = env.to_env_dict()
        self.assertEqual(d[ENV_CONTEXT_LENGTH], "8000")
        self.assertEqual(d[ENV_MAX_CONCURRENCY], "4")
        self.assertEqual(d[ENV_DEFAULT_TEMPERATURE], "0.7")
        self.assertEqual(d[ENV_KV_CACHE_DTYPE], "fp8")
        self.assertNotIn(ENV_DEFAULT_TOP_P, d)

    def test_pydantic_model_from_env_dict_roundtrip(self):
        original = {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10", ENV_DEFAULT_TOP_K: "-1"}
        env = SageMakerEndpointEnvironment.from_env_dict(original)
        self.assertEqual(env.default_top_k, -1)
        self.assertEqual(env.to_env_dict(), original)

    def test_pydantic_model_string_coercion(self):
        env = SageMakerEndpointEnvironment(
            context_length="8000",
            max_concurrency="4",  # type: ignore[arg-type]
        )
        self.assertEqual(env.context_length, 8000)
        self.assertEqual(env.max_concurrency, 4)

    def test_pydantic_model_defaults(self):
        """Constructing with no arguments uses defaults."""
        env = SageMakerEndpointEnvironment()
        self.assertEqual(env.context_length, 4000)
        self.assertEqual(env.max_concurrency, 1)

    def test_pydantic_model_invalid_literal_raises(self):
        with self.assertRaises(ValidationError):
            SageMakerEndpointEnvironment(
                context_length=100,
                max_concurrency=10,
                kv_cache_dtype="bf16",  # type: ignore[arg-type]
            )

    def _make_env(self, env_vars):
        return SageMakerEndpointEnvironment.from_env_dict(env_vars)

    def test_smi_valid_within_tier(self):
        self._make_env(
            {ENV_CONTEXT_LENGTH: "4000", ENV_MAX_CONCURRENCY: "12"}
        ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge")

    def test_smi_valid_lower_context_same_concurrency(self):
        self._make_env(
            {ENV_CONTEXT_LENGTH: "2000", ENV_MAX_CONCURRENCY: "12"}
        ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge")

    def test_smi_valid_lower_concurrency(self):
        self._make_env(
            {ENV_CONTEXT_LENGTH: "8000", ENV_MAX_CONCURRENCY: "4"}
        ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge")

    def test_smi_context_length_exceeds_max(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_env(
                {ENV_CONTEXT_LENGTH: "10000", ENV_MAX_CONCURRENCY: "1"}
            ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge")
        self.assertIn("CONTEXT_LENGTH", str(ctx.exception))

    def test_smi_concurrency_exceeds_tier_max(self):
        with self.assertRaises(ValueError) as ctx:
            self._make_env(
                {ENV_CONTEXT_LENGTH: "8000", ENV_MAX_CONCURRENCY: "32"}
            ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge")
        self.assertIn("MAX_CONCURRENCY", str(ctx.exception))

    def test_smi_p5_multi_tier(self):
        for ctx_len, conc in [("16000", "128"), ("64000", "32"), ("128000", "8")]:
            self._make_env(
                {ENV_CONTEXT_LENGTH: ctx_len, ENV_MAX_CONCURRENCY: conc}
            ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.p5.48xlarge")

    def test_smi_p5_concurrency_exceeds_mid_tier(self):
        with self.assertRaises(ValueError):
            self._make_env(
                {ENV_CONTEXT_LENGTH: "20000", ENV_MAX_CONCURRENCY: "50"}
            ).validate_smi_config_bounds(model=Model.NOVA_MICRO, instance_type="ml.p5.48xlarge")

    def test_smi_unknown_combo_warns_but_passes(self):
        self._make_env(
            {ENV_CONTEXT_LENGTH: "100000", ENV_MAX_CONCURRENCY: "999"}
        ).validate_smi_config_bounds(model=Model.NOVA_PRO, instance_type="ml.g5.12xlarge")

    def test_smi_no_model_no_bounds_check(self):
        """Without calling validate_smi_config_bounds, no bounds check."""
        SageMakerEndpointEnvironment.from_env_dict(
            {ENV_CONTEXT_LENGTH: "100000", ENV_MAX_CONCURRENCY: "999"}
        )

    def test_is_sagemaker_arn_standard_partition(self):
        self.assertTrue(
            is_sagemaker_arn("arn:aws:sagemaker:us-east-1:123456789012:model-package/group/1")
        )

    def test_is_sagemaker_arn_govcloud(self):
        self.assertTrue(
            is_sagemaker_arn(
                "arn:aws-us-gov:sagemaker:us-gov-east-1:123456789012:model-package/group/1"
            )
        )

    def test_is_sagemaker_arn_china(self):
        self.assertTrue(
            is_sagemaker_arn("arn:aws-cn:sagemaker:cn-north-1:123456789012:model-package/group/1")
        )

    def test_is_sagemaker_arn_rejects_s3_path(self):
        self.assertFalse(is_sagemaker_arn("s3://bucket/checkpoint/"))

    def test_is_sagemaker_arn_rejects_empty_string(self):
        self.assertFalse(is_sagemaker_arn(""))

    def test_is_sagemaker_arn_rejects_bedrock_arn(self):
        self.assertFalse(
            is_sagemaker_arn("arn:aws:bedrock:us-east-1:123456789012:model-customization-job/abc")
        )


if __name__ == "__main__":
    unittest.main()
