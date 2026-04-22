import unittest

from amzn_nova_forge.core.enums import Model
from amzn_nova_forge.validation.endpoint_validator import (
    ENV_CONTEXT_LENGTH,
    ENV_DEFAULT_LOGPROBS,
    ENV_DEFAULT_MAX_NEW_TOKENS,
    ENV_DEFAULT_TEMPERATURE,
    ENV_DEFAULT_TOP_K,
    ENV_DEFAULT_TOP_P,
    ENV_DISABLE_SPECULATIVE_DECODING,
    ENV_MAX_CONCURRENCY,
    ENV_SPECULATIVE_DECODING_METHOD,
    ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS,
    ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR,
    ENV_SUFFIX_DECODING_MAX_TREE_DEPTH,
    ENV_SUFFIX_DECODING_MIN_TOKEN_PROB,
    is_sagemaker_arn,
    validate_endpoint_arn,
    validate_s3_uri_prefix,
    validate_sagemaker_environment_variables,
    validate_unit_count,
)


class TestEndpointValidator(unittest.TestCase):
    def test_validate_s3_uri_prefix_raise_exception(self):
        with self.assertRaises(ValueError) as context:
            validate_s3_uri_prefix("s3://sthree-doesntendwithadash")

    def test_validate_s3_uri_prefix_does_not_raise_exception(self):
        validate_s3_uri_prefix("s3://sthree-validname/")

    def test_valid_environment_variables(self):
        """
        Test that a valid set of environment variables passes validation,
        including speculative decoding parameters.
        """
        valid_env_vars = {
            ENV_CONTEXT_LENGTH: "100",
            ENV_MAX_CONCURRENCY: "10",
            ENV_DEFAULT_TEMPERATURE: "0.7",
            ENV_DEFAULT_TOP_P: "0.9",
            ENV_SPECULATIVE_DECODING_METHOD: "suffix",
            ENV_DISABLE_SPECULATIVE_DECODING: "false",
            ENV_SUFFIX_DECODING_MAX_TREE_DEPTH: "24",
            ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS: "10000",
            ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR: "1.0",
            ENV_SUFFIX_DECODING_MIN_TOKEN_PROB: "0.1",
        }

        try:
            validate_sagemaker_environment_variables(valid_env_vars)
        except ValueError:
            self.fail("Valid environment variables raised unexpected ValueError")

    def test_missing_required_keys(self):
        """
        Test that missing required keys raises a ValueError
        """
        incomplete_env_vars = {ENV_CONTEXT_LENGTH: "100", ENV_DEFAULT_TEMPERATURE: "0.7"}

        with self.assertRaises(ValueError):
            validate_sagemaker_environment_variables(incomplete_env_vars)

    def test_invalid_value_ranges(self):
        """
        Test various invalid value scenarios that should raise ValueError,
        including speculative decoding parameters.
        """
        test_cases = [
            # Negative CONTEXT_LENGTH
            {ENV_CONTEXT_LENGTH: "-10", ENV_MAX_CONCURRENCY: "10"},
            # Non-integer MAX_CONCURRENCY
            {ENV_CONTEXT_LENGTH: "100", ENV_MAX_CONCURRENCY: "10.5"},
            # Temperature out of range
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_DEFAULT_TEMPERATURE: "101",
            },
            # Invalid DEFAULT_TOP_P value
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_DEFAULT_TOP_P: "0.00000000001",
            },
            # Invalid speculative decoding method
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SPECULATIVE_DECODING_METHOD: "invalid_method",
            },
            # Invalid disable speculative decoding value
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_DISABLE_SPECULATIVE_DECODING: "yes",
            },
            # Non-positive SUFFIX_DECODING_MAX_TREE_DEPTH
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_TREE_DEPTH: "0",
            },
            # Negative SUFFIX_DECODING_MAX_CACHED_REQUESTS
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_CACHED_REQUESTS: "-1",
            },
            # Negative SUFFIX_DECODING_MAX_SPEC_FACTOR
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MAX_SPEC_FACTOR: "-0.5",
            },
            # SUFFIX_DECODING_MIN_TOKEN_PROB above 1.0
            {
                ENV_CONTEXT_LENGTH: "100",
                ENV_MAX_CONCURRENCY: "10",
                ENV_SUFFIX_DECODING_MIN_TOKEN_PROB: "1.5",
            },
        ]

        for invalid_env_vars in test_cases:
            with self.assertRaises(ValueError):
                validate_sagemaker_environment_variables(invalid_env_vars)

    def test_validate_endpoint_arn_invalid_arn(self):
        with self.assertRaises(ValueError):
            validate_endpoint_arn("bad_arn")

    def test_validate_endpoint_arn_valid_arn(self):
        validate_endpoint_arn("arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint")
        validate_endpoint_arn(
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/model"
        )

    def test_validate_unit_count_good_value(self):
        try:
            validate_unit_count(1)
        except ValueError:
            self.fail("Valid unit count raised unexpected ValueError")

    def test_validate_unit_count_bad_values(self):
        with self.assertRaises(ValueError):
            validate_unit_count(None)

        with self.assertRaises(ValueError):
            validate_unit_count(0)

    # --- SMI config bounds tests ---

    def test_smi_valid_within_tier(self):
        """Valid: context_length and concurrency within a known tier."""
        env = {ENV_CONTEXT_LENGTH: "4000", ENV_MAX_CONCURRENCY: "12"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_valid_lower_context_same_concurrency(self):
        """Valid: lower context length inherits the tier's max concurrency."""
        env = {ENV_CONTEXT_LENGTH: "2000", ENV_MAX_CONCURRENCY: "12"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_valid_lower_concurrency(self):
        """Valid: concurrency below the tier max."""
        env = {ENV_CONTEXT_LENGTH: "8000", ENV_MAX_CONCURRENCY: "4"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_context_length_exceeds_max(self):
        """Invalid: context length exceeds the largest supported tier."""
        env = {ENV_CONTEXT_LENGTH: "10000", ENV_MAX_CONCURRENCY: "1"}
        with self.assertRaises(ValueError) as ctx:
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
            )
        self.assertIn("CONTEXT_LENGTH", str(ctx.exception))

    def test_smi_concurrency_exceeds_tier_max(self):
        """Invalid: concurrency exceeds the max for the applicable context tier."""
        env = {ENV_CONTEXT_LENGTH: "8000", ENV_MAX_CONCURRENCY: "32"}
        with self.assertRaises(ValueError) as ctx:
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
            )
        self.assertIn("MAX_CONCURRENCY", str(ctx.exception))

    def test_smi_p5_multi_tier(self):
        """Valid: p5.48xlarge supports multiple tiers."""
        # Tier 1: context<=16000, concurrency<=128
        validate_sagemaker_environment_variables(
            {ENV_CONTEXT_LENGTH: "16000", ENV_MAX_CONCURRENCY: "128"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )
        # Tier 2: context<=64000, concurrency<=32
        validate_sagemaker_environment_variables(
            {ENV_CONTEXT_LENGTH: "64000", ENV_MAX_CONCURRENCY: "32"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )
        # Tier 3: context<=128000, concurrency<=8
        validate_sagemaker_environment_variables(
            {ENV_CONTEXT_LENGTH: "128000", ENV_MAX_CONCURRENCY: "8"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )

    def test_smi_p5_concurrency_exceeds_mid_tier(self):
        """Invalid: concurrency 50 at context 20000 falls in tier <=64000 which allows max 32."""
        env = {ENV_CONTEXT_LENGTH: "20000", ENV_MAX_CONCURRENCY: "50"}
        with self.assertRaises(ValueError):
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.p5.48xlarge"
            )

    def test_smi_unknown_combo_warns_but_passes(self):
        """Unknown model+instance combo should not raise (just warns)."""
        env = {ENV_CONTEXT_LENGTH: "100000", ENV_MAX_CONCURRENCY: "999"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_PRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_no_model_skips_bounds_check(self):
        """Without model/instance_type, no bounds check (backward compat)."""
        env = {ENV_CONTEXT_LENGTH: "100000", ENV_MAX_CONCURRENCY: "999"}
        validate_sagemaker_environment_variables(env)

    def test_is_sagemaker_arn_standard_partition(self):
        self.assertTrue(is_sagemaker_arn("arn:aws:sagemaker:us-east-1:123:model-package/group/1"))

    def test_is_sagemaker_arn_govcloud(self):
        self.assertTrue(
            is_sagemaker_arn("arn:aws-us-gov:sagemaker:us-gov-east-1:123:model-package/group/1")
        )

    def test_is_sagemaker_arn_china(self):
        self.assertTrue(
            is_sagemaker_arn("arn:aws-cn:sagemaker:cn-north-1:123:model-package/group/1")
        )

    def test_is_sagemaker_arn_rejects_s3_path(self):
        self.assertFalse(is_sagemaker_arn("s3://bucket/checkpoint/"))

    def test_is_sagemaker_arn_rejects_empty_string(self):
        self.assertFalse(is_sagemaker_arn(""))

    def test_is_sagemaker_arn_rejects_bedrock_arn(self):
        self.assertFalse(
            is_sagemaker_arn("arn:aws:bedrock:us-east-1:123:model-customization-job/abc")
        )


if __name__ == "__main__":
    unittest.main()
