import unittest

from amzn_nova_forge_sdk.model.model_enums import Model
from amzn_nova_forge_sdk.validation.endpoint_validator import (
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
        Test that a valid set of environment variables passes validation
        """
        valid_env_vars = {
            "CONTEXT_LENGTH": "100",
            "MAX_CONCURRENCY": "10",
            "DEFAULT_TEMPERATURE": "0.7",
            "DEFAULT_TOP_P": "0.9",
        }

        try:
            validate_sagemaker_environment_variables(valid_env_vars)
        except ValueError:
            self.fail("Valid environment variables raised unexpected ValueError")

    def test_missing_required_keys(self):
        """
        Test that missing required keys raises a ValueError
        """
        incomplete_env_vars = {"CONTEXT_LENGTH": "100", "DEFAULT_TEMPERATURE": "0.7"}

        with self.assertRaises(ValueError):
            validate_sagemaker_environment_variables(incomplete_env_vars)

    def test_invalid_value_ranges(self):
        """
        Test various invalid value scenarios that should raise ValueError
        """
        test_cases = [
            # Negative CONTEXT_LENGTH
            {"CONTEXT_LENGTH": "-10", "MAX_CONCURRENCY": "10"},
            # Non-integer MAX_CONCURRENCY
            {"CONTEXT_LENGTH": "100", "MAX_CONCURRENCY": "10.5"},
            # Temperature out of range
            {
                "CONTEXT_LENGTH": "100",
                "MAX_CONCURRENCY": "10",
                "DEFAULT_TEMPERATURE": "101",
            },
            # Invalid DEFAULT_TOP_P value
            {
                "CONTEXT_LENGTH": "100",
                "MAX_CONCURRENCY": "10",
                "DEFAULT_TOP_P": "0.00000000001",
            },
        ]

        for invalid_env_vars in test_cases:
            with self.assertRaises(ValueError):
                validate_sagemaker_environment_variables(invalid_env_vars)

    def test_validate_endpoint_arn_invalid_arn(self):
        with self.assertRaises(ValueError):
            validate_endpoint_arn("bad_arn")

    def test_validate_endpoint_arn_valid_arn(self):
        validate_endpoint_arn(
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/endpoint"
        )
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
        env = {"CONTEXT_LENGTH": "4000", "MAX_CONCURRENCY": "32"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_valid_lower_context_same_concurrency(self):
        """Valid: lower context length inherits the tier's max concurrency."""
        env = {"CONTEXT_LENGTH": "2000", "MAX_CONCURRENCY": "32"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_valid_lower_concurrency(self):
        """Valid: concurrency below the tier max."""
        env = {"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "4"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_context_length_exceeds_max(self):
        """Invalid: context length exceeds the largest supported tier."""
        env = {"CONTEXT_LENGTH": "10000", "MAX_CONCURRENCY": "1"}
        with self.assertRaises(ValueError) as ctx:
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
            )
        self.assertIn("CONTEXT_LENGTH", str(ctx.exception))

    def test_smi_concurrency_exceeds_tier_max(self):
        """Invalid: concurrency exceeds the max for the applicable context tier."""
        env = {"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "32"}
        with self.assertRaises(ValueError) as ctx:
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.g5.12xlarge"
            )
        self.assertIn("MAX_CONCURRENCY", str(ctx.exception))

    def test_smi_p5_multi_tier(self):
        """Valid: p5.48xlarge supports multiple tiers."""
        # Tier 1: context<=8000, concurrency<=32
        validate_sagemaker_environment_variables(
            {"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "32"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )
        # Tier 2: context<=16000, concurrency<=2
        validate_sagemaker_environment_variables(
            {"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "2"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )
        # Tier 3: context<=24000, concurrency<=1
        validate_sagemaker_environment_variables(
            {"CONTEXT_LENGTH": "24000", "MAX_CONCURRENCY": "1"},
            model=Model.NOVA_MICRO,
            instance_type="ml.p5.48xlarge",
        )

    def test_smi_p5_concurrency_exceeds_mid_tier(self):
        """Invalid: concurrency 10 at context 12000 falls in tier <=16000 which allows max 2."""
        env = {"CONTEXT_LENGTH": "12000", "MAX_CONCURRENCY": "10"}
        with self.assertRaises(ValueError):
            validate_sagemaker_environment_variables(
                env, model=Model.NOVA_MICRO, instance_type="ml.p5.48xlarge"
            )

    def test_smi_unknown_combo_warns_but_passes(self):
        """Unknown model+instance combo should not raise (just warns)."""
        env = {"CONTEXT_LENGTH": "100000", "MAX_CONCURRENCY": "999"}
        validate_sagemaker_environment_variables(
            env, model=Model.NOVA_PRO, instance_type="ml.g5.12xlarge"
        )

    def test_smi_no_model_skips_bounds_check(self):
        """Without model/instance_type, no bounds check (backward compat)."""
        env = {"CONTEXT_LENGTH": "100000", "MAX_CONCURRENCY": "999"}
        validate_sagemaker_environment_variables(env)


if __name__ == "__main__":
    unittest.main()
