import unittest

from amzn_nova_customization_sdk.validation.endpoint_validator import (
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


if __name__ == "__main__":
    unittest.main()
