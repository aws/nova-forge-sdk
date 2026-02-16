"""Unit tests for RFT multiturn validation utilities."""

import pytest

from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
    validate_dict_values,
    validate_env_id,
    validate_path,
    validate_region,
    validate_stack_name,
    validate_url,
)


class TestValidateEnvId:
    """Test validate_env_id function."""

    def test_accepts_valid_alphanumeric_ids(self):
        """Test that valid alphanumeric environment IDs are accepted."""
        valid_ids = [
            "wordle",
            "test123",
            "env_with_underscore",
            "env-with-dash",
            "MixedCase123",
            "a",  # Single character
            "very_long_environment_id_with_many_characters_123",
        ]
        for env_id in valid_ids:
            validate_env_id(env_id)  # Should not raise

    def test_accepts_underscores_and_hyphens(self):
        """Test that underscores and hyphens are allowed."""
        valid_ids = [
            "test_env",
            "test-env",
            "test_env-123",
            "my-custom_env",
            "___",
            "---",
        ]
        for env_id in valid_ids:
            validate_env_id(env_id)  # Should not raise

    def test_rejects_spaces(self):
        """Test that environment IDs with spaces are rejected."""
        invalid_ids = [
            "env with spaces",
            "test env",
            " leading_space",
            "trailing_space ",
            "middle space",
        ]
        for env_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid environment ID"):
                validate_env_id(env_id)

    def test_rejects_shell_metacharacters(self):
        """Test that shell metacharacters are rejected."""
        invalid_ids = [
            "env;rm -rf /",
            "env$(whoami)",
            "env`ls`",
            "env|cat",
            "env&echo",
            "env>file",
            "env<file",
            "env'test",
            'env"test',
            "env\\test",
            "env/test",
            "env*test",
            "env?test",
            "env[test]",
            "env{test}",
            "env(test)",
            "env$test",
            "env!test",
            "env#test",
            "env%test",
            "env^test",
            "env@test",
            "env~test",
            "env+test",
            "env=test",
        ]
        for env_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid environment ID"):
                validate_env_id(env_id)

    def test_rejects_command_injection_attempts(self):
        """Test that command injection attempts are rejected."""
        injection_attempts = [
            "; rm -rf /",
            "$(malicious_command)",
            "`malicious_command`",
            "| cat /etc/passwd",
            "&& echo hacked",
            "|| echo hacked",
            "; echo hacked",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_env_id(attempt)


class TestValidatePath:
    """Test validate_path function."""

    def test_accepts_valid_absolute_paths(self):
        """Test that valid absolute paths are accepted."""
        valid_paths = [
            "/home/user/project",
            "/tmp/test_file.txt",
            "/var/log/app.log",
            "/root/v1/starter-kit",
            "/test/path_with-dots.tar.gz",
            "/a",
            "/path/to/file.json",
            "/usr/local/bin/python3.12",
        ]
        for path in valid_paths:
            validate_path(path)  # Should not raise

    def test_accepts_relative_paths(self):
        """Test that valid relative paths are accepted."""
        valid_paths = [
            "relative/path",
            "file.txt",
            "path/to/file",
            "test_file-123.tar.gz",
        ]
        for path in valid_paths:
            validate_path(path)  # Should not raise

    def test_accepts_paths_with_dots_and_dashes(self):
        """Test that paths with dots and dashes are accepted."""
        valid_paths = [
            "/path/file.tar.gz",
            "/path/my-file.txt",
            "/path/file_name.json",
            "/path/version-1.0.0.tar.gz",
        ]
        for path in valid_paths:
            validate_path(path)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValueError, match="Path must be a non-empty string"):
            validate_path("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        invalid_paths = [None, 123, [], {}, True]
        for path in invalid_paths:
            with pytest.raises(ValueError, match="Path must be a non-empty string"):
                validate_path(path)

    def test_rejects_path_traversal(self):
        """Test that path traversal attempts are rejected."""
        invalid_paths = [
            "/path/../etc/passwd",
            "../etc/passwd",
            "/home/user/../../root",
            "path/../../../etc",
            "/path/to/../../../sensitive",
        ]
        for path in invalid_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_path(path)

    def test_rejects_spaces(self):
        """Test that paths with spaces are rejected."""
        invalid_paths = [
            "/path with spaces",
            "/path/to/file name.txt",
            " /path",
            "/path ",
        ]
        for path in invalid_paths:
            with pytest.raises(ValueError, match="Invalid path"):
                validate_path(path)

    def test_rejects_shell_metacharacters(self):
        """Test that shell metacharacters in paths are rejected."""
        invalid_paths = [
            "/path;rm -rf /",
            "/path$(whoami)",
            "/path`ls`",
            "/path|cat",
            "/path&echo",
            "/path>file",
            "/path<file",
            "/path'test",
            '/path"test',
            "/path\\test",
            "/path*test",
            "/path?test",
            "/path[test]",
            "/path{test}",
            "/path(test)",
            "/path$test",
            "/path!test",
            "/path#test",
            "/path%test",
            "/path^test",
            "/path@test",
            "/path~test",
            "/path+test",
            "/path=test",
        ]
        for path in invalid_paths:
            with pytest.raises(ValueError, match="Invalid path"):
                validate_path(path)


class TestValidateUrl:
    """Test validate_url function."""

    def test_accepts_valid_https_urls(self):
        """Test that valid HTTPS URLs are accepted."""
        valid_urls = [
            "https://example.com/path",
            "https://test.amazonaws.com",
            "https://lambda.us-west-2.amazonaws.com",
            "https://api.example.com/v1/endpoint",
            "https://subdomain.example.com/path/to/resource",
        ]
        for url in valid_urls:
            validate_url(url)  # Should not raise

    def test_accepts_valid_http_urls(self):
        """Test that valid HTTP URLs are accepted."""
        valid_urls = [
            "http://example.com/path",
            "http://test.amazonaws.com",
            "http://localhost/api",
        ]
        for url in valid_urls:
            validate_url(url)  # Should not raise

    def test_accepts_valid_s3_urls(self):
        """Test that valid S3 URLs are accepted."""
        valid_urls = [
            "s3://bucket-name/key",
            "s3://my-bucket/path/to/file.tar.gz",
            "s3://bucket/prefix/object",
        ]
        for url in valid_urls:
            validate_url(url)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValueError, match="URL must be a non-empty string"):
            validate_url("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        invalid_urls = [123, [], {}, True]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="URL must be a non-empty string"):
                validate_url(url)

        # None is handled separately - raises "is required" when required=True
        with pytest.raises(ValueError, match="URL is required"):
            validate_url(None)

    def test_rejects_invalid_protocols(self):
        """Test that invalid protocols are rejected."""
        invalid_urls = [
            "ftp://invalid.com",
            "file:///path/to/file",
            "ssh://server.com",
            "telnet://server.com",
            "javascript:alert(1)",
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid URL"):
                validate_url(url)

    def test_rejects_urls_without_protocol(self):
        """Test that URLs without protocol are rejected."""
        invalid_urls = [
            "example.com",
            "www.example.com",
            "//example.com",
            "example.com/path",
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid URL"):
                validate_url(url)

    def test_rejects_shell_injection_in_urls(self):
        """Test that shell injection attempts in URLs are rejected."""
        invalid_urls = [
            "https://test.com; rm -rf /",
            "http://test$(whoami).com",
            "s3://bucket`ls`/key",
            "https://test.com|cat",
            "http://test.com&echo",
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid URL"):
                validate_url(url)

    def test_custom_url_type_in_error_message(self):
        """Test that custom URL type appears in error message."""
        with pytest.raises(ValueError, match="Lambda URL"):
            validate_url("invalid", "Lambda URL")

        with pytest.raises(ValueError, match="Queue URL"):
            validate_url("", "Queue URL")


class TestValidateStackName:
    """Test validate_stack_name function."""

    def test_accepts_valid_stack_names(self):
        """Test that valid CloudFormation stack names are accepted."""
        valid_names = [
            "my-stack",
            "TestStack",
            "stack-123",
            "MyApp-Production",
            "Stack",
            "a",
            "MyVeryLongStackNameWithManyCharacters123",
        ]
        for name in valid_names:
            validate_stack_name(name)  # Should not raise

    def test_accepts_alphanumeric_and_hyphens(self):
        """Test that alphanumeric characters and hyphens are accepted."""
        valid_names = [
            "stack-with-hyphens",
            "StackWithNumbers123",
            "UPPERCASE-STACK",
            "lowercase-stack",
            "MixedCase-Stack-123",
        ]
        for name in valid_names:
            validate_stack_name(name)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValueError, match="Stack name must be a non-empty string"):
            validate_stack_name("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        invalid_names = [None, 123, [], {}, True]
        for name in invalid_names:
            with pytest.raises(
                ValueError, match="Stack name must be a non-empty string"
            ):
                validate_stack_name(name)

    def test_rejects_names_starting_with_number(self):
        """Test that stack names starting with numbers are rejected."""
        invalid_names = [
            "123-stack",
            "1stack",
            "9MyStack",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid stack name"):
                validate_stack_name(name)

    def test_rejects_names_starting_with_hyphen(self):
        """Test that stack names starting with hyphens are rejected."""
        invalid_names = [
            "-stack",
            "-MyStack",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid stack name"):
                validate_stack_name(name)

    def test_rejects_underscores(self):
        """Test that underscores are rejected."""
        invalid_names = [
            "stack_name",
            "my_stack",
            "Stack_With_Underscores",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid stack name"):
                validate_stack_name(name)

    def test_rejects_spaces(self):
        """Test that spaces are rejected."""
        invalid_names = [
            "stack name",
            "my stack",
            " stack",
            "stack ",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid stack name"):
                validate_stack_name(name)

    def test_rejects_special_characters(self):
        """Test that special characters are rejected."""
        invalid_names = [
            "stack;rm",
            "stack$test",
            "stack`ls`",
            "stack|cat",
            "stack&echo",
            "stack>file",
            "stack<file",
            "stack'test",
            'stack"test',
            "stack\\test",
            "stack/test",
            "stack*test",
            "stack?test",
            "stack[test]",
            "stack{test}",
            "stack(test)",
            "stack!test",
            "stack#test",
            "stack%test",
            "stack^test",
            "stack@test",
            "stack~test",
            "stack+test",
            "stack=test",
            "stack.test",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid stack name"):
                validate_stack_name(name)


class TestValidateRegion:
    """Test validate_region function."""

    def test_accepts_valid_aws_regions(self):
        """Test that valid AWS region names are accepted."""
        valid_regions = [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "eu-central-1",
            "ap-southeast-2",
            "ap-northeast-1",
            "sa-east-1",
            "ca-central-1",
            "af-south-1",
            "me-south-1",
        ]
        for region in valid_regions:
            validate_region(region)  # Should not raise

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValueError, match="Region must be a non-empty string"):
            validate_region("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        invalid_regions = [None, 123, [], {}, True]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Region must be a non-empty string"):
                validate_region(region)

    def test_rejects_invalid_format(self):
        """Test that invalid region formats are rejected."""
        invalid_regions = [
            "invalid-region",
            "us-east",  # Missing number
            "useast1",  # Missing hyphens
            "us_east_1",  # Wrong separator
            "us-1-east",  # Wrong order
            "1-east-us",  # Starts with number
            "east-us-1",  # Wrong order
        ]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Invalid AWS region"):
                validate_region(region)

    def test_rejects_uppercase(self):
        """Test that uppercase region names are rejected."""
        invalid_regions = [
            "US-EAST-1",
            "Us-East-1",
            "US-east-1",
        ]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Invalid AWS region"):
                validate_region(region)

    def test_rejects_spaces(self):
        """Test that regions with spaces are rejected."""
        invalid_regions = [
            "us east 1",
            " us-east-1",
            "us-east-1 ",
        ]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Invalid AWS region"):
                validate_region(region)

    def test_rejects_special_characters(self):
        """Test that special characters are rejected."""
        invalid_regions = [
            "us-east-1;rm",
            "us-east-1$(whoami)",
            "us-east-1`ls`",
            "us-east-1|cat",
        ]
        for region in invalid_regions:
            with pytest.raises(ValueError, match="Invalid AWS region"):
                validate_region(region)


class TestValidateDictValues:
    """Test validate_dict_values function."""

    def test_accepts_empty_dict(self):
        """Test that empty dictionary is accepted."""
        validate_dict_values({})  # Should not raise

    def test_accepts_simple_types(self):
        """Test that simple JSON-serializable types are accepted."""
        valid_dicts = [
            {"key": "value"},
            {"num": 123},
            {"float": 45.6},
            {"bool": True},
            {"none": None},
            {"false": False},
            {"zero": 0},
            {"empty_string": ""},
        ]
        for d in valid_dicts:
            validate_dict_values(d)  # Should not raise

    def test_accepts_mixed_types(self):
        """Test that dictionaries with mixed types are accepted."""
        valid_dict = {
            "string": "value",
            "number": 123,
            "float": 45.6,
            "bool": True,
            "none": None,
        }
        validate_dict_values(valid_dict)  # Should not raise

    def test_accepts_nested_dicts(self):
        """Test that nested dictionaries are accepted."""
        valid_dicts = [
            {"nested": {"key": "value"}},
            {"deep": {"nested": {"dict": {"value": 123}}}},
            {"mixed": {"string": "value", "number": 123}},
        ]
        for d in valid_dicts:
            validate_dict_values(d)  # Should not raise

    def test_accepts_lists(self):
        """Test that lists are accepted."""
        valid_dicts = [
            {"list": [1, 2, 3]},
            {"strings": ["a", "b", "c"]},
            {"mixed": [1, "two", 3.0, True, None]},
            {"empty": []},
        ]
        for d in valid_dicts:
            validate_dict_values(d)  # Should not raise

    def test_accepts_nested_lists_and_dicts(self):
        """Test that nested lists and dictionaries are accepted."""
        valid_dicts = [
            {"nested": [{"key": "value"}]},
            {"list_of_lists": [[1, 2], [3, 4]]},
            {"complex": {"nested": [1, "two", {"three": 3}]}},
        ]
        for d in valid_dicts:
            validate_dict_values(d)  # Should not raise

    def test_accepts_deeply_nested_structures(self):
        """Test that deeply nested structures are accepted (no depth limit)."""
        # Create a deeply nested structure (20 levels)
        deep_dict = {"level": 0}
        current = deep_dict
        for i in range(1, 20):
            current["nested"] = {"level": i}
            current = current["nested"]

        validate_dict_values(deep_dict)  # Should not raise

    def test_rejects_non_dict(self):
        """Test that non-dictionary types are rejected."""
        invalid_inputs = ["string", 123, [1, 2, 3], None, True]
        for inp in invalid_inputs:
            with pytest.raises(ValueError, match="must be a dictionary"):
                validate_dict_values(inp)

    def test_rejects_null_bytes_in_strings(self):
        """Test that strings with null bytes are rejected."""
        invalid_dicts = [
            {"key": "value\x00"},
            {"nested": {"key": "value\x00"}},
            {"list": ["value\x00"]},
        ]
        for d in invalid_dicts:
            with pytest.raises(ValueError, match="null bytes"):
                validate_dict_values(d)

    def test_rejects_non_string_keys(self):
        """Test that non-string dictionary keys are rejected."""
        invalid_dicts = [
            {123: "value"},
            {None: "value"},
            {True: "value"},
            {("tuple",): "value"},
        ]
        for d in invalid_dicts:
            with pytest.raises(ValueError, match="key must be string"):
                validate_dict_values(d)

    def test_rejects_non_string_keys_in_nested_dicts(self):
        """Test that non-string keys in nested dicts are rejected."""
        invalid_dict = {"valid": {"nested": {123: "value"}}}
        with pytest.raises(ValueError, match="key must be string"):
            validate_dict_values(invalid_dict)

    def test_rejects_unsupported_types(self):
        """Test that unsupported types are rejected."""
        invalid_dicts = [
            {"object": object()},
            {"function": lambda x: x},
            {"set": {1, 2, 3}},
            {"tuple": (1, 2, 3)},
            {"bytes": b"bytes"},
            {"complex": 1 + 2j},
        ]
        for d in invalid_dicts:
            with pytest.raises(ValueError, match="Unsupported type"):
                validate_dict_values(d)

    def test_rejects_unsupported_types_in_lists(self):
        """Test that unsupported types in lists are rejected."""
        invalid_dict = {"list": [1, 2, object()]}
        with pytest.raises(ValueError, match="Unsupported type"):
            validate_dict_values(invalid_dict)

    def test_rejects_unsupported_types_in_nested_dicts(self):
        """Test that unsupported types in nested dicts are rejected."""
        invalid_dict = {"nested": {"key": object()}}
        with pytest.raises(ValueError, match="Unsupported type"):
            validate_dict_values(invalid_dict)

    def test_custom_dict_name_in_error_message(self):
        """Test that custom dictionary name appears in error message."""
        with pytest.raises(ValueError, match="vf_env_args"):
            validate_dict_values("not a dict", "vf_env_args")

        with pytest.raises(ValueError, match="custom_dict"):
            validate_dict_values({123: "value"}, "custom_dict")

    def test_error_path_in_nested_structures(self):
        """Test that error messages include the path to the invalid value."""
        # Null byte in nested dict
        with pytest.raises(ValueError, match=r"\.nested\.key"):
            validate_dict_values({"nested": {"key": "value\x00"}})

        # Unsupported type in list
        with pytest.raises(ValueError, match=r"\[0\]"):
            validate_dict_values({"list": [object()]})

        # Non-string key in nested dict
        with pytest.raises(ValueError, match=r"\.nested"):
            validate_dict_values({"nested": {123: "value"}})

    def test_accepts_large_dictionaries(self):
        """Test that large dictionaries are accepted (no size limit)."""
        # Create a dictionary with 1000 keys
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        validate_dict_values(large_dict)  # Should not raise

    def test_accepts_long_strings(self):
        """Test that long strings are accepted (no length limit)."""
        long_string = "a" * 10000
        validate_dict_values({"key": long_string})  # Should not raise

    def test_accepts_large_lists(self):
        """Test that large lists are accepted (no size limit)."""
        large_list = list(range(1000))
        validate_dict_values({"list": large_list})  # Should not raise

    def test_accepts_large_numbers(self):
        """Test that large numbers are accepted (no range limit)."""
        valid_dict = {
            "large_int": 999999999999999999,
            "large_float": 1.7976931348623157e308,
            "negative": -999999999999999999,
        }
        validate_dict_values(valid_dict)  # Should not raise


class TestValidatePlatform:
    """Test validate_platform function."""

    def test_accepts_valid_platforms(self):
        """Test that valid platform values are accepted."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_platform,
        )

        valid_platforms = ["local", "ec2", "ecs"]
        for platform in valid_platforms:
            validate_platform(platform)  # Should not raise

    def test_rejects_invalid_platforms(self):
        """Test that invalid platform values are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_platform,
        )

        invalid_platforms = [
            "lambda",
            "fargate",
            "eks",
            "LOCAL",  # Wrong case
            "EC2",  # Wrong case
            "ECS",  # Wrong case
            "local ",  # Trailing space
            " local",  # Leading space
            "ec2-instance",
            "ecs-cluster",
        ]
        for platform in invalid_platforms:
            with pytest.raises(ValueError, match="Invalid platform"):
                validate_platform(platform)

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_platform,
        )

        with pytest.raises(ValueError, match="Platform must be a non-empty string"):
            validate_platform("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_platform,
        )

        invalid_platforms = [None, 123, [], {}, True]
        for platform in invalid_platforms:
            with pytest.raises(ValueError, match="Platform must be a non-empty string"):
                validate_platform(platform)


class TestValidateEc2InstanceIdentifier:
    """Test validate_ec2_instance_identifier function."""

    def test_accepts_valid_instance_ids(self):
        """Test that valid EC2 instance IDs are accepted."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        valid_ids = [
            "i-1234567890abcdef0",  # 17 hex chars
            "i-12345678",  # 8 hex chars (older format)
            "i-abcdef12",
            "i-0a1b2c3d",
            "i-0123456789abcdef0",
        ]
        for instance_id in valid_ids:
            result = validate_ec2_instance_identifier(instance_id)
            assert result == instance_id  # Should return the same ID

    def test_accepts_valid_instance_arns(self):
        """Test that valid EC2 instance ARNs are accepted and extracted."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        test_cases = [
            (
                "arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0",
                "i-1234567890abcdef0",
            ),
            (
                "arn:aws:ec2:us-west-2:987654321098:instance/i-12345678",
                "i-12345678",
            ),
            (
                "arn:aws:ec2:eu-west-1:111111111111:instance/i-abcdef12",
                "i-abcdef12",
            ),
        ]
        for arn, expected_id in test_cases:
            result = validate_ec2_instance_identifier(arn)
            assert result == expected_id

    def test_rejects_invalid_instance_ids(self):
        """Test that invalid instance IDs are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        invalid_ids = [
            "i-",  # Too short
            "i-123",  # Too short
            "i-1234567",  # Too short (7 chars)
            "i-123456789abcdef012",  # Too long (18 chars)
            "i-ABCDEF12",  # Uppercase not allowed
            "i-12345g78",  # Invalid hex char 'g'
            "i_12345678",  # Wrong separator
            "instance-12345678",  # Wrong prefix
            "12345678",  # Missing prefix
        ]
        for instance_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid EC2 instance identifier"):
                validate_ec2_instance_identifier(instance_id)

    def test_rejects_invalid_arns(self):
        """Test that invalid ARNs are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        invalid_arns = [
            "arn:aws:ec2:us-east-1:123456789012:instance/invalid",
            "arn:aws:ec2:us-east-1:123456789012:volume/vol-12345678",  # Wrong resource
            "arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",  # Wrong service
            "arn:aws:ec2:us-east-1:123456789012:i-12345678",  # Missing instance/ prefix
            "arn:aws:ec2:USEAST1:123456789012:instance/i-12345678",  # Invalid region
            "arn:aws:ec2:us-east-1:12345:instance/i-12345678",  # Invalid account ID
        ]
        for arn in invalid_arns:
            with pytest.raises(ValueError, match="Invalid EC2 instance identifier"):
                validate_ec2_instance_identifier(arn)

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        with pytest.raises(
            ValueError, match="EC2 instance identifier must be a non-empty string"
        ):
            validate_ec2_instance_identifier("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ec2_instance_identifier,
        )

        invalid_identifiers = [None, 123, [], {}, True]
        for identifier in invalid_identifiers:
            with pytest.raises(
                ValueError, match="EC2 instance identifier must be a non-empty string"
            ):
                validate_ec2_instance_identifier(identifier)


class TestValidateEcsClusterArn:
    """Test validate_ecs_cluster_arn function."""

    def test_accepts_valid_cluster_arns(self):
        """Test that valid ECS cluster ARNs are accepted."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ecs_cluster_arn,
        )

        valid_arns = [
            "arn:aws:ecs:us-east-1:123456789012:cluster/my-cluster",
            "arn:aws:ecs:us-west-2:987654321098:cluster/test-cluster",
            "arn:aws:ecs:eu-west-1:111111111111:cluster/prod-cluster",
            "arn:aws:ecs:ap-southeast-2:222222222222:cluster/cluster_name",
            "arn:aws:ecs:us-east-1:123456789012:cluster/cluster-with-hyphens",
            "arn:aws:ecs:us-east-1:123456789012:cluster/ClusterWithMixedCase",
        ]
        for arn in valid_arns:
            validate_ecs_cluster_arn(arn)  # Should not raise

    def test_rejects_invalid_cluster_arns(self):
        """Test that invalid ECS cluster ARNs are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ecs_cluster_arn,
        )

        invalid_arns = [
            "arn:aws:ec2:us-east-1:123456789012:instance/i-12345678",  # Wrong service
            "arn:aws:ecs:us-east-1:123456789012:task/my-task",  # Wrong resource type
            "arn:aws:ecs:us-east-1:123456789012:service/my-service",  # Wrong resource
            "arn:aws:ecs:us-east-1:123456789012:my-cluster",  # Missing cluster/ prefix
            "arn:aws:ecs:USEAST1:123456789012:cluster/my-cluster",  # Invalid region
            "arn:aws:ecs:us-east-1:12345:cluster/my-cluster",  # Invalid account ID
            "arn:aws:ecs:us-east-1:123456789012:cluster/",  # Empty cluster name
            "arn:aws:ecs:us-east-1:123456789012:cluster/my cluster",  # Space in name
            "arn:aws:ecs:us-east-1:123456789012:cluster/my@cluster",  # Invalid char
            "my-cluster",  # Not an ARN
            "cluster/my-cluster",  # Incomplete ARN
        ]
        for arn in invalid_arns:
            with pytest.raises(ValueError, match="Invalid ECS cluster ARN"):
                validate_ecs_cluster_arn(arn)

    def test_rejects_empty_string(self):
        """Test that empty string is rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ecs_cluster_arn,
        )

        with pytest.raises(
            ValueError, match="ECS cluster ARN must be a non-empty string"
        ):
            validate_ecs_cluster_arn("")

    def test_rejects_non_string(self):
        """Test that non-string types are rejected."""
        from amzn_nova_customization_sdk.validation.rft_multiturn_validator import (
            validate_ecs_cluster_arn,
        )

        invalid_arns = [None, 123, [], {}, True]
        for arn in invalid_arns:
            with pytest.raises(
                ValueError, match="ECS cluster ARN must be a non-empty string"
            ):
                validate_ecs_cluster_arn(arn)
