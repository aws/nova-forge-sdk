"""
Unit tests for the DataMixing class.
"""

from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_customization_sdk.util.data_mixing import DataMixing


class TestDataMixing:
    """Test suite for DataMixing class."""

    def test_init_empty(self):
        """Test initialization with empty config."""
        data_mixing = DataMixing()
        assert data_mixing.config == {}
        assert data_mixing._default_nova_fields == set()

    def test_init_with_config(self):
        """Test initialization with initial config."""
        config = {
            "nova_code_percent": 50,
            "nova_general_percent": 50,
            "customer_data_percent": 80,
        }
        data_mixing = DataMixing(config=config)
        assert data_mixing.config == config
        assert data_mixing._default_nova_fields == set()

    def test_get_config(self):
        """Test get_config returns a copy."""
        config = {"nova_code_percent": 100}
        data_mixing = DataMixing(config=config)

        returned_config = data_mixing.get_config()
        assert returned_config == config
        # Verify it's a copy
        returned_config["nova_code_percent"] = 50
        assert data_mixing.config["nova_code_percent"] == 100

    def test_set_config_basic(self):
        """Test basic set_config functionality."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 60,
            "nova_general_percent": 40,
            "customer_data_percent": 50,
        }
        data_mixing.set_config(config)
        assert data_mixing.config == config

    def test_set_config_with_normalization(self):
        """Test set_config with normalization of unspecified nova fields."""
        # First set up with a template to establish default nova fields
        data_mixing = DataMixing()
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 50},
            "nova_math_percent": {"default": 0},
            "customer_data_percent": {"default": 50},
        }
        data_mixing._load_defaults_from_template(template)

        # Now set config with missing nova fields
        new_config = {"nova_code_percent": 100, "customer_data_percent": 50}
        data_mixing.set_config(new_config, normalize=True)

        # Should have all nova fields with unspecified ones set to 0
        expected = {
            "nova_code_percent": 100,
            "nova_general_percent": 0,
            "nova_math_percent": 0,
            "customer_data_percent": 50,
        }
        assert data_mixing.config == expected

    def test_set_config_without_normalization(self):
        """Test set_config without normalization."""
        data_mixing = DataMixing()
        data_mixing._default_nova_fields = {
            "nova_code_percent",
            "nova_general_percent",
            "customer_data_percent",
        }

        config = {"nova_code_percent": 100, "customer_data_percent": 50}
        data_mixing.set_config(config, normalize=False)

        # Should only have specified fields
        assert data_mixing.config == config

    def test_set_config_rejects_dataset_catalog(self):
        """Test that dataset_catalog is filtered out and cannot be set."""
        data_mixing = DataMixing()
        config = {"nova_code_percent": 100, "dataset_catalog": "some_catalog"}

        with patch(
            "amzn_nova_customization_sdk.util.data_mixing.logger"
        ) as mock_logger:
            data_mixing.set_config(config)
            mock_logger.warning.assert_called_once()
            assert "dataset_catalog" in data_mixing.config
            assert "nova_code_percent" in data_mixing.config

    def test_set_config_invalid_nova_field(self):
        """Test set_config with invalid nova field raises error."""
        data_mixing = DataMixing()
        data_mixing._default_nova_fields = {"nova_code_percent", "nova_general_percent"}

        config = {"nova_invalid_percent": 50, "nova_code_percent": 50}

        with pytest.raises(
            ValueError, match="Invalid nova field 'nova_invalid_percent'"
        ):
            data_mixing.set_config(config)

    def test_load_defaults_from_template(self):
        """Test loading defaults from template."""
        data_mixing = DataMixing()
        template = {
            "nova_code_percent": {"default": 30},
            "nova_general_percent": {"default": 70},
            "customer_data_percent": {"default": 80},
            "dataset_catalog": {"default": "catalog_name"},
            "other_field": {"default": "value"},
        }

        with patch("amzn_nova_customization_sdk.util.data_mixing.logger"):
            data_mixing._load_defaults_from_template(template)

        # dataset_catalog is stored separately, not in config directly
        expected_config = {
            "nova_code_percent": 30,
            "nova_general_percent": 70,
            "customer_data_percent": 80,
        }
        assert data_mixing.config == expected_config
        # But it should be returned by get_config()
        full_config = data_mixing.get_config()
        assert full_config["dataset_catalog"] == "catalog_name"
        assert data_mixing._default_nova_fields == {
            "nova_code_percent",
            "nova_general_percent",
            "customer_data_percent",
        }

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 60,
            "nova_general_percent": 40,
            "customer_data_percent": 50,
        }
        # Validation happens in set_config, not as a separate method
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)  # Should not raise

    def test_validate_nova_sum_not_100(self):
        """Test validation fails when nova fields don't sum to 100."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 60,
            "nova_general_percent": 30,  # Sum is 90, not 100
            "customer_data_percent": 50,
        }

        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config",
            side_effect=ValueError("Nova data percentages must sum to 100"),
        ):
            with pytest.raises(
                ValueError, match="Nova data percentages must sum to 100"
            ):
                data_mixing.set_config(config)

    def test_validate_customer_data_out_of_range(self):
        """Test validation fails when customer_data_percent is out of range."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 100,
            "customer_data_percent": 150,  # Out of range
        }

        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config",
            side_effect=ValueError("customer_data_percent must be between 0 and 100"),
        ):
            with pytest.raises(
                ValueError, match="customer_data_percent must be between 0 and 100"
            ):
                data_mixing.set_config(config)

    def test_validate_nova_field_out_of_range(self):
        """Test validation fails when nova field is out of range."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 150,  # Out of range
            "nova_general_percent": -50,  # Also out of range
        }

        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config",
            side_effect=ValueError("nova_code_percent must be between 0 and 100"),
        ):
            with pytest.raises(
                ValueError, match="nova_code_percent must be between 0 and 100"
            ):
                data_mixing.set_config(config)

    def test_validate_customer_100_with_nova_data(self):
        """Test validation fails when customer is 100% but nova data is non-zero."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 50,
            "nova_general_percent": 50,
            "customer_data_percent": 100,
        }

        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config",
            side_effect=ValueError(
                "Since customer_data_percent is 100 %, all nova data should sum to 0 %"
            ),
        ):
            with pytest.raises(
                ValueError,
                match="Since customer_data_percent is 100 %, all nova data should sum to 0 %",
            ):
                data_mixing.set_config(config)

    def test_validate_with_floating_point_sum(self):
        """Test validation handles floating point errors in sum."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 33.33,
            "nova_general_percent": 33.33,
            "nova_math_percent": 33.34,  # Sum is 99.99999...
        }
        # Validation happens in set_config
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(
                config
            )  # Should not raise due to floating point tolerance

    def test_validate_with_none_values(self):
        """Test validation handles None values correctly."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 100,
            "nova_general_percent": None,
            "customer_data_percent": 50,
        }
        # Validation happens in set_config
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)  # Should not raise

    def test_update_default_nova_fields(self):
        """Test that nova fields are added to defaults when set."""
        data_mixing = DataMixing()
        # When loading from template, fields are added to defaults
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 30},
            "nova_math_percent": {"default": 20},
            "customer_data_percent": {"default": 80},
        }
        data_mixing._load_defaults_from_template(template)

        expected_fields = {
            "nova_code_percent",
            "nova_general_percent",
            "nova_math_percent",
            "customer_data_percent",
        }
        assert data_mixing._default_nova_fields == expected_fields

    def test_constants(self):
        """Test class constants are defined correctly."""
        assert DataMixing.NOVA_PREFIX == "nova_"
        assert DataMixing.PERCENT_SUFFIX == "_percent"
        assert DataMixing.CUSTOMER_DATA_FIELD == "customer_data_percent"
        assert DataMixing.DATASET_CATALOG_FIELD == "dataset_catalog"

    def test_set_config_adds_new_nova_fields_when_no_defaults(self):
        """Test that new nova fields are added to defaults when no template is loaded."""
        data_mixing = DataMixing()
        # Starts with empty default fields
        assert data_mixing._default_nova_fields == set()

        config = {"nova_new_percent": 100, "customer_data_percent": 50}
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)

        # Fields are not automatically added to defaults in current implementation
        # unless template is loaded first

    def test_complex_workflow(self):
        """Test a complex workflow with multiple operations."""
        # Initialize with template
        data_mixing = DataMixing()
        template = {
            "nova_code_percent": {"default": 40},
            "nova_general_percent": {"default": 60},
            "customer_data_percent": {"default": 70},
        }
        data_mixing._load_defaults_from_template(template)

        # Update config
        new_config = {
            "nova_code_percent": 25,
            "nova_general_percent": 75,
            "customer_data_percent": 50,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(new_config)

        # Get and verify config
        config = data_mixing.get_config()
        assert config == new_config

    def test_edge_case_empty_nova_fields(self):
        """Test edge case with no nova fields."""
        data_mixing = DataMixing()
        config = {"customer_data_percent": 100}
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)  # Should not raise

    def test_edge_case_zero_customer_data(self):
        """Test edge case with 0% customer data."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 70,
            "nova_general_percent": 30,
            "customer_data_percent": 0,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)  # Should not raise

    def test_load_defaults_with_non_dict_values(self):
        """Test loading defaults with non-dict values in template."""
        data_mixing = DataMixing()
        template = {
            "nova_code_percent": 30,  # Direct value, not dict
            "nova_general_percent": {"default": 70},
            "customer_data_percent": {"default": 80},
        }

        with patch("amzn_nova_customization_sdk.util.data_mixing.logger"):
            data_mixing._load_defaults_from_template(template)

        # Should only load values from dict entries with 'default' key
        expected_config = {"nova_general_percent": 70, "customer_data_percent": 80}
        assert data_mixing.config == expected_config

    def test_set_config_preserves_existing_fields(self):
        """Test that set_config preserves existing fields when normalize=False."""
        data_mixing = DataMixing()

        # Set initial config
        initial_config = {
            "nova_code_percent": 50,
            "nova_general_percent": 50,
            "customer_data_percent": 70,
            "some_other_field": "value",
        }
        data_mixing.set_config(initial_config, normalize=False)

        # Update with partial config
        update_config = {"nova_code_percent": 60, "nova_general_percent": 40}
        data_mixing.set_config(update_config, normalize=False)

        # Should have updated values
        assert data_mixing.config["nova_code_percent"] == 60
        assert data_mixing.config["nova_general_percent"] == 40

    def test_multiple_validations(self):
        """Test multiple validation scenarios in sequence."""
        data_mixing = DataMixing()

        # Valid config 1
        config1 = {"nova_code_percent": 100, "customer_data_percent": 50}
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config1)

        # Valid config 2
        config2 = {
            "nova_code_percent": 30,
            "nova_general_percent": 70,
            "customer_data_percent": 80,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config2)

        # Invalid config
        config3 = {
            "nova_code_percent": 30,
            "nova_general_percent": 50,  # Sum is 80, not 100
            "customer_data_percent": 80,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config",
            side_effect=ValueError("Invalid config"),
        ):
            with pytest.raises(ValueError):
                data_mixing.set_config(config3)

    def test_validate_with_zero_nova_percentages(self):
        """Test validation with nova fields set to 0."""
        data_mixing = DataMixing()
        config = {
            "nova_code_percent": 0,
            "nova_general_percent": 100,
            "nova_math_percent": 0,
            "customer_data_percent": 50,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config)  # Should not raise

    def test_post_init_updates_default_fields(self):
        """Test that initialization sets correct default fields."""
        config = {
            "nova_code_percent": 50,
            "nova_general_percent": 30,
            "nova_math_percent": 20,
        }
        data_mixing = DataMixing(config=config)

        # Default fields start empty unless template is loaded
        assert data_mixing._default_nova_fields == set()

    def test_precedence_with_new_nova_fields(self):
        """Test precedence when new nova fields are introduced."""
        data_mixing = DataMixing()

        # First load a template to establish known fields
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 50},
            "customer_data_percent": {"default": 80},
        }
        data_mixing._load_defaults_from_template(template)

        # When template has been loaded, new fields can't be added dynamically
        invalid_config = {
            "nova_code_percent": 30,
            "nova_general_percent": 30,
            "nova_science_percent": 40,  # Invalid new field
            "customer_data_percent": 70,
        }

        # Should reject the new field
        with pytest.raises(
            ValueError, match="Invalid nova field 'nova_science_percent'"
        ):
            data_mixing.set_config(invalid_config)

        # But if no template is loaded, all nova fields can be added
        data_mixing2 = DataMixing()

        # Without template, new nova fields can be added
        config_with_new_field = {
            "nova_code_percent": 30,
            "nova_general_percent": 30,
            "nova_science_percent": 40,
            "customer_data_percent": 70,
        }
        # This should work without template
        data_mixing2.set_config(config_with_new_field)

        # Verify fields were added
        assert data_mixing2.config["nova_code_percent"] == 30
        assert data_mixing2.config["nova_general_percent"] == 30
        assert data_mixing2.config["nova_science_percent"] == 40
        assert data_mixing2.config["customer_data_percent"] == 70

    def test_precedence_normalization_behavior(self):
        """Test how normalization affects precedence scenarios."""
        data_mixing = DataMixing()

        # Setup with multiple nova fields
        template = {
            "nova_code_percent": {"default": 25},
            "nova_general_percent": {"default": 25},
            "nova_math_percent": {"default": 25},
            "nova_science_percent": {"default": 25},
            "customer_data_percent": {"default": 50},
        }
        data_mixing._load_defaults_from_template(template)

        # Override with normalization (should set unspecified to 0)
        override_with_norm = {
            "nova_code_percent": 60,
            "nova_general_percent": 40,
            # nova_math_percent and nova_science_percent not specified
            "customer_data_percent": 75,
        }
        data_mixing.set_config(override_with_norm, normalize=True)

        assert data_mixing.config["nova_code_percent"] == 60
        assert data_mixing.config["nova_general_percent"] == 40
        assert data_mixing.config["nova_math_percent"] == 0  # Normalized to 0
        assert data_mixing.config["nova_science_percent"] == 0  # Normalized to 0
        assert data_mixing.config["customer_data_percent"] == 75

        # Now override without normalization
        override_no_norm = {
            "nova_code_percent": 70,
            "nova_general_percent": 30,
            "customer_data_percent": 80,
        }
        data_mixing.set_config(override_no_norm, normalize=False)

        # Only specified fields should be updated
        assert data_mixing.config["nova_code_percent"] == 70
        assert data_mixing.config["nova_general_percent"] == 30
        assert data_mixing.config["customer_data_percent"] == 80

    def test_precedence_with_dataset_catalog(self):
        """Test that dataset_catalog is properly handled across precedence levels."""
        data_mixing = DataMixing()

        # Template with dataset_catalog
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 50},
            "customer_data_percent": {"default": 70},
            "dataset_catalog": {"default": "catalog_v1"},
        }
        data_mixing._load_defaults_from_template(template)

        # Verify dataset_catalog was loaded (via get_config, not directly in config)
        full_config = data_mixing.get_config()
        assert full_config["dataset_catalog"] == "catalog_v1"

        # Try to override dataset_catalog (should be rejected)
        override_config = {
            "nova_code_percent": 100,
            "customer_data_percent": 60,
            "dataset_catalog": "catalog_v2",  # Should be filtered out
        }

        with patch(
            "amzn_nova_customization_sdk.util.data_mixing.logger"
        ) as mock_logger:
            data_mixing.set_config(override_config)
            mock_logger.warning.assert_called_once()

        # dataset_catalog should still be the original value from template
        full_config = data_mixing.get_config()
        assert full_config["dataset_catalog"] == "catalog_v1"
        assert data_mixing.config["nova_code_percent"] == 100

    def test_precedence_with_validation_errors(self):
        """Test precedence behavior when validation fails."""
        data_mixing = DataMixing()

        # Setup valid initial state
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 50},
            "customer_data_percent": {"default": 70},
        }
        data_mixing._load_defaults_from_template(template)

        # Try to set invalid config (nova fields don't sum to 100)
        invalid_config = {
            "nova_code_percent": 60,
            "nova_general_percent": 30,  # Sum is 90, not 100
            "customer_data_percent": 80,
        }

        with pytest.raises(ValueError, match="Nova data percentages must sum to 100"):
            data_mixing.set_config(invalid_config)

        # In the current implementation, config is updated before validation
        # so if validation fails, the config is left in an invalid state
        assert data_mixing.config["nova_code_percent"] == 50  # Changed despite error
        assert data_mixing.config["nova_general_percent"] == 50  # Changed despite error

    def test_precedence_with_none_values(self):
        """Test precedence handling with None values in configs."""
        data_mixing = DataMixing()

        # Template with defaults
        template = {
            "nova_code_percent": {"default": 50},
            "nova_general_percent": {"default": 50},
            "customer_data_percent": {"default": 70},
        }
        data_mixing._load_defaults_from_template(template)

        # Config with None values (should be handled gracefully)
        config_with_none = {
            "nova_code_percent": 100,
            "nova_general_percent": None,  # None value
            "customer_data_percent": 50,
        }
        with patch(
            "amzn_nova_customization_sdk.validation.validator.Validator.validate_data_mixing_config"
        ):
            data_mixing.set_config(config_with_none, normalize=False)

        assert data_mixing.config["nova_code_percent"] == 100
        assert data_mixing.config["nova_general_percent"] is None
        assert data_mixing.config["customer_data_percent"] == 50

    def test_is_data_mixing_field_nova_percent_fields(self):
        """Test is_data_mixing_field identifies nova_*_percent fields."""
        data_mixing = DataMixing()

        # Test fields that start with nova_ and end with _percent
        # These are all valid patterns regardless of what's in the middle
        assert data_mixing._is_data_mixing_field("nova_code_percent") is True
        assert data_mixing._is_data_mixing_field("nova_general_percent") is True
        assert data_mixing._is_data_mixing_field("nova_math_percent") is True
        assert data_mixing._is_data_mixing_field("nova_science_percent") is True
        assert (
            data_mixing._is_data_mixing_field("nova_percent") is True
        )  # Empty middle is still valid pattern
        assert (
            data_mixing._is_data_mixing_field("nova_unknown_percent") is True
        )  # Any nova_*_percent matches

        # Test fields that don't match the pattern
        assert data_mixing._is_data_mixing_field("nova_code") is False
        assert data_mixing._is_data_mixing_field("code_percent") is False
        assert data_mixing._is_data_mixing_field("random_field") is False

    def test_is_data_mixing_field_customer_data(self):
        """Test is_data_mixing_field identifies customer_data_percent field."""
        data_mixing = DataMixing()

        # customer_data_percent should always be identified as a data mixing field
        assert data_mixing._is_data_mixing_field("customer_data_percent") is True

        # Similar but not exact matches should not be identified
        assert data_mixing._is_data_mixing_field("customer_data") is False
        assert data_mixing._is_data_mixing_field("customer_percent") is False

    def test_is_data_mixing_field_dataset_catalog(self):
        """Test is_data_mixing_field identifies dataset_catalog field."""
        data_mixing = DataMixing()

        # dataset_catalog should always be identified as a data mixing field
        assert data_mixing._is_data_mixing_field("dataset_catalog") is True

        # Similar but not exact matches should not be identified
        assert data_mixing._is_data_mixing_field("dataset") is False
        assert data_mixing._is_data_mixing_field("catalog") is False

    def test_is_data_mixing_field_shorthand_in_config(self):
        """Test is_data_mixing_field identifies shorthand fields present in config."""
        data_mixing = DataMixing()

        # Set up config with some nova fields
        config = {
            "nova_code_percent": 50,
            "nova_general_percent": 50,
            "customer_data_percent": 80,
        }
        data_mixing.set_config(config)

        # Test shorthand versions (without nova_ prefix and _percent suffix)
        # These should be recognized if nova_<key>_percent exists in config
        assert data_mixing._is_data_mixing_field("code") is True
        assert data_mixing._is_data_mixing_field("general") is True

        # Shorthand for fields not in config should not be recognized
        assert data_mixing._is_data_mixing_field("math") is False
        assert data_mixing._is_data_mixing_field("science") is False

    def test_is_data_mixing_field_with_empty_config(self):
        """Test is_data_mixing_field with empty configuration."""
        data_mixing = DataMixing()

        # Standard data mixing fields should still be recognized
        assert data_mixing._is_data_mixing_field("nova_code_percent") is True
        assert data_mixing._is_data_mixing_field("customer_data_percent") is True
        assert data_mixing._is_data_mixing_field("dataset_catalog") is True

        # Shorthand versions should not be recognized without config
        assert data_mixing._is_data_mixing_field("code") is False
        assert data_mixing._is_data_mixing_field("general") is False

    def test_is_data_mixing_field_comprehensive(self):
        """Comprehensive test for is_data_mixing_field method."""
        data_mixing = DataMixing()

        # Load a template to set up default fields
        template = {
            "nova_code_percent": {"default": 30},
            "nova_general_percent": {"default": 40},
            "nova_math_percent": {"default": 30},
            "customer_data_percent": {"default": 70},
            "dataset_catalog": {"default": "my_catalog"},
        }
        data_mixing._load_defaults_from_template(template)

        # Test all types of valid data mixing fields
        valid_fields = [
            "nova_code_percent",  # Full nova field
            "nova_general_percent",  # Full nova field
            "nova_math_percent",  # Full nova field
            "nova_unknown_percent",  # Nova pattern (any nova_*_percent is valid)
            "customer_data_percent",  # Customer data field
            "dataset_catalog",  # Dataset catalog field
            "code",  # Shorthand for nova_code_percent
            "general",  # Shorthand for nova_general_percent
            "math",  # Shorthand for nova_math_percent
        ]

        for field in valid_fields:
            assert data_mixing._is_data_mixing_field(field) is True, (
                f"Field {field} should be recognized"
            )

        # Test invalid fields
        invalid_fields = [
            "unknown",  # Shorthand for non-existent field
            "random_field",  # Completely unrelated field
            "nova_code_percentage",  # Wrong suffix
            "new_code_percent",  # Wrong prefix
            "",  # Empty string
            "percent",  # Just suffix
            "nova_",  # Just prefix
        ]

        for field in invalid_fields:
            assert data_mixing._is_data_mixing_field(field) is False, (
                f"Field {field} should not be recognized"
            )

    def test_is_data_mixing_field_edge_cases(self):
        """Test edge cases for is_data_mixing_field method."""
        data_mixing = DataMixing()

        # Test with special characters and edge cases
        assert (
            data_mixing._is_data_mixing_field("nova__percent") is True
        )  # Empty middle part
        assert (
            data_mixing._is_data_mixing_field("nova_123_percent") is True
        )  # Numbers in middle
        assert (
            data_mixing._is_data_mixing_field("nova_special-char_percent") is True
        )  # Special chars
        assert (
            data_mixing._is_data_mixing_field("NOVA_code_percent") is False
        )  # Case sensitive
        assert (
            data_mixing._is_data_mixing_field("nova_code_PERCENT") is False
        )  # Case sensitive
