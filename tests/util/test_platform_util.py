"""
Tests for platform detection and validation utilities
"""

import unittest
from unittest.mock import patch

from amzn_nova_customization_sdk.model.model_enums import Platform
from amzn_nova_customization_sdk.util.platform_util import (
    detect_platform_from_path,
    validate_platform_compatibility,
)


class TestPlatformDetection(unittest.TestCase):
    def test_detect_platform_from_smtj_path(self):
        """Test detection of SMTJ platform from checkpoint path"""
        path = "s3://customer-escrow-123456-smtj-abc123/model/checkpoint"
        result = detect_platform_from_path(path)
        self.assertEqual(result, Platform.SMTJ)

    def test_detect_platform_from_smhp_path(self):
        """Test detection of SMHP platform from checkpoint path"""
        path = "s3://customer-escrow-123456-hp-abc123/model/checkpoint"
        result = detect_platform_from_path(path)
        self.assertEqual(result, Platform.SMHP)

    def test_detect_platform_from_non_escrow_path(self):
        """Test detection returns None for non-escrow paths"""
        path = "s3://my-bucket/output"
        result = detect_platform_from_path(path)
        self.assertIsNone(result)

    def test_detect_platform_case_sensitive(self):
        """Test that detection is case-sensitive"""
        # Should not match uppercase
        path = "s3://customer-escrow-123456-SMTJ-abc123/model/checkpoint"
        result = detect_platform_from_path(path)
        self.assertIsNone(result)


class TestPlatformValidation(unittest.TestCase):
    def test_validate_platform_compatibility_matching_smtj(self):
        """Test validation passes when platforms match (SMTJ)"""
        # Should not raise
        validate_platform_compatibility(
            checkpoint_platform=Platform.SMTJ,
            execution_platform=Platform.SMTJ,
            checkpoint_source="test checkpoint",
        )

    def test_validate_platform_compatibility_matching_smhp(self):
        """Test validation passes when platforms match (SMHP)"""
        # Should not raise
        validate_platform_compatibility(
            checkpoint_platform=Platform.SMHP,
            execution_platform=Platform.SMHP,
            checkpoint_source="test checkpoint",
        )

    def test_validate_platform_compatibility_mismatch_smhp_to_smtj(self):
        """Test validation fails when SMHP checkpoint used on SMTJ"""
        with self.assertRaises(ValueError) as context:
            validate_platform_compatibility(
                checkpoint_platform=Platform.SMHP,
                execution_platform=Platform.SMTJ,
                checkpoint_source="test checkpoint",
            )
        self.assertIn("Platform mismatch", str(context.exception))
        self.assertIn("SMHP", str(context.exception))
        self.assertIn("SMTJ", str(context.exception))

    def test_validate_platform_compatibility_mismatch_smtj_to_smhp(self):
        """Test validation fails when SMTJ checkpoint used on SMHP"""
        with self.assertRaises(ValueError) as context:
            validate_platform_compatibility(
                checkpoint_platform=Platform.SMTJ,
                execution_platform=Platform.SMHP,
                checkpoint_source="test checkpoint",
            )
        self.assertIn("Platform mismatch", str(context.exception))
        self.assertIn("SMTJ", str(context.exception))
        self.assertIn("SMHP", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.logging.logger")
    def test_validate_platform_compatibility_unknown_checkpoint(self, mock_logger):
        """Test validation logs warning when checkpoint platform is unknown"""
        # Should not raise, but should log warning
        validate_platform_compatibility(
            checkpoint_platform=None,
            execution_platform=Platform.SMTJ,
            checkpoint_source="test checkpoint",
        )
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Cannot determine platform", warning_msg)
        self.assertIn("test checkpoint", warning_msg)


if __name__ == "__main__":
    unittest.main()
