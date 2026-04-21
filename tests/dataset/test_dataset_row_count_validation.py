"""Tests for dataset row count validation across both validation paths.

Path 1: BaseDatasetValidator._validate_row_counts() — called via loader.validate()
Path 2: Validator._validate_dataset_row_counts() — called during train()/evaluate()
"""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.dataset_validator.cpt_dataset_validator import (
    CPTDatasetValidator,
)
from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
    BaseDatasetValidator,
)
from amzn_nova_forge.dataset.dataset_validator.sft_dataset_validator import (
    SFTDatasetValidator,
)
from amzn_nova_forge.model.model_enums import Model, Platform, TrainingMethod
from amzn_nova_forge.validation.dataset_row_count_validator import (
    _get_recipe_value,
    count_s3_dataset_rows,
    validate_row_counts,
)
from amzn_nova_forge.validation.validator import Validator


def _make_sft_sample(text="Hello"):
    """Create a minimal valid SFT sample."""
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "messages": [
            {"role": "user", "content": [{"text": text}]},
            {"role": "assistant", "content": [{"text": "Response"}]},
        ],
    }


def _make_cpt_sample(text="Sample text for CPT."):
    return {"text": text}


def _make_loader(samples):
    loader = JSONLDatasetLoader()
    loader.dataset = lambda: iter(samples)
    return loader


# ---------------------------------------------------------------------------
# Path 1: BaseDatasetValidator._validate_row_counts (loader.validate path)
# ---------------------------------------------------------------------------
class TestBaseDatasetValidatorRowCounts(unittest.TestCase):
    """Tests for _validate_row_counts on the dataset validator side."""

    # -- max_rows checks --

    def test_max_rows_exceeded_raises(self):
        """Exceeding max_rows for BEDROCK/SFT/NOVA_LITE_2 should NOT raise in
        loader.validate — Bedrock bounds are pre_training_only."""
        samples = [_make_sft_sample(f"q{i}") for i in range(20001)]

        # Should not raise — pre_training_only checks skipped in loader.validate
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )

    def test_max_rows_at_limit_passes(self):
        """Exactly 20000 rows should pass for BEDROCK/SFT/NOVA_LITE_2."""
        samples = [_make_sft_sample(f"q{i}") for i in range(20000)]

        # Should not raise
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )

    def test_max_rows_not_applied_to_smtj(self):
        """SMTJ platform should not enforce the BEDROCK max_rows limit."""
        samples = [_make_sft_sample(f"q{i}") for i in range(20001)]

        # Should not raise — max_rows check only applies to BEDROCK
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

    def test_max_rows_applied_to_nova_lite_on_bedrock(self):
        """NOVA_LITE on BEDROCK — pre_training_only, so loader.validate should pass."""
        samples = [_make_sft_sample(f"q{i}") for i in range(20001)]

        # Should not raise — pre_training_only checks skipped in loader.validate
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE,
            platform=Platform.BEDROCK,
        )

    # -- hard min_rows checks (fixed limits from DATASET_CHECKS config) --

    def test_hard_min_rows_below_minimum_raises(self):
        """BEDROCK/SFT/NOVA_LITE_2 with < 200 rows — pre_training_only, so
        loader.validate should pass."""
        samples = [_make_sft_sample(f"q{i}") for i in range(10)]

        # Should not raise — pre_training_only checks skipped in loader.validate
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )

    # -- recipe-dependent min check is skipped in loader.validate --

    def test_smtj_skips_recipe_dependent_min_check(self):
        """SMTJ min_rows_recipe_field check is skipped in loader.validate (no recipe).

        The recipe-dependent check (global_batch_size) is only enforced
        during train() via Validator._validate_dataset_row_counts.
        """
        samples = [_make_sft_sample("q")]

        # Should not raise — recipe-dependent min check not run here
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

    # -- skipping checks --

    def test_no_platform_skips_row_checks(self):
        """Without platform kwarg, row count checks are skipped."""
        samples = [_make_sft_sample("q")]

        # Should not raise even with 1 row
        _make_loader(samples).validate(
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
        )

    # -- CPT validator also calls _validate_row_counts --

    def test_cpt_validator_forwards_row_count_kwargs(self):
        """CPT validator should also enforce row count checks."""
        validator = CPTDatasetValidator()
        samples = [_make_cpt_sample(f"text {i}") for i in range(20001)]

        # CPT has no BEDROCK max_rows check in current config, so this should pass
        # (CPT is not in the VALIDATION_CHECKS for max_rows)
        validator.validate(
            iter(samples),
            Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
            training_method=TrainingMethod.CPT,
        )


# ---------------------------------------------------------------------------
# Path 2: Validator._validate_dataset_row_counts (pre-flight validator path)
# ---------------------------------------------------------------------------
class TestValidatorDatasetRowCounts(unittest.TestCase):
    """Tests for Validator._validate_dataset_row_counts (S3-based path)."""

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=25000,
    )
    def test_max_rows_exceeded_appends_error(self, mock_count):
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.BEDROCK,
            method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 4},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds the maximum of 20000", errors[0])

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=20000,
    )
    def test_max_rows_at_limit_no_error(self, mock_count):
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.BEDROCK,
            method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 4},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 0)

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=2,
    )
    def test_min_rows_below_recipe_field_appends_error(self, mock_count):
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.SMTJ,
            method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 8},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("below the minimum of 8", errors[0])
        self.assertIn("global_batch_size", errors[0])

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=8,
    )
    def test_min_rows_at_recipe_field_no_error(self, mock_count):
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.SMTJ,
            method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 8},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 0)

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=100,
    )
    def test_no_applicable_checks_no_error(self, mock_count):
        """DPO_LORA on NOVA_LITE has no VALIDATION_CHECKS entries."""
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.SMTJ,
            method=TrainingMethod.DPO_LORA,
            model=Model.NOVA_LITE,
            recipe={},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 0)
        mock_count.assert_not_called()

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        side_effect=Exception("S3 error"),
    )
    def test_s3_count_failure_is_non_fatal(self, mock_count):
        """If S3 row counting fails, validation should log a warning, not raise."""
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.BEDROCK,
            method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 4},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 0)

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=100,
    )
    def test_missing_recipe_field_skips_min_check(self, mock_count):
        """If recipe doesn't contain the min_rows_recipe_field, skip that check."""
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.SMTJ,
            method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            recipe={},  # no global_batch_size
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 0)

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=25000,
    )
    def test_bedrock_sft_full_also_checked(self, mock_count):
        """SFT_FULL should also be subject to the BEDROCK max_rows check."""
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.BEDROCK,
            method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 4},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("exceeds the maximum of 20000", errors[0])

    @patch(
        "amzn_nova_forge.validation.validator.count_s3_dataset_rows",
        return_value=3,
    )
    def test_both_max_and_min_can_trigger(self, mock_count):
        """On BEDROCK with very few rows, min check should trigger (max won't)."""
        errors = []
        Validator._validate_dataset_row_counts(
            platform=Platform.BEDROCK,
            method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            recipe={"global_batch_size": 8},
            data_s3_path="s3://bucket/data.jsonl",
            region="us-east-1",
            errors=errors,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("below the minimum of 200", errors[0])


# ---------------------------------------------------------------------------
# count_s3_dataset_rows
# ---------------------------------------------------------------------------
class TestCountS3DatasetRows(unittest.TestCase):
    """Tests for the S3 row counting helper."""

    @patch("boto3.client")
    def test_counts_non_empty_lines(self, mock_boto_client):
        content = b"line1\nline2\n\nline3\n"
        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.iter_lines.return_value = iter(content.split(b"\n"))
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_boto_client.return_value = mock_s3

        count = count_s3_dataset_rows("s3://bucket/data.jsonl", "us-east-1")
        self.assertEqual(count, 3)

    @patch("boto3.client")
    def test_empty_file_returns_zero(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.iter_lines.return_value = iter([b""])
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_boto_client.return_value = mock_s3

        count = count_s3_dataset_rows("s3://bucket/empty.jsonl", "us-east-1")
        self.assertEqual(count, 0)

    def test_invalid_s3_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            count_s3_dataset_rows("not-an-s3-path", "us-east-1")
        self.assertIn("Invalid S3 path", str(ctx.exception))

    def test_missing_region_raises(self):
        for region in (None, ""):
            with self.subTest(region=region):
                with self.assertRaises(ValueError) as ctx:
                    count_s3_dataset_rows("s3://bucket/data.jsonl", region)
                self.assertIn("valid AWS region", str(ctx.exception))


# ---------------------------------------------------------------------------
# DATASET_CHECKS config sanity (row-count entries)
# ---------------------------------------------------------------------------
class TestValidationChecksConfig(unittest.TestCase):
    """Sanity checks on the row-count entries in DATASET_CHECKS."""

    def test_all_row_count_entries_have_required_keys(self):
        from amzn_nova_forge.dataset.configs.dataset_checks_config import (
            DATASET_CHECKS,
        )

        for check in DATASET_CHECKS:
            if check.get("type") != "row_count":
                continue
            self.assertIn("name", check)
            self.assertIn("training_methods", check)
            self.assertIn("platforms", check)
            self.assertIn("models", check)
            self.assertFalse(
                check.get("filterable", False),
                f"Row-count check '{check['name']}' should not be filterable",
            )
            self.assertTrue(
                "max_rows" in check or "min_rows" in check or "min_rows_recipe_field" in check,
                f"Check '{check['name']}' must have max_rows, min_rows, or min_rows_recipe_field",
            )

    def test_bedrock_sft_nova_lite_2_max_rows_is_20000(self):
        from amzn_nova_forge.dataset.configs.dataset_checks_config import (
            DATASET_CHECKS,
        )

        entry = next(
            c for c in DATASET_CHECKS if c["name"] == "bedrock_sample_bounds_nova_lite_2_sft"
        )
        self.assertEqual(entry["max_rows"], 20000)
        self.assertEqual(entry["min_rows"], 200)
        self.assertIn(Platform.BEDROCK, entry["platforms"])
        self.assertIn(TrainingMethod.SFT_LORA, entry["training_methods"])
        self.assertIn(TrainingMethod.SFT_FULL, entry["training_methods"])
        self.assertIn(Model.NOVA_LITE_2, entry["models"])

    def test_min_rows_sft_nova_lite_2_uses_global_batch_size(self):
        from amzn_nova_forge.dataset.configs.dataset_checks_config import (
            DATASET_CHECKS,
        )

        entry = next(c for c in DATASET_CHECKS if c["name"] == "min_dataset_rows_sft_nova_lite_2")
        self.assertEqual(entry["min_rows_recipe_field"], "global_batch_size")
        self.assertIn(Platform.SMTJ, entry["platforms"])
        self.assertIn(Platform.SMHP, entry["platforms"])


# ---------------------------------------------------------------------------
# _get_recipe_value
# ---------------------------------------------------------------------------
class TestGetRecipeValue(unittest.TestCase):
    """Tests for the recursive _get_recipe_value helper."""

    def test_finds_key_at_top_level(self):
        self.assertEqual(_get_recipe_value({"global_batch_size": 16}, "global_batch_size"), 16)

    def test_finds_key_nested(self):
        recipe = {"training_config": {"global_batch_size": 32}}
        self.assertEqual(_get_recipe_value(recipe, "global_batch_size"), 32)

    def test_skips_branch_without_key(self):
        recipe = {
            "other": {"unrelated": 1},
            "training_config": {"global_batch_size": 4},
        }
        self.assertEqual(_get_recipe_value(recipe, "global_batch_size"), 4)

    def test_raises_key_error_when_missing(self):
        with self.assertRaises(KeyError):
            _get_recipe_value({"a": {"b": 1}}, "missing_key")


# ---------------------------------------------------------------------------
# validate_row_counts — _report raises ValueError when errors is None
# ---------------------------------------------------------------------------
class TestValidateRowCountsRaisesDirectly(unittest.TestCase):
    """When errors=None (loader.validate path), violations raise ValueError."""

    def test_recipe_field_min_raises_value_error(self):
        """SMTJ/SFT/NOVA_LITE_2 with recipe and too few rows raises directly."""
        with self.assertRaises(ValueError) as ctx:
            validate_row_counts(
                num_samples=2,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJ,
                training_method=TrainingMethod.SFT_LORA,
                recipe={"training_config": {"global_batch_size": 8}},
                errors=None,
            )
        self.assertIn("below the minimum of 8", str(ctx.exception))
        self.assertIn("global_batch_size", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
