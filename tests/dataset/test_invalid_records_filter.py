"""Tests for FilterMethod.INVALID_RECORDS — schema validation and reserved keyword filtering.

Most tests call ``get_filter_operation()`` + ``op.execute()`` directly to unit-test
the ``InvalidRecordsFilterOperation`` in isolation (no loader orchestration, no
``_has_transforms`` guard).  ``TestLoaderFilterIntegration`` separately verifies
that ``DatasetLoader.filter()`` correctly wires up to the operation (chaining,
immediate execution, pre-transform guard, ``last_filter_result``).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock pymediainfo at module level so all tests use the mock
sys.modules.pop("pymediainfo", None)
_mock_mediainfo_cls = MagicMock()
_mock_track = MagicMock()
_mock_track.duration = 5000
_mock_track.track_type = "Video"
_mock_mediainfo_cls.parse.return_value = MagicMock(tracks=[_mock_track])
_mock_pymediainfo = MagicMock()
_mock_pymediainfo.MediaInfo = _mock_mediainfo_cls
sys.modules["pymediainfo"] = _mock_pymediainfo

from amzn_nova_forge.dataset.configs.dataset_checks_config import (
    CONVERSE_FORMAT_RESERVED_KEYWORDS,
    MAX_IMAGE_FILE_SIZE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    MAX_VIDEO_FILE_SIZE_BYTES,
    MAX_VIDEOS_PER_MESSAGE,
)
from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.dataset_validator.eval_dataset_validator import (
    EvalDatasetSample,
)
from amzn_nova_forge.dataset.dataset_validator.rft_multiturn_dataset_validator import (
    RFTMultiturnDatasetSample,
)
from amzn_nova_forge.dataset.operations.filter_operation import (
    FilterMethod,
    get_filter_operation,
)
from amzn_nova_forge.dataset.operations.invalid_records_filter_operation import (
    FILTER_CHECKS,
    InvalidRecordsFilterOperation,
    _get_applicable_checks,
    _get_sample_model,
    _sample_fails_schema,
)
from amzn_nova_forge.model.model_enums import Model, Platform, TrainingMethod
from amzn_nova_forge.recipe.recipe_config import EvaluationTask

_SCHEMA_VERSION = "bedrock-conversation-2024"

_SFT_KWARGS = {
    "training_method": TrainingMethod.SFT_LORA,
    "model": Model.NOVA_LITE_2,
    "platform": Platform.SMTJ,
}


def _make_loader(samples):
    loader = JSONLDatasetLoader()
    loader.dataset = lambda: iter(samples)
    loader._load_path = "in-memory"
    return loader


_CLEAN_SAMPLE = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "What is 2+2?"}]},
        {"role": "assistant", "content": [{"text": "4"}]},
    ],
}

_SAMPLE_WITH_RESERVED = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "User: What is 2+2?"}]},
        {"role": "assistant", "content": [{"text": "4"}]},
    ],
}

_SAMPLE_WITH_IMAGE_TAG = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "Describe this <image> please"}]},
        {"role": "assistant", "content": [{"text": "A cat."}]},
    ],
}

_SAMPLE_WITH_EOS = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there [EOS]"}]},
    ],
}

_SAMPLE_NESTED_RESERVED = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "Tell me"}]},
        {"role": "assistant", "content": [{"text": "Sure, ASSISTANT: here you go"}]},
    ],
}

_SAMPLE_BAD_SCHEMA = {
    "schemaVersion": "wrong-version",
    "messages": [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
    ],
}

_SAMPLE_MISSING_MESSAGES = {
    "schemaVersion": _SCHEMA_VERSION,
}


class TestFactoryRegistration(unittest.TestCase):
    def test_factory_returns_correct_class(self):
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        self.assertIsInstance(op, InvalidRecordsFilterOperation)


class TestSFTFiltering(unittest.TestCase):
    """SFT methods should apply reserved keyword checks."""

    def test_removes_sample_with_reserved_keyword(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.status, "SUCCEEDED")
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_sft_full_also_filters(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [])

    def test_keeps_clean_samples(self):
        loader = _make_loader([_CLEAN_SAMPLE, _CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.filtered_count, 0)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(len(list(loader.dataset())), 2)

    def test_removes_all_when_all_invalid(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED, _SAMPLE_WITH_IMAGE_TAG, _SAMPLE_WITH_EOS])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.filtered_count, 3)
        self.assertEqual(list(loader.dataset()), [])

    def test_filters_each_reserved_keyword(self):
        """Each reserved keyword individually triggers removal."""
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        for kw in CONVERSE_FORMAT_RESERVED_KEYWORDS:
            sample = {
                "schemaVersion": _SCHEMA_VERSION,
                "messages": [
                    {"role": "user", "content": [{"text": f"prefix {kw} suffix"}]},
                    {"role": "assistant", "content": [{"text": "ok"}]},
                ],
            }
            loader = _make_loader([sample])
            result = op.execute(
                loader,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJ,
            )
            self.assertEqual(result.filtered_count, 1, f"Keyword {kw!r} should trigger filtering")

    def test_filters_nested_reserved_keyword(self):
        loader = _make_loader([_SAMPLE_NESTED_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.filtered_count, 1)

    def test_filters_reserved_keyword_in_system_message(self):
        sample = {
            "schemaVersion": _SCHEMA_VERSION,
            "system": [{"text": "You are Bot: a helpful assistant"}],
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi"}]},
            ],
        }
        loader = _make_loader([sample])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertEqual(result.filtered_count, 1)


class TestSchemaFiltering(unittest.TestCase):
    """Schema validation filters out structurally invalid samples."""

    def test_filters_bad_schema_version(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_BAD_SCHEMA])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(result.total_count, 2)
        self.assertIn("schema", result.filters_applied)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_filters_missing_messages(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_MISSING_MESSAGES])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_schema_and_keyword_both_filter(self):
        """A sample failing schema and another failing keywords — both removed."""
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_BAD_SCHEMA, _SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertEqual(result.filtered_count, 2)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])


class TestUnsupportedTrainingMethodRaises(unittest.TestCase):
    """Unsupported training methods should raise ValueError, not skip silently."""

    def _assert_raises(self, training_method):
        loader = _make_loader([_SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        with self.assertRaises(ValueError) as ctx:
            op.execute(
                loader,
                training_method=training_method,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJ,
            )
        self.assertIn("does not support", str(ctx.exception))

    def test_dpo_lora_raises(self):
        self._assert_raises(TrainingMethod.DPO_LORA)

    def test_dpo_full_raises(self):
        self._assert_raises(TrainingMethod.DPO_FULL)


class TestNonSFTNonDPOKeywordSkipping(unittest.TestCase):
    """CPT, RFT, and Evaluation have schema checks but no keyword checks."""

    def _assert_no_keyword_filtering(self, training_method):
        """Schema validation filters the sample (SFT-format sample is invalid for non-SFT methods),
        but keyword checks are not applied."""
        loader = _make_loader([_SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=training_method,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )

        self.assertEqual(result.status, "SUCCEEDED")
        self.assertIn("schema", result.filters_applied)
        self.assertNotIn("converse_format_reserved_keywords", result.filters_applied)

    def test_cpt_no_keyword_checks(self):
        self._assert_no_keyword_filtering(TrainingMethod.CPT)

    def test_rft_lora_no_keyword_checks(self):
        self._assert_no_keyword_filtering(TrainingMethod.RFT_LORA)

    def test_rft_full_no_keyword_checks(self):
        self._assert_no_keyword_filtering(TrainingMethod.RFT_FULL)

    def test_evaluation_no_keyword_checks(self):
        self._assert_no_keyword_filtering(TrainingMethod.EVALUATION)


class TestValidation(unittest.TestCase):
    def test_missing_training_method_raises(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        with self.assertRaises(ValueError) as ctx:
            op.execute(loader, model=Model.NOVA_LITE_2, platform=Platform.SMTJ)
        self.assertIn("training_method", str(ctx.exception))

    def test_missing_model_raises(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        with self.assertRaises(ValueError) as ctx:
            op.execute(loader, training_method=TrainingMethod.SFT_LORA, platform=Platform.SMTJ)
        self.assertIn("model", str(ctx.exception))

    def test_missing_platform_raises(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        with self.assertRaises(ValueError) as ctx:
            op.execute(loader, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
        self.assertIn("platform", str(ctx.exception))


class TestEnvironmentFiltering(unittest.TestCase):
    """Reserved keyword check applies across all configured platforms."""

    def test_filters_on_all_configured_platforms(self):
        """Check applies on SMTJ, SMHP, and BEDROCK."""
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        for platform in (Platform.SMTJ, Platform.SMHP, Platform.BEDROCK):
            loader = _make_loader([_SAMPLE_WITH_RESERVED])
            result = op.execute(
                loader,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=platform,
            )
            self.assertEqual(result.filtered_count, 1, f"Should filter on {platform.value}")


class TestSMHPFiltering(unittest.TestCase):
    """Reserved keyword filtering on SMHP platform."""

    def test_removes_reserved_keyword_sft_lora(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMHP,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_removes_reserved_keyword_sft_full(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMHP,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [])

    def test_filters_each_reserved_keyword(self):
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        for kw in CONVERSE_FORMAT_RESERVED_KEYWORDS:
            sample = {
                "schemaVersion": _SCHEMA_VERSION,
                "messages": [
                    {"role": "user", "content": [{"text": f"prefix {kw} suffix"}]},
                    {"role": "assistant", "content": [{"text": "ok"}]},
                ],
            }
            loader = _make_loader([sample])
            result = op.execute(
                loader,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMHP,
            )
            self.assertEqual(
                result.filtered_count,
                1,
                f"SMHP: keyword {kw!r} should trigger filtering",
            )

    def test_removes_all_when_all_invalid(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED, _SAMPLE_WITH_IMAGE_TAG, _SAMPLE_WITH_EOS])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMHP,
        )
        self.assertEqual(result.filtered_count, 3)
        self.assertEqual(list(loader.dataset()), [])

    def test_filters_nested_reserved_keyword(self):
        loader = _make_loader([_SAMPLE_NESTED_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMHP,
        )
        self.assertEqual(result.filtered_count, 1)

    def test_keeps_clean_samples(self):
        loader = _make_loader([_CLEAN_SAMPLE, _CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMHP,
        )
        self.assertEqual(result.filtered_count, 0)
        self.assertEqual(len(list(loader.dataset())), 2)


class TestBedrockFiltering(unittest.TestCase):
    """Reserved keyword filtering on BEDROCK platform."""

    def test_removes_reserved_keyword_sft_lora(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_removes_reserved_keyword_sft_full(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_FULL,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )
        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [])

    def test_filters_each_reserved_keyword(self):
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        for kw in CONVERSE_FORMAT_RESERVED_KEYWORDS:
            sample = {
                "schemaVersion": _SCHEMA_VERSION,
                "messages": [
                    {"role": "user", "content": [{"text": f"prefix {kw} suffix"}]},
                    {"role": "assistant", "content": [{"text": "ok"}]},
                ],
            }
            loader = _make_loader([sample])
            result = op.execute(
                loader,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.BEDROCK,
            )
            self.assertEqual(
                result.filtered_count,
                1,
                f"BEDROCK: keyword {kw!r} should trigger filtering",
            )

    def test_removes_all_when_all_invalid(self):
        loader = _make_loader([_SAMPLE_WITH_RESERVED, _SAMPLE_WITH_IMAGE_TAG, _SAMPLE_WITH_EOS])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )
        self.assertEqual(result.filtered_count, 3)
        self.assertEqual(list(loader.dataset()), [])

    def test_filters_nested_reserved_keyword(self):
        loader = _make_loader([_SAMPLE_NESTED_RESERVED])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )
        self.assertEqual(result.filtered_count, 1)

    def test_keeps_clean_samples(self):
        loader = _make_loader([_CLEAN_SAMPLE, _CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(
            loader,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.BEDROCK,
        )
        self.assertEqual(result.filtered_count, 0)
        self.assertEqual(len(list(loader.dataset())), 2)


class TestLoaderFilterIntegration(unittest.TestCase):
    """Tests for loader.filter(method=FilterMethod.INVALID_RECORDS, ...) — the high-level API."""

    def test_filter_removes_invalid_records(self):
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        loader._has_transforms = True
        loader.filter(
            method=FilterMethod.INVALID_RECORDS,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        loader.execute()
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_filter_returns_self_for_chaining(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        loader._has_transforms = True
        result = loader.filter(
            method=FilterMethod.INVALID_RECORDS,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertIs(result, loader)

    def test_filter_warns_without_prior_transform(self):
        """INVALID_RECORDS without transform() logs a warning but does not raise."""
        loader = _make_loader([_CLEAN_SAMPLE])
        with self.assertLogs("nova_forge_sdk", level="WARNING") as cm:
            loader.filter(
                method=FilterMethod.INVALID_RECORDS,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJ,
            )
        self.assertTrue(
            any("verify it is in the expected schema format" in msg for msg in cm.output)
        )
        self.assertEqual(len(loader._pending_operations), 1)

    def test_filter_queued_as_pending(self):
        """INVALID_RECORDS is queued lazily, runs on execute()."""
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        loader._has_transforms = True
        loader.filter(
            method=FilterMethod.INVALID_RECORDS,
            training_method=TrainingMethod.SFT_LORA,
            model=Model.NOVA_LITE_2,
            platform=Platform.SMTJ,
        )
        self.assertEqual(len(loader._pending_operations), 1)
        # Not yet filtered — execute() needed
        self.assertEqual(len(list(loader.dataset())), 2)
        loader.execute()
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])

    def test_filter_without_transform_on_pre_transformed_data(self):
        """load() → filter(INVALID_RECORDS) works when data is already in Converse format."""
        loader = _make_loader([_CLEAN_SAMPLE, _SAMPLE_WITH_RESERVED])
        with self.assertLogs("nova_forge_sdk", level="WARNING"):
            loader.filter(
                method=FilterMethod.INVALID_RECORDS,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                platform=Platform.SMTJ,
            )
        loader.execute()
        self.assertEqual(list(loader.dataset()), [_CLEAN_SAMPLE])


class TestSchemaValidationHelpers(unittest.TestCase):
    """Tests that _sample_fails_schema catches keyword and structural issues via Pydantic."""

    def test_sample_with_keyword_fails_schema(self):
        sample_model = _get_sample_model(TrainingMethod.SFT_LORA, Model.NOVA_LITE_2)
        sample = {
            "schemaVersion": _SCHEMA_VERSION,
            "messages": [
                {"role": "user", "content": [{"text": "has BAD: User: in it"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ],
        }
        self.assertTrue(_sample_fails_schema(sample, sample_model, Model.NOVA_LITE_2, MagicMock()))

    def test_clean_sample_passes_schema(self):
        sample_model = _get_sample_model(TrainingMethod.SFT_LORA, Model.NOVA_LITE_2)
        sample = {
            "schemaVersion": _SCHEMA_VERSION,
            "messages": [
                {"role": "user", "content": [{"text": "all good"}]},
                {"role": "assistant", "content": [{"text": "ok"}]},
            ],
        }
        self.assertFalse(_sample_fails_schema(sample, sample_model, Model.NOVA_LITE_2, MagicMock()))

    def test_missing_required_field_fails_schema(self):
        sample_model = _get_sample_model(TrainingMethod.SFT_LORA, Model.NOVA_LITE_2)
        sample = {"schemaVersion": _SCHEMA_VERSION}
        self.assertTrue(_sample_fails_schema(sample, sample_model, Model.NOVA_LITE_2, MagicMock()))


class TestGetSampleModelBranches(unittest.TestCase):
    """Cover all _get_sample_model branches with specific return type checks."""

    def test_rft_multiturn_lora(self):
        result = _get_sample_model(TrainingMethod.RFT_MULTITURN_LORA, Model.NOVA_LITE_2)
        self.assertIs(result, RFTMultiturnDatasetSample)

    def test_rft_multiturn_full(self):
        result = _get_sample_model(TrainingMethod.RFT_MULTITURN_FULL, Model.NOVA_LITE_2)
        self.assertIs(result, RFTMultiturnDatasetSample)

    def test_evaluation_rft_multiturn_eval(self):
        result = _get_sample_model(
            TrainingMethod.EVALUATION,
            Model.NOVA_LITE_2,
            eval_task=EvaluationTask.RFT_MULTITURN_EVAL,
        )
        self.assertIs(result, RFTMultiturnDatasetSample)

    def test_evaluation_default(self):
        result = _get_sample_model(TrainingMethod.EVALUATION, Model.NOVA_LITE_2)
        self.assertIs(result, EvalDatasetSample)

    def test_unknown_training_method_returns_none(self):
        fake_method = MagicMock()
        result = _get_sample_model(fake_method, Model.NOVA_LITE_2)
        self.assertIsNone(result)


class TestHelpers(unittest.TestCase):
    def test_get_applicable_checks_sft(self):
        checks = _get_applicable_checks(TrainingMethod.SFT_LORA, Platform.SMTJ)
        self.assertTrue(len(checks) > 0)

    def test_get_applicable_checks_cpt_no_multimodal(self):
        checks = _get_applicable_checks(TrainingMethod.CPT, Platform.SMTJ)
        multimodal = [c for c in checks if c["type"] in ("image", "video")]
        self.assertEqual(len(multimodal), 0)


# ---------------------------------------------------------------------------
# Multimodal test helpers
# ---------------------------------------------------------------------------


def _make_converse_image_sample(uri="s3://bucket/img.png"):
    return {
        "schemaVersion": _SCHEMA_VERSION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Describe this"},
                    {
                        "image": {
                            "format": "png",
                            "source": {"s3Location": {"uri": uri, "bucketOwner": "123"}},
                        }
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "A cat."}]},
        ],
    }


def _make_openai_image_sample(url="s3://bucket/img.png"):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ],
    }


def _make_video_sample(uri="s3://bucket/vid.mp4"):
    return {
        "schemaVersion": _SCHEMA_VERSION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "Describe this"},
                    {
                        "video": {
                            "format": "mp4",
                            "source": {"s3Location": {"uri": uri, "bucketOwner": "123"}},
                        }
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "A video."}]},
        ],
    }


def _make_multi_image_sample(count):
    images = [
        {
            "image": {
                "format": "png",
                "source": {
                    "s3Location": {
                        "uri": f"s3://bucket/img{i}.png",
                        "bucketOwner": "123",
                    }
                },
            }
        }
        for i in range(count)
    ]
    return {
        "schemaVersion": _SCHEMA_VERSION,
        "messages": [
            {"role": "user", "content": [{"text": "Describe"}] + images},
            {"role": "assistant", "content": [{"text": "Done."}]},
        ],
    }


# ---------------------------------------------------------------------------
# End-to-end filter operation tests for multimodal
# ---------------------------------------------------------------------------


@patch("amzn_nova_forge.dataset.operations.invalid_records_filter_operation.boto3")
@patch("amzn_nova_forge.dataset.dataset_validator.dataset_validator.boto3")
class TestMultimodalFilterEndToEnd(unittest.TestCase):
    def test_removes_oversized_image(self, mock_dv_boto3, mock_op_boto3):
        s3 = MagicMock()
        for m in (mock_dv_boto3, mock_op_boto3):
            m.client.return_value = s3
        s3.head_object.return_value = {"ContentLength": MAX_IMAGE_FILE_SIZE_BYTES + 1}

        loader = _make_loader([_make_converse_image_sample()])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(loader, **_SFT_KWARGS)

        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(list(loader.dataset()), [])

    def test_keeps_valid_image(self, mock_dv_boto3, mock_op_boto3):
        s3 = MagicMock()
        for m in (mock_dv_boto3, mock_op_boto3):
            m.client.return_value = s3
        s3.head_object.return_value = {"ContentLength": 5 * 1024 * 1024}

        loader = _make_loader([_make_converse_image_sample()])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(loader, **_SFT_KWARGS)

        self.assertEqual(result.filtered_count, 0)
        self.assertEqual(len(list(loader.dataset())), 1)

    def test_removes_oversized_video(self, mock_dv_boto3, mock_op_boto3):
        s3 = MagicMock()
        for m in (mock_dv_boto3, mock_op_boto3):
            m.client.return_value = s3
        s3.head_object.return_value = {"ContentLength": MAX_VIDEO_FILE_SIZE_BYTES + 1}

        loader = _make_loader([_make_video_sample()])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(loader, **_SFT_KWARGS)

        self.assertEqual(result.filtered_count, 1)

    def test_removes_too_many_images(self, mock_dv_boto3, mock_op_boto3):
        s3 = MagicMock()
        for m in (mock_dv_boto3, mock_op_boto3):
            m.client.return_value = s3
        s3.head_object.return_value = {"ContentLength": 1024}

        loader = _make_loader([_make_multi_image_sample(MAX_IMAGES_PER_MESSAGE + 1)])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(loader, **_SFT_KWARGS)

        self.assertEqual(result.filtered_count, 1)

    def test_mixed_samples_filters_only_invalid(self, mock_dv_boto3, mock_op_boto3):
        s3 = MagicMock()
        for m in (mock_dv_boto3, mock_op_boto3):
            m.client.return_value = s3
        s3.head_object.side_effect = [
            {"ContentLength": MAX_IMAGE_FILE_SIZE_BYTES + 1},
            {"ContentLength": 1024},
        ]

        loader = _make_loader(
            [
                _make_converse_image_sample("s3://b/big.png"),
                _make_converse_image_sample("s3://b/small.png"),
            ]
        )
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        result = op.execute(loader, **_SFT_KWARGS)

        self.assertEqual(result.filtered_count, 1)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(len(list(loader.dataset())), 1)


if __name__ == "__main__":
    unittest.main()
