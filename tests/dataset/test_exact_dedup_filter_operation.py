"""Unit tests for ExactDedupFilterOperation."""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import (
    _PIPELINE_ID,
    ExactDedupFilterOperation,
)
from amzn_nova_forge.dataset.operations.filter_operation import (
    FilterMethod,
    get_filter_operation,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig


def _s3_state(path="s3://test-bucket/in", fmt="parquet"):
    return DataState(path=path, format=fmt, location=DataLocation.S3)


class TestExactDedupFilterOperationFactory(unittest.TestCase):
    """Tests for FilterMethod.EXACT_DEDUP and the factory."""

    def test_factory_returns_exact_dedup_instance(self):
        op = get_filter_operation(FilterMethod.EXACT_DEDUP)
        self.assertIsInstance(op, ExactDedupFilterOperation)

    def test_enum_value(self):
        self.assertEqual(FilterMethod.EXACT_DEDUP.value, "exact_dedup_filter")


class TestExactDedupFilterOperationSupportedRuntimes(unittest.TestCase):
    def test_supported_runtimes_includes_glue(self):
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager

        op = ExactDedupFilterOperation()
        runtimes = op.get_supported_runtimes()
        self.assertIn(GlueRuntimeManager, runtimes)


class TestExactDedupFilterOperationExecute(unittest.TestCase):
    """Tests for execute() parameter forwarding and config construction."""

    def _make_mock_manager(self):
        manager = MagicMock()
        manager.execute.return_value = "jr_test_run_id"
        return manager

    def test_pipeline_id_cannot_be_overridden(self):
        """Verify pipeline_id is set after extra_args merge (not before)."""
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                    extra_args={"pipeline_id": "malicious_override"},
                )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)

    def test_builds_correct_job_config(self):
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ):
                result = op.execute(
                    loader=None,
                    state=_s3_state("s3://bucket/input/"),
                    input_path="s3://bucket/input/",
                    output_path="s3://bucket/output/",
                    input_format="jsonl",
                    output_format="jsonl",
                    text_field="body",
                )

        job_config = manager.execute.call_args[0][0]
        self.assertIsInstance(job_config, DataPrepJobConfig)
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)
        self.assertEqual(job_config.data_s3_path, "s3://bucket/input/")
        self.assertEqual(job_config.output_s3_path, "s3://bucket/output/")
        self.assertEqual(job_config.text_field, "body")
        self.assertEqual(result.status, "SUCCEEDED")

    def test_auto_generated_job_name_uses_pipeline_id(self):
        """Auto-generated job name includes the pipeline id with dashes."""
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ):
                with patch(
                    "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation.time.time",
                    return_value=1700000000,
                ):
                    op.execute(
                        loader=None,
                        state=_s3_state(),
                        input_path="s3://test-bucket/in",
                        output_path="s3://test-bucket/out",
                    )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.job_name, "nova-forge-exact-dedup-filter-1700000000")

    def test_explicit_job_name_is_preserved(self):
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                    job_name="my-custom-job",
                )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.job_name, "my-custom-job")

    def test_reloads_loader_when_provided(self):
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()
        mock_loader = MagicMock()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ) as mock_reload:
                op.execute(
                    loader=mock_loader,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                )

        mock_reload.assert_called_once_with(mock_loader, "s3://test-bucket/out", "jsonl")

    def test_skips_reload_when_loader_is_none(self):
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ) as mock_reload:
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                )

        mock_reload.assert_not_called()

    def test_extra_args_are_forwarded(self):
        op = ExactDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                    extra_args={"custom_key": "custom_value"},
                )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.extra_args["custom_key"], "custom_value")
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)


if __name__ == "__main__":
    unittest.main()
