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
"""Unit tests for FuzzyDedupFilterOperation."""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.filter_operation import (
    FilterMethod,
    get_filter_operation,
)
from amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation import (
    _FUZZY_DEDUP_PARAM_KEYS,
    _PIPELINE_ID,
    FuzzyDedupFilterOperation,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig


def _s3_state(path="s3://test-bucket/in", fmt="parquet"):
    return DataState(path=path, format=fmt, location=DataLocation.S3)


class TestFuzzyDedupFilterOperationFactory(unittest.TestCase):
    """Tests for FilterMethod.FUZZY_DEDUP and the factory."""

    def test_factory_returns_fuzzy_dedup_instance(self):
        op = get_filter_operation(FilterMethod.FUZZY_DEDUP)
        self.assertIsInstance(op, FuzzyDedupFilterOperation)

    def test_enum_value(self):
        self.assertEqual(FilterMethod.FUZZY_DEDUP.value, "fuzzy_dedup")


class TestFuzzyDedupFilterOperationSupportedRuntimes(unittest.TestCase):
    def test_supported_runtimes_includes_smtj_and_glue(self):
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeManager

        op = FuzzyDedupFilterOperation()
        runtimes = op.get_supported_runtimes()
        self.assertIn(SMTJRuntimeManager, runtimes)
        self.assertIn(GlueRuntimeManager, runtimes)


class TestFuzzyDedupFilterOperationExecute(unittest.TestCase):
    """Tests for execute() parameter forwarding and config construction."""

    def _make_mock_manager(self):
        manager = MagicMock()
        manager.execute.return_value = "jr_test_run_id"
        return manager

    def test_pipeline_id_cannot_be_overridden(self):
        """Verify pipeline_id is set after extra_args merge (not before)."""
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
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

    def test_forwards_algorithm_params(self):
        """Verify all _FUZZY_DEDUP_PARAM_KEYS are forwarded to extra_args."""
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        kwargs = {
            "input_path": "s3://test-bucket/in",
            "output_path": "s3://test-bucket/out",
            "state": _s3_state(),
            "num_perm": 128,
            "ngram_size": 16,
            "jaccard_threshold": 0.7,
            "num_bands": 10,
            "rows_per_band": 5,
            "bands_per_iteration": 2,
            "seed": 99,
            "lowercase": False,
        }

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(loader=None, **kwargs)

        job_config = manager.execute.call_args[0][0]
        for key in _FUZZY_DEDUP_PARAM_KEYS:
            self.assertEqual(
                job_config.extra_args[key],
                kwargs[key],
                f"Parameter {key!r} not forwarded correctly",
            )

    def test_builds_correct_job_config(self):
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ):
                result = op.execute(
                    loader=None,
                    state=_s3_state("s3://bucket/input/"),
                    input_path="s3://bucket/input/",
                    output_path="s3://bucket/output/",
                    input_format="parquet",
                    output_format="parquet",
                    text_field="body",
                )

        job_config = manager.execute.call_args[0][0]
        self.assertIsInstance(job_config, DataPrepJobConfig)
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)
        self.assertEqual(job_config.data_s3_path, "s3://bucket/input/")
        self.assertEqual(job_config.output_s3_path, "s3://bucket/output/")
        self.assertEqual(job_config.text_field, "body")
        self.assertEqual(result.status, "SUCCEEDED")

    def test_reloads_loader_when_provided(self):
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()
        mock_loader = MagicMock()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ) as mock_reload:
                op.execute(
                    loader=mock_loader,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                )

        mock_reload.assert_called_once_with(mock_loader, "s3://test-bucket/out", "parquet")

    def test_skips_reload_when_loader_is_none(self):
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ) as mock_reload:
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                )

        mock_reload.assert_not_called()

    def test_threshold_alias_forwards_to_jaccard_threshold(self):
        """Passing threshold= (without jaccard_threshold) sets jaccard_threshold in extra_args."""
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                    threshold=0.7,
                )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.extra_args["jaccard_threshold"], 0.7)

    def test_explicit_jaccard_threshold_takes_precedence(self):
        """When both threshold and jaccard_threshold are passed, jaccard_threshold wins."""
        op = FuzzyDedupFilterOperation()
        manager = self._make_mock_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=manager):
            with patch(
                "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
            ):
                op.execute(
                    loader=None,
                    state=_s3_state(),
                    input_path="s3://test-bucket/in",
                    output_path="s3://test-bucket/out",
                    threshold=0.5,
                    jaccard_threshold=0.9,
                )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.extra_args["jaccard_threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()
