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
"""Tests for filter operation behavior (FilterOperationResult, ordering, counts, Glue summary)."""

import json
from typing import Any, Tuple, Type
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.dataset_validator.dataset_validator import (
    InfrastructureError,
)
from amzn_nova_forge.dataset.operations.base import (
    FilterOperationResult,
)
from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
    DefaultTextFilterOperation,
)
from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import (
    ExactDedupFilterOperation,
)
from amzn_nova_forge.dataset.operations.filter_operation import (
    FilterMethod,
    NovaForgeFilterOperationBase,
    get_filter_operation,
)
from amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation import (
    FuzzyDedupFilterOperation,
)
from amzn_nova_forge.dataset.operations.transform_operation import TransformMethod
from amzn_nova_forge.model.model_enums import Model, Platform, TrainingMethod


class _StubFilter(NovaForgeFilterOperationBase):
    """Concrete stub for testing the base class."""

    _FILTER_NAME = "Stub Filter"

    def __init__(self, result=None):
        self._result = result

    def get_supported_runtimes(self) -> Tuple[Type, ...]:
        return ()

    def execute(self, loader: Any, **kwargs: Any) -> FilterOperationResult:
        return self._result


class TestLogCompleteWarning:
    def test_warns_on_100_percent_drop(self, caplog):
        result = FilterOperationResult(
            status="SUCCEEDED", output_state=None, total_count=50, filtered_count=50
        )
        op = _StubFilter(result=result)
        op._log_complete("s3://bucket/output/", result)
        assert any("All 50 records were removed" in msg for msg in caplog.messages)

    def test_no_warning_when_some_kept(self, caplog):
        result = FilterOperationResult(
            status="SUCCEEDED", output_state=None, total_count=50, filtered_count=10
        )
        op = _StubFilter(result=result)
        op._log_complete("s3://bucket/output/", result)
        assert not any("All" in msg and "removed" in msg for msg in caplog.messages)

    def test_no_warning_when_counts_unavailable(self, caplog):
        result = FilterOperationResult(
            status="SUCCEEDED", output_state=None, total_count=0, filtered_count=0
        )
        op = _StubFilter(result=result)
        op._log_complete("s3://bucket/output/", result)
        assert not any("All" in msg and "removed" in msg for msg in caplog.messages)


class TestFilterOperationResultStr:
    def test_str_with_counts(self):
        result = FilterOperationResult(
            status="SUCCEEDED", output_state=None, total_count=1000, filtered_count=153
        )
        assert str(result) == "1000 records → 847 kept, 153 dropped"

    def test_str_zero_total(self):
        result = FilterOperationResult(
            status="SUCCEEDED", output_state=None, total_count=0, filtered_count=0
        )
        assert str(result) == "counts unknown"


class TestFilterOrderingIndependence:
    """filter() accepts any FilterMethod regardless of whether transform() has been called."""

    def test_filter_accepts_any_method_without_transform(self):
        loader = JSONLDatasetLoader()
        loader._load_path = "s3://bucket/data.jsonl"
        loader.dataset = lambda: iter([])

        for method in FilterMethod:
            loader.filter(method=method)

        assert len(loader._pending_operations) == len(FilterMethod)
        queued_methods = [m for _, m, _ in loader._pending_operations]
        assert set(queued_methods) == set(FilterMethod)

    def test_filter_accepts_any_method_after_transform(self):
        loader = JSONLDatasetLoader()
        loader._load_path = "s3://bucket/data.jsonl"
        loader.dataset = lambda: iter([])

        loader.transform(
            method=TransformMethod.SCHEMA,
            training_method=MagicMock(),
            model=MagicMock(),
        )

        for method in FilterMethod:
            loader.filter(method=method)

        # 1 transform + N filters
        assert len(loader._pending_operations) == 1 + len(FilterMethod)


_SCHEMA_VERSION = "bedrock-conversation-2024"

_CLEAN_SAMPLE = {
    "schemaVersion": _SCHEMA_VERSION,
    "messages": [
        {"role": "user", "content": [{"text": "What is 2+2?"}]},
        {"role": "assistant", "content": [{"text": "4"}]},
    ],
}

_BAD_SAMPLE = {
    "schemaVersion": "wrong-version",
    "messages": [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
    ],
}

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


_BOTO3_PATH = "amzn_nova_forge.dataset.operations.invalid_records_filter_operation.boto3"
_FAILS_SCHEMA_PATH = (
    "amzn_nova_forge.dataset.operations.invalid_records_filter_operation._sample_fails_schema"
)


class TestInvalidRecordsCountConsistency:
    """Counts are consistent: total == kept + filtered, and match actual dataset."""

    @pytest.fixture(autouse=True)
    def _patch_boto3(self):
        with patch(_BOTO3_PATH) as mock_boto:
            mock_boto.client.return_value = MagicMock()
            yield

    @pytest.mark.parametrize(
        "samples, expected_filtered",
        [
            ([_CLEAN_SAMPLE, _BAD_SAMPLE, _CLEAN_SAMPLE], 1),
            ([_CLEAN_SAMPLE, _CLEAN_SAMPLE], 0),
            ([_BAD_SAMPLE, _BAD_SAMPLE], 2),
            ([], 0),
        ],
        ids=["mixed", "all-valid", "all-invalid", "empty"],
    )
    def test_counts_match_dataset(self, samples, expected_filtered):
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)
        loader = _make_loader(samples)
        result = op.execute(loader, **_SFT_KWARGS)
        kept = list(loader.dataset())

        assert result.total_count == len(samples)
        assert result.filtered_count == expected_filtered
        assert len(kept) == result.total_count - result.filtered_count


class TestInvalidRecordsPerSampleHandling:
    """InfrastructureError and unexpected exceptions propagate through the generator."""

    @pytest.fixture(autouse=True)
    def _patch_boto3(self):
        with patch(_BOTO3_PATH) as mock_boto:
            mock_boto.client.return_value = MagicMock()
            yield

    def test_infrastructure_error_propagates(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)

        with patch(_FAILS_SCHEMA_PATH, side_effect=InfrastructureError("S3 unreachable")):
            op.execute(loader, **_SFT_KWARGS)
            with pytest.raises(InfrastructureError, match="S3 unreachable"):
                list(loader.dataset())

    def test_unexpected_exception_propagates(self):
        loader = _make_loader([_CLEAN_SAMPLE])
        op = get_filter_operation(FilterMethod.INVALID_RECORDS)

        with patch(_FAILS_SCHEMA_PATH, side_effect=TypeError("unexpected")):
            op.execute(loader, **_SFT_KWARGS)
            with pytest.raises(TypeError, match="unexpected"):
                list(loader.dataset())


_GLUE_STATE = DataState(
    path="s3://bucket/input/data.jsonl",
    format="jsonl",
    location=DataLocation.S3,
)

_EXACT_DEDUP_RELOAD = (
    "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._reload_output_into_loader"
)
_EXACT_DEDUP_READ_SUMMARY = (
    "amzn_nova_forge.dataset.operations.exact_dedup_filter_operation._read_summary_json"
)
_FUZZY_DEDUP_RELOAD = (
    "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._reload_output_into_loader"
)
_FUZZY_DEDUP_READ_SUMMARY = (
    "amzn_nova_forge.dataset.operations.fuzzy_dedup_filter_operation._read_summary_json"
)
_DEFAULT_TEXT_RELOAD = (
    "amzn_nova_forge.dataset.operations.default_text_filter_operation._reload_output_into_loader"
)


def _make_glue_manager(side_effect=None):
    mgr = MagicMock()
    mgr.runtime_name = "MockGlue"
    if side_effect is not None:
        mgr.execute.side_effect = side_effect
    else:
        mgr.execute.return_value = "jr_123"
    return mgr


class TestGlueRuntimeErrorWrapping:
    """RuntimeError from manager.execute() propagates."""

    _RELOAD_PATHS = {
        DefaultTextFilterOperation: _DEFAULT_TEXT_RELOAD,
        ExactDedupFilterOperation: _EXACT_DEDUP_RELOAD,
        FuzzyDedupFilterOperation: _FUZZY_DEDUP_RELOAD,
    }

    @pytest.mark.parametrize(
        "op_cls",
        [DefaultTextFilterOperation, ExactDedupFilterOperation, FuzzyDedupFilterOperation],
        ids=["default_text", "exact_dedup", "fuzzy_dedup"],
    )
    def test_wraps_runtime_error(self, op_cls):
        op = op_cls()
        mgr = _make_glue_manager(side_effect=RuntimeError("job exploded"))

        with patch.object(op, "_resolve_runtime_manager", return_value=mgr):
            with patch(self._RELOAD_PATHS[op_cls]):
                with pytest.raises(RuntimeError, match="job exploded"):
                    op.execute(loader=None, state=_GLUE_STATE, output_path="s3://bucket/output/")


class TestGlueSummaryJsonReading:
    """FilterOperationResult counts are populated from _summary.json via the base helper."""

    def test_default_text_filter_returns_zero_counts(self):
        op = DefaultTextFilterOperation()
        mgr = _make_glue_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=mgr):
            with patch(_DEFAULT_TEXT_RELOAD):
                result = op.execute(
                    loader=None, state=_GLUE_STATE, output_path="s3://bucket/output/"
                )

        assert isinstance(result, FilterOperationResult)
        assert result.total_count == 0
        assert result.filtered_count == 0
        assert result.status == "SUCCEEDED"

    def test_exact_dedup_reads_summary(self):
        op = ExactDedupFilterOperation()
        mgr = _make_glue_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=mgr):
            with patch(_EXACT_DEDUP_RELOAD):
                with patch(_EXACT_DEDUP_READ_SUMMARY, return_value=(500, 42)):
                    result = op.execute(
                        loader=None, state=_GLUE_STATE, output_path="s3://bucket/output/"
                    )

        assert result.total_count == 500
        assert result.filtered_count == 42

    def test_fuzzy_dedup_reads_summary(self):
        op = FuzzyDedupFilterOperation()
        mgr = _make_glue_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=mgr):
            with patch(_FUZZY_DEDUP_RELOAD):
                with patch(_FUZZY_DEDUP_READ_SUMMARY, return_value=(1000, 150)):
                    result = op.execute(
                        loader=None, state=_GLUE_STATE, output_path="s3://bucket/output/"
                    )

        assert result.total_count == 1000
        assert result.filtered_count == 150

    @pytest.mark.parametrize(
        "op_cls, reload_path, summary_path",
        [
            (ExactDedupFilterOperation, _EXACT_DEDUP_RELOAD, _EXACT_DEDUP_READ_SUMMARY),
            (FuzzyDedupFilterOperation, _FUZZY_DEDUP_RELOAD, _FUZZY_DEDUP_READ_SUMMARY),
        ],
        ids=["exact_dedup", "fuzzy_dedup"],
    )
    def test_missing_summary_falls_back_to_zero(self, op_cls, reload_path, summary_path):
        op = op_cls()
        mgr = _make_glue_manager()

        with patch.object(op, "_resolve_runtime_manager", return_value=mgr):
            with patch(reload_path):
                with patch(summary_path, return_value=(0, 0)):
                    result = op.execute(
                        loader=None, state=_GLUE_STATE, output_path="s3://bucket/output/"
                    )

        assert result.total_count == 0
        assert result.filtered_count == 0


_BOTO3_FILTER_OP = "amzn_nova_forge.dataset.operations.filter_operation.boto3"


class TestReadSummaryJson:
    """Direct tests for _read_summary_json with mocked boto3."""

    def test_happy_path(self):
        from amzn_nova_forge.dataset.operations.filter_operation import _read_summary_json

        body = json.dumps({"input_count": 500, "duplicates_removed": 42}).encode()
        mock_resp = {"Body": MagicMock(read=MagicMock(return_value=body))}
        with patch(_BOTO3_FILTER_OP) as mock_boto:
            mock_boto.client.return_value.get_object.return_value = mock_resp
            total, filtered = _read_summary_json("s3://bucket/output")
        assert (total, filtered) == (500, 42)

    def test_missing_keys_default_to_zero(self):
        from amzn_nova_forge.dataset.operations.filter_operation import _read_summary_json

        body = json.dumps({}).encode()
        mock_resp = {"Body": MagicMock(read=MagicMock(return_value=body))}
        with patch(_BOTO3_FILTER_OP) as mock_boto:
            mock_boto.client.return_value.get_object.return_value = mock_resp
            assert _read_summary_json("s3://bucket/output") == (0, 0)

    def test_s3_error_falls_back_to_zero(self):
        from amzn_nova_forge.dataset.operations.filter_operation import _read_summary_json

        with patch(_BOTO3_FILTER_OP) as mock_boto:
            mock_boto.client.return_value.get_object.side_effect = Exception("denied")
            assert _read_summary_json("s3://bucket/output") == (0, 0)

    def test_malformed_json_falls_back_to_zero(self):
        from amzn_nova_forge.dataset.operations.filter_operation import _read_summary_json

        mock_resp = {"Body": MagicMock(read=MagicMock(return_value=b"not json"))}
        with patch(_BOTO3_FILTER_OP) as mock_boto:
            mock_boto.client.return_value.get_object.return_value = mock_resp
            assert _read_summary_json("s3://bucket/output") == (0, 0)
