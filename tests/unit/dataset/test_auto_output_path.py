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
"""Unit tests for auto-derived output paths when input is not on S3.

Verifies that filter operations auto-derive S3 output paths using the
default data-prep bucket when no explicit output_path is provided and
input is local or HuggingFace.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge.core.enums import FilterMethod
from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.base import OperationResult

FAKE_DATAPREP_BUCKET = "sagemaker-forge-dataprep-123456789012-us-east-1"
_SESSION_RE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"


def _assert_auto_derived_path(output_path, expected_parent, expected_stem, expected_method):
    """Assert output_path matches ``<parent>/<stem>/<session>/<method>_output/``."""
    parent = expected_parent.rstrip("/")
    pattern = (
        rf"^{re.escape(parent)}/{re.escape(expected_stem)}"
        rf"/{_SESSION_RE}/{re.escape(expected_method)}_output/$"
    )
    assert re.match(pattern, output_path), (
        f"Auto-derived output_path does not match expected pattern.\n"
        f"  Got:      '{output_path}'\n"
        f"  Expected: '{parent}/{expected_stem}/<session>/{expected_method}_output/'"
    )


def _s3_state(path="s3://bucket/out/", fmt="parquet"):
    """Shorthand for an S3 DataState used as a mock operation result."""
    return DataState(path=path, format=fmt, location=DataLocation.S3, generator=lambda: iter([]))


def _mock_op(*output_paths):
    """Build a mock filter operation returning successive S3 output states."""
    op = MagicMock()
    op.execute.side_effect = [
        OperationResult(status="SUCCEEDED", output_state=_s3_state(p)) for p in output_paths
    ]
    return op


class TestPrepareInputGuardsDirectCalls:
    """prepare_input() without output_path raises ValueError for non-S3 inputs.

    In normal flow, execute() provides output_path via OutputPathResolver.
    These tests guard against direct misuse of prepare_input().
    """

    @pytest.fixture()
    def filter_op(self):
        from amzn_nova_forge.dataset.operations.filter_operation import get_filter_operation

        return get_filter_operation(FilterMethod.DEFAULT_TEXT_FILTER)

    @pytest.mark.parametrize(
        "path, fmt, location",
        [
            ("/home/user/data/train.jsonl", "jsonl", DataLocation.LOCAL),
            ("/home/user/data/train.csv", "csv", DataLocation.LOCAL),
            ("hf://fake-org/fake-dataset/train", "huggingface", DataLocation.HUGGINGFACE),
        ],
        ids=["local-jsonl", "local-csv", "huggingface"],
    )
    def test_non_s3_input_without_output_path_raises(self, filter_op, path, fmt, location):
        """Direct call to prepare_input() without output_path still raises ValueError."""
        state = DataState(
            path=path,
            format=fmt,
            location=location,
            generator=lambda: iter([{"text": "hello"}]),
        )
        with pytest.raises(ValueError, match="require an S3 output path"):
            filter_op.prepare_input(state)

    def test_s3_input_passes_through(self, filter_op):
        """S3 input in compatible format → returned unchanged, no upload."""
        state = DataState(
            path="s3://my-bucket/data/train.jsonl",
            format="jsonl",
            location=DataLocation.S3,
            generator=lambda: iter([{"text": "hello"}]),
        )

        result = filter_op.prepare_input(state)

        assert result is state

    def test_explicit_output_path_still_honored(self, filter_op):
        """Explicit output_path → used as S3 base (backward compat)."""
        state = DataState(
            path="/home/user/data/train.jsonl",
            format="jsonl",
            location=DataLocation.LOCAL,
            generator=lambda: iter([{"text": "hello"}]),
        )

        with patch(
            "amzn_nova_forge.dataset.operations.filter_operation.upload_local_file_to_s3",
            return_value="s3://explicit-bucket/uploaded.jsonl",
        ) as mock_upload:
            filter_op.prepare_input(state, output_path="s3://explicit-bucket/filtered/")

        s3_base = mock_upload.call_args[0][1]
        assert s3_base.startswith("s3://explicit-bucket"), (
            f"Explicit output_path should be used, got: '{s3_base}'"
        )


class TestExecuteAutoDerivesOutputPath:
    """execute() should produce S3 output paths for non-S3 inputs.

    Auto-derived path format: s3://<dataprep-bucket>/<stem>/<session>/<method>_output/
    """

    @staticmethod
    def _execute_single_filter(load_path, method=FilterMethod.DEFAULT_TEXT_FILTER):
        """Queue one filter on a stub loader, execute with a mock op, return the output_path."""
        from amzn_nova_forge.dataset.jsonl_dataset_loader import JSONLDatasetLoader

        op = _mock_op(f"s3://{FAKE_DATAPREP_BUCKET}/out/")
        loader = JSONLDatasetLoader()
        loader._load_path = load_path
        loader.dataset = lambda: iter([{"text": "hello"}])
        loader.filter(method=method, text_field="text")

        with (
            patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation", return_value=op),
            patch(
                "amzn_nova_forge.dataset.data_state.get_dataprep_bucket_name",
                return_value=FAKE_DATAPREP_BUCKET,
            ),
            patch("amzn_nova_forge.dataset.data_state.ensure_bucket_exists"),
        ):
            loader.execute()

        return op.execute.call_args.kwargs["output_path"]

    @pytest.mark.parametrize(
        "load_path, expected_parent",
        [
            ("/home/user/data/train.jsonl", f"s3://{FAKE_DATAPREP_BUCKET}"),
            ("hf://fake-org/fake-dataset/train", f"s3://{FAKE_DATAPREP_BUCKET}"),
        ],
        ids=["local", "huggingface"],
    )
    def test_non_s3_input(self, load_path, expected_parent):
        """Non-S3 load path → output under data-prep bucket."""
        output_path = self._execute_single_filter(load_path)
        _assert_auto_derived_path(
            output_path,
            expected_parent=expected_parent,
            expected_stem="train",
            expected_method="default_text_filter",
        )

    def test_s3_input_uses_load_path_parent(self):
        """S3 load path → output derived from the S3 parent (existing behavior)."""
        output_path = self._execute_single_filter("s3://my-bucket/data/train.jsonl")
        _assert_auto_derived_path(
            output_path,
            expected_parent="s3://my-bucket/data",
            expected_stem="train",
            expected_method="default_text_filter",
        )


class TestThreeFilterChainOutputPathThreading:
    """Verify output_path and state.path threading across three-filter chains.

    Key rules:
    - output_path: auto-derived from resolver (anchored to load path), or
      explicit if user-provided. Explicit paths do NOT shift the resolver base.
    - state.path (input to next op): always the previous operation's output.
    """

    @staticmethod
    def _execute_three_filters(load_path, filter_specs):
        """Queue three filters, execute, return list of (output_path, input_state_path) per call.

        ``filter_specs`` is a list of (method, explicit_output_path_or_None, mock_result_path).
        """
        from amzn_nova_forge.dataset.jsonl_dataset_loader import JSONLDatasetLoader

        mock_op = _mock_op(*[spec[2] for spec in filter_specs])

        loader = JSONLDatasetLoader()
        loader._load_path = load_path
        loader.dataset = lambda: iter([{"text": "hello"}])

        for method, explicit_path, _ in filter_specs:
            kwargs = {"text_field": "text"}
            if explicit_path is not None:
                kwargs["output_path"] = explicit_path
            loader.filter(method=method, **kwargs)

        with (
            patch(
                "amzn_nova_forge.dataset.dataset_loader.get_filter_operation", return_value=mock_op
            ),
            patch(
                "amzn_nova_forge.dataset.data_state.get_dataprep_bucket_name",
                return_value=FAKE_DATAPREP_BUCKET,
            ),
            patch("amzn_nova_forge.dataset.data_state.ensure_bucket_exists"),
        ):
            loader.execute()

        return [
            (call.kwargs["output_path"], call.kwargs["state"].path)
            for call in mock_op.execute.call_args_list
        ]

    @pytest.mark.parametrize(
        "load_path",
        ["/home/user/data/train.jsonl", "hf://fake-org/fake-dataset/train"],
        ids=["local", "huggingface"],
    )
    def test_auto_explicit_auto(self, load_path):
        """f1(auto) → f2(explicit) → f3(auto): f3 output from resolver, f3 input from f2."""
        dp = f"s3://{FAKE_DATAPREP_BUCKET}"
        calls = self._execute_three_filters(
            load_path,
            [
                (FilterMethod.DEFAULT_TEXT_FILTER, None, f"{dp}/f1-out/"),
                (FilterMethod.EXACT_DEDUP, "s3://user-bucket/custom/", "s3://user-bucket/custom/"),
                (FilterMethod.FUZZY_DEDUP, None, f"{dp}/f3-out/"),
            ],
        )

        f1_output, _ = calls[0]
        f2_output, _ = calls[1]
        f3_output, f3_input = calls[2]

        # f1: auto-derived under data-prep bucket
        _assert_auto_derived_path(f1_output, dp, "train", "default_text_filter")

        # f2: explicit path honoured
        assert f2_output == "s3://user-bucket/custom/"

        # f3: auto-derived under data-prep bucket (NOT relative to f2)
        _assert_auto_derived_path(f3_output, dp, "train", "fuzzy_dedup")
        assert "user-bucket" not in f3_output

        # f3 reads from f2's output
        assert f3_input == "s3://user-bucket/custom/"

    @pytest.mark.parametrize(
        "load_path",
        ["/home/user/data/train.jsonl", "hf://fake-org/fake-dataset/train"],
        ids=["local", "huggingface"],
    )
    def test_auto_explicit_explicit(self, load_path):
        """f1(auto) → f2(explicit A) → f3(explicit B): each uses its own path."""
        dp = f"s3://{FAKE_DATAPREP_BUCKET}"
        calls = self._execute_three_filters(
            load_path,
            [
                (FilterMethod.DEFAULT_TEXT_FILTER, None, f"{dp}/f1-out/"),
                (FilterMethod.EXACT_DEDUP, "s3://user-bucket/dedup/", "s3://user-bucket/dedup/"),
                (FilterMethod.FUZZY_DEDUP, "s3://user-bucket/fuzzy/", "s3://user-bucket/fuzzy/"),
            ],
        )

        f1_output, _ = calls[0]
        f2_output, _ = calls[1]
        f3_output, f3_input = calls[2]

        # f1: auto-derived
        _assert_auto_derived_path(f1_output, dp, "train", "default_text_filter")

        # f2: its own explicit path
        assert f2_output == "s3://user-bucket/dedup/"

        # f3: its own explicit path (NOT f2's)
        assert f3_output == "s3://user-bucket/fuzzy/"

        # f3 reads from f2's output
        assert f3_input == "s3://user-bucket/dedup/"

    @pytest.mark.parametrize(
        "load_path",
        ["/home/user/data/train.jsonl", "hf://fake-org/fake-dataset/train"],
        ids=["local", "huggingface"],
    )
    def test_explicit_auto(self, load_path):
        """f1(explicit) → f2(auto): f2 output from resolver (data-prep bucket), f2 input from f1."""
        dp = f"s3://{FAKE_DATAPREP_BUCKET}"
        calls = self._execute_three_filters(
            load_path,
            [
                (
                    FilterMethod.DEFAULT_TEXT_FILTER,
                    "s3://user-bucket/first/",
                    "s3://user-bucket/first/",
                ),
                (FilterMethod.EXACT_DEDUP, None, f"{dp}/f2-out/"),
                (FilterMethod.FUZZY_DEDUP, None, f"{dp}/f3-out/"),
            ],
        )

        f1_output, _ = calls[0]
        f2_output, f2_input = calls[1]
        f3_output, f3_input = calls[2]

        # f1: explicit path honoured
        assert f1_output == "s3://user-bucket/first/"

        # f2: auto-derived under data-prep bucket (NOT relative to f1's explicit path)
        _assert_auto_derived_path(f2_output, dp, "train", "exact_dedup_filter")
        assert "user-bucket" not in f2_output

        # f2 reads from f1's output
        assert f2_input == "s3://user-bucket/first/"

        # f3: also auto-derived under data-prep bucket
        _assert_auto_derived_path(f3_output, dp, "train", "fuzzy_dedup")

        # f3 reads from f2's output
        assert f3_input == f"{dp}/f2-out/"
