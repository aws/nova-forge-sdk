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
import hashlib
import json
import logging
import os
import secrets
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import pytest

from amzn_nova_forge.trainer.utils.batch_trace import (
    BatchTraceError,
    _cache_key,
    _collect_step_fingerprints,
    _extract_matched_lines,
    _fingerprint,
    _fingerprint_line,
    _load_or_build_index,
    _normalize_text,
    _SampleMatch,
    run,
)

logger = logging.getLogger(__name__)

_NUM_DATA_LINES = 30
_CUSTOMER_MATCHES_STEP_3 = 6
_HASHES_PER_STEP_PER_RANK = 8


class TestNormalizeText(unittest.TestCase):
    """Tests for _normalize_text."""

    def test_single_doc(self):
        self.assertEqual(_normalize_text("hello world"), "hello world")

    def test_multiple_docs(self):
        self.assertEqual(
            _normalize_text("part1 [DOC] part2 [DOC] part3"),
            "part1 [DOC] part2 [DOC] part3",
        )

    def test_inconsistent_doc_spacing(self):
        # [DOC] with no leading space
        self.assertEqual(
            _normalize_text("part1[DOC]part2"),
            "part1 [DOC] part2",
        )

    def test_leading_doc_separator(self):
        self.assertEqual(
            _normalize_text("[DOC] part1 [DOC] part2"),
            "part1 [DOC] part2",
        )

    def test_trailing_doc_separator(self):
        self.assertEqual(
            _normalize_text("part1 [DOC] part2 [DOC] "),
            "part1 [DOC] part2",
        )

    def test_empty_between_docs(self):
        self.assertEqual(
            _normalize_text("part1 [DOC] [DOC] part2"),
            "part1 [DOC] part2",
        )


class TestFingerprint(unittest.TestCase):
    """Tests for _fingerprint."""

    def test_deterministic(self):
        self.assertEqual(_fingerprint("hello"), _fingerprint("hello"))

    def test_different_inputs(self):
        self.assertNotEqual(_fingerprint("hello"), _fingerprint("world"))

    def test_length_is_16_hex_chars(self):
        result = _fingerprint("test content")
        self.assertEqual(len(result), 16)
        # All hex chars
        int(result, 16)

    def test_matches_sha256_prefix(self):
        content = "test content"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(_fingerprint(content), expected)


class TestFingerprintLine(unittest.TestCase):
    """Tests for _fingerprint_line."""

    def test_text_field(self):
        data = {"text": "hello world"}
        result = _fingerprint_line(data)
        self.assertIsNotNone(result)
        self.assertEqual(result, _fingerprint(_normalize_text("hello world")))

    def test_text_field_with_doc_markers(self):
        data = {"text": "part1 [DOC] part2"}
        result = _fingerprint_line(data)
        self.assertEqual(result, _fingerprint("part1 [DOC] part2"))

    def test_conversations_field(self):
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        data = {"conversations": conv}
        result = _fingerprint_line(data)
        expected = _fingerprint(json.dumps(conv, sort_keys=True, ensure_ascii=False))
        self.assertEqual(result, expected)

    def test_messages_field(self):
        msgs = [{"role": "user", "content": "hi"}]
        data = {"messages": msgs}
        result = _fingerprint_line(data)
        expected = _fingerprint(json.dumps(msgs, sort_keys=True, ensure_ascii=False))
        self.assertEqual(result, expected)

    def test_conversations_takes_priority_over_messages(self):
        conv = [{"role": "user", "content": "conv"}]
        msgs = [{"role": "user", "content": "msg"}]
        data = {"conversations": conv, "messages": msgs}
        result = _fingerprint_line(data)
        expected = _fingerprint(json.dumps(conv, sort_keys=True, ensure_ascii=False))
        self.assertEqual(result, expected)

    def test_empty_record_returns_none(self):
        self.assertIsNone(_fingerprint_line({}))

    def test_empty_text_falls_through_to_conversations(self):
        conv = [{"role": "user", "content": "hi"}]
        data = {"text": "", "conversations": conv}
        result = _fingerprint_line(data)
        expected = _fingerprint(json.dumps(conv, sort_keys=True, ensure_ascii=False))
        self.assertEqual(result, expected)


class TestCacheKey(unittest.TestCase):
    """Tests for _cache_key."""

    def test_deterministic(self):
        self.assertEqual(_cache_key("s3://bucket/data.jsonl"), _cache_key("s3://bucket/data.jsonl"))

    def test_different_paths_different_keys(self):
        self.assertNotEqual(
            _cache_key("s3://bucket/data1.jsonl"),
            _cache_key("s3://bucket/data2.jsonl"),
        )

    def test_format(self):
        result = _cache_key("s3://bucket/customer_data.jsonl")
        self.assertTrue(result.endswith(".index.csv"))
        self.assertIn("customer_data", result)


class TestLoadOrBuildIndex(unittest.TestCase):
    """Tests for _load_or_build_index."""

    def test_builds_index_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello world"}\n')
                f.write('{"text": "second line"}\n')
                f.write('{"text": "third line"}\n')

            index = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)

            self.assertEqual(len(index), 3)
            # Line numbers are 0-indexed
            fp0 = _fingerprint(_normalize_text("hello world"))
            fp1 = _fingerprint(_normalize_text("second line"))
            fp2 = _fingerprint(_normalize_text("third line"))
            self.assertEqual(index[fp0], 0)
            self.assertEqual(index[fp1], 1)
            self.assertEqual(index[fp2], 2)

    def test_loads_from_cache_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')

            index1 = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)
            index2 = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)

            self.assertEqual(index1, index2)

    def test_cache_invalidated_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')

            index1 = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)
            self.assertEqual(len(index1), 1)

            # Modify the file — cache should be invalidated
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')
                f.write('{"text": "world"}\n')

            index2 = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)
            self.assertEqual(len(index2), 2)

    def test_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')
                f.write("\n")
                f.write('{"text": "world"}\n')

            index = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)
            self.assertEqual(len(index), 2)

    def test_skips_records_without_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')
                f.write('{"other_field": "no text"}\n')

            index = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)
            self.assertEqual(len(index), 1)

    def test_duplicate_lines_first_occurrence_wins(self):
        """Duplicate fingerprints map to the first occurrence (lowest line number)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "duplicate line"}\n')  # line 0
                f.write('{"text": "unique line"}\n')  # line 1
                f.write('{"text": "duplicate line"}\n')  # line 2 (duplicate of line 0)

            index = _load_or_build_index(data_file, data_file, cache_dir=tmpdir)

            # Only 2 unique fingerprints despite 3 lines
            self.assertEqual(len(index), 2)

            # The duplicate fingerprint maps to line 0 (first occurrence wins)
            dup_fp = _fingerprint(_normalize_text("duplicate line"))
            self.assertEqual(index[dup_fp], 0)


class TestCollectStepFingerprints(unittest.TestCase):
    """Tests for _collect_step_fingerprints with local files."""

    def test_collects_hashes_for_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 0, "hashes": ["aaa", "bbb"]}) + "\n")
                f.write(json.dumps({"step": 1, "hashes": ["ccc", "ddd"]}) + "\n")

            results = _collect_step_fingerprints(tmpdir, step=1, cache_dir=tmpdir)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].fp, "ccc")
            self.assertEqual(results[0].dp_rank, 0)
            self.assertEqual(results[1].fp, "ddd")

    def test_handles_samples_schema(self):
        """Test the alternative schema: samples[].hash instead of hashes[]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(
                    json.dumps({"step": 0, "samples": [{"hash": "aaa"}, {"hash": "bbb"}]}) + "\n"
                )

            results = _collect_step_fingerprints(tmpdir, step=0, cache_dir=tmpdir)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].fp, "aaa")
            self.assertEqual(results[1].fp, "bbb")

    def test_multiple_dp_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for rank in range(3):
                log_file = os.path.join(tmpdir, f"batch_hashes_dp{rank}.jsonl")
                with open(log_file, "w") as f:
                    f.write(json.dumps({"step": 5, "hashes": [f"hash_r{rank}"]}) + "\n")

            results = _collect_step_fingerprints(tmpdir, step=5, cache_dir=tmpdir)

            self.assertEqual(len(results), 3)
            ranks = {r.dp_rank for r in results}
            self.assertEqual(ranks, {0, 1, 2})

    def test_raises_when_no_log_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(BatchTraceError) as ctx:
                _collect_step_fingerprints(tmpdir, step=0, cache_dir=tmpdir)
            self.assertIn("No training output log files", str(ctx.exception))

    def test_step_not_found_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 0, "hashes": ["aaa"]}) + "\n")

            results = _collect_step_fingerprints(tmpdir, step=999, cache_dir=tmpdir)
            self.assertEqual(len(results), 0)


class TestRun(unittest.TestCase):
    """End-to-end tests for run() with local files."""

    def test_end_to_end_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data file
            data_file = os.path.join(tmpdir, "data.jsonl")
            lines = [
                '{"text": "sample zero"}',
                '{"text": "sample one"}',
                '{"text": "sample two"}',
            ]
            with open(data_file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

            # Compute expected fingerprints
            fp0 = _fingerprint(_normalize_text("sample zero"))
            fp2 = _fingerprint(_normalize_text("sample two"))

            # Create log dir with batch hashes referencing lines 0 and 2
            log_dir = os.path.join(tmpdir, "batch_tracing")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 42, "hashes": [fp0, fp2]}) + "\n")

            output_file = os.path.join(tmpdir, "output.jsonl")
            result = run(
                data_path=data_file,
                log_dir=log_dir,
                step=42,
                output_path=output_file,
                cache_dir=tmpdir,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result, Path(output_file))
            self.assertTrue(os.path.exists(output_file))

            with open(output_file) as f:
                output_lines = f.readlines()
            self.assertEqual(len(output_lines), 2)
            # Lines should be the original data
            self.assertEqual(json.loads(output_lines[0])["text"], "sample zero")
            self.assertEqual(json.loads(output_lines[1])["text"], "sample two")

    def test_no_matches_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')

            log_dir = os.path.join(tmpdir, "batch_tracing")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 1, "hashes": ["nonexistent_hash"]}) + "\n")

            result = run(
                data_path=data_file,
                log_dir=log_dir,
                step=1,
                cache_dir=tmpdir,
            )
            self.assertIsNone(result)

    def test_step_not_found_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')

            log_dir = os.path.join(tmpdir, "batch_tracing")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 1, "hashes": ["aaa"]}) + "\n")

            result = run(
                data_path=data_file,
                log_dir=log_dir,
                step=999,
                cache_dir=tmpdir,
            )
            self.assertIsNone(result)

    def test_missing_data_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "batch_tracing")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 0, "hashes": ["aaa"]}) + "\n")

            with self.assertRaises(BatchTraceError):
                run(
                    data_path="/nonexistent/data.jsonl",
                    log_dir=log_dir,
                    step=0,
                    cache_dir=tmpdir,
                )

    def test_missing_log_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "hello"}\n')

            with self.assertRaises(BatchTraceError):
                run(
                    data_path=data_file,
                    log_dir=os.path.join(tmpdir, "nonexistent"),
                    step=0,
                    cache_dir=tmpdir,
                )


def _random_hex_hash() -> str:
    """Generate a random 16-char hex string (simulates non-customer data hashes)."""
    return secrets.token_hex(8)


@pytest.fixture(scope="module")
def synthetic_test_data(tmp_path_factory):
    """Generate synthetic CPT data and hash log files.

    Creates:
      - cpt_test_data.jsonl: 30 short synthetic records
      - batch_hashes_dp0.jsonl: 5 steps, 8 hashes each
      - batch_hashes_dp1.jsonl: 5 steps, 8 hashes each

    Step 3 is engineered to have exactly _CUSTOMER_MATCHES_STEP_3 matches
    split across ranks 0 and 1. Step 5 has a different set of matches.
    """
    tmp = tmp_path_factory.mktemp("batch_trace")

    records = [
        {"text": f"Synthetic CPT sample line {i} for batch trace testing."}
        for i in range(_NUM_DATA_LINES)
    ]

    data_path = tmp / "cpt_test_data.jsonl"
    with open(data_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    fingerprints = []
    for record in records:
        fp = _fingerprint_line(record)
        assert fp is not None
        fingerprints.append(fp)

    step3_customer_fps = fingerprints[5:11]
    assert len(step3_customer_fps) == _CUSTOMER_MATCHES_STEP_3

    step5_customer_fps = fingerprints[20:24]

    steps = [1, 2, 3, 4, 5]
    for rank in [0, 1]:
        log_path = tmp / f"batch_hashes_dp{rank}.jsonl"
        with open(log_path, "w") as f:
            for step in steps:
                hashes = []
                if step == 3:
                    if rank == 0:
                        hashes.extend(step3_customer_fps[:3])
                    else:
                        hashes.extend(step3_customer_fps[3:])
                    while len(hashes) < _HASHES_PER_STEP_PER_RANK:
                        hashes.append(_random_hex_hash())
                elif step == 5:
                    if rank == 0:
                        hashes.extend(step5_customer_fps[:2])
                    else:
                        hashes.extend(step5_customer_fps[2:])
                    while len(hashes) < _HASHES_PER_STEP_PER_RANK:
                        hashes.append(_random_hex_hash())
                else:
                    hashes = [_random_hex_hash() for _ in range(_HASHES_PER_STEP_PER_RANK)]

                f.write(json.dumps({"step": step, "hashes": hashes}) + "\n")

    return {
        "data_path": str(data_path),
        "log_dir": str(tmp) + "/",
        "tmp_dir": tmp,
        "step3_expected_matches": _CUSTOMER_MATCHES_STEP_3,
        "step5_expected_matches": 4,
    }


class TestBatchTraceEndToEnd:
    """Validate batch_trace.run() end-to-end against synthetically generated data."""

    def test_trace_step_3(self, synthetic_test_data):
        """batch_trace.run() matches expected customer samples at step 3."""
        td = synthetic_test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = str(Path(tmpdir) / "step_3_samples.jsonl")
            cache_dir = str(Path(tmpdir) / "cache")

            result = run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=3,
                output_path=output_file,
                cache_dir=cache_dir,
            )

            assert result is not None, (
                f"trace returned None for step 3 — expected {td['step3_expected_matches']} matches"
            )
            assert result.exists(), f"Output file not created: {result}"
            assert result.stat().st_size > 0, f"Output file is empty: {result}"

            with open(result) as f:
                output_lines = f.readlines()

            logger.info("Step 3: matched %d customer samples", len(output_lines))

            assert len(output_lines) == td["step3_expected_matches"], (
                f"Expected {td['step3_expected_matches']} matches, got {len(output_lines)}"
            )

            for i, line in enumerate(output_lines):
                record = json.loads(line.strip())
                assert "text" in record, f"Output line {i} missing 'text' field: {record}"

    def test_trace_different_step(self, synthetic_test_data):
        """batch_trace.run() produces results for a different step."""
        td = synthetic_test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = str(Path(tmpdir) / "step_5_samples.jsonl")
            cache_dir = str(Path(tmpdir) / "cache")

            result = run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=5,
                output_path=output_file,
                cache_dir=cache_dir,
            )

            assert result is not None, "Expected matches at step 5"
            with open(result) as f:
                lines = f.readlines()
            logger.info("Step 5: matched %d customer samples", len(lines))
            assert len(lines) == td["step5_expected_matches"]
            for line in lines:
                record = json.loads(line.strip())
                assert "text" in record

    def test_nonexistent_step_returns_none(self, synthetic_test_data):
        """batch_trace.run() returns None for a step that doesn't exist."""
        td = synthetic_test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=99999,
                output_path=str(Path(tmpdir) / "step_99999.jsonl"),
                cache_dir=str(Path(tmpdir) / "cache"),
            )

            assert result is None, f"Expected None for non-existent step 99999, got {result}"

    def test_step_with_no_customer_matches_returns_none(self, synthetic_test_data):
        """batch_trace.run() returns None when step has only non-customer hashes."""
        td = synthetic_test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=1,
                output_path=str(Path(tmpdir) / "step_1.jsonl"),
                cache_dir=str(Path(tmpdir) / "cache"),
            )

            assert result is None, f"Expected None for noise-only step 1, got {result}"

    def test_index_caching(self, synthetic_test_data):
        """Second run reuses cached fingerprint index (faster execution)."""
        td = synthetic_test_data
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = str(Path(tmpdir) / "cache")

            # First run — builds the index
            run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=3,
                output_path=str(Path(tmpdir) / "run1.jsonl"),
                cache_dir=cache_dir,
            )

            cache_path = Path(cache_dir)
            assert cache_path.exists(), "Cache directory not created"
            cache_files = list(cache_path.rglob("*.index.csv"))
            assert len(cache_files) > 0, "No index cache files found"
            logger.info("Cache files: %s", [f.name for f in cache_files])

            # Second run — should reuse cached index
            result = run(
                data_path=td["data_path"],
                log_dir=td["log_dir"],
                step=3,
                output_path=str(Path(tmpdir) / "run2.jsonl"),
                cache_dir=cache_dir,
            )

            assert result is not None
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == td["step3_expected_matches"]


class TestCollectStepFingerprintsS3(unittest.TestCase):
    """Tests for _collect_step_fingerprints with S3 paths."""

    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.read_lines")
    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.list_s3_prefix")
    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.is_s3", return_value=True)
    def test_s3_path_collects_fingerprints(self, mock_is_s3, mock_list, mock_read):
        mock_list.return_value = ["s3://bucket/batch_tracing/batch_hashes_dp0.jsonl"]
        mock_read.return_value = [json.dumps({"step": 1, "hashes": ["abc123"]}) + "\n"]
        mock_client = unittest.mock.MagicMock()
        results = _collect_step_fingerprints(
            "s3://bucket/batch_tracing/", step=1, s3_client=mock_client
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fp, "abc123")

    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.is_s3", return_value=True)
    def test_s3_path_without_client_raises(self, mock_is_s3):
        with self.assertRaises(BatchTraceError) as ctx:
            _collect_step_fingerprints("s3://bucket/logs/", step=1, s3_client=None)
        self.assertIn("S3 client required", str(ctx.exception))

    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.list_s3_prefix")
    @unittest.mock.patch("amzn_nova_forge.trainer.utils.batch_trace.is_s3", return_value=True)
    def test_s3_list_failure_raises_batch_trace_error(self, mock_is_s3, mock_list):
        mock_list.side_effect = Exception("Access denied")
        mock_client = unittest.mock.MagicMock()
        with self.assertRaises(BatchTraceError) as ctx:
            _collect_step_fingerprints("s3://bucket/logs/", step=1, s3_client=mock_client)
        self.assertIn("Failed to list training output", str(ctx.exception))


class TestFingerprintRegression(unittest.TestCase):
    """Pinned-value tests to catch algorithm drift."""

    def test_known_text_hash(self):
        # Pinned: changing this breaks compatibility with existing batch hash logs
        expected = hashlib.sha256("hello world".encode("utf-8")).hexdigest()[:16]
        self.assertEqual(_fingerprint("hello world"), expected)
        self.assertEqual(_fingerprint("hello world"), "b94d27b9934d3e08")

    def test_known_line_hash(self):
        record = {"text": "hello world"}
        result = _fingerprint_line(record)
        # _normalize_text("hello world") == "hello world" (no [DOC] markers)
        self.assertEqual(result, "b94d27b9934d3e08")


class TestExtractMatchedLines(unittest.TestCase):
    """Direct tests for _extract_matched_lines."""

    def test_extracts_correct_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "line zero"}\n')
                f.write('{"text": "line one"}\n')
                f.write('{"text": "line two"}\n')
                f.write('{"text": "line three"}\n')
            matches = [
                _SampleMatch(fp="a", dp_rank=0, line_number=1),
                _SampleMatch(fp="b", dp_rank=0, line_number=3),
            ]
            output = os.path.join(tmpdir, "output.jsonl")
            result = _extract_matched_lines(data_file, matches, output)
            self.assertEqual(result, Path(output))
            with open(output) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("line one", lines[0])
            self.assertIn("line three", lines[1])

    def test_empty_matches_produces_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write('{"text": "line zero"}\n')
            output = os.path.join(tmpdir, "output.jsonl")
            result = _extract_matched_lines(data_file, [], output)
            self.assertEqual(result, Path(output))
            with open(output) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 0)

    def test_output_sorted_by_line_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                for i in range(5):
                    f.write(json.dumps({"text": f"line {i}"}) + "\n")
            # Matches in reverse order
            matches = [
                _SampleMatch(fp="b", dp_rank=0, line_number=3),
                _SampleMatch(fp="a", dp_rank=0, line_number=1),
            ]
            output = os.path.join(tmpdir, "output.jsonl")
            _extract_matched_lines(data_file, matches, output)
            with open(output) as f:
                lines = f.readlines()
            # Should be sorted: line 1 before line 3
            self.assertIn("line 1", lines[0])
            self.assertIn("line 3", lines[1])


class TestCollectStepFingerprintsMalformedJSON(unittest.TestCase):
    """Tests for graceful handling of malformed JSON in log files."""

    def test_skips_malformed_json_in_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "batch_tracing")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "batch_hashes_dp0.jsonl")
            with open(log_file, "w") as f:
                f.write(json.dumps({"step": 1, "hashes": ["aaa"]}) + "\n")
                f.write("NOT_VALID_JSON\n")
                f.write(json.dumps({"step": 2, "hashes": ["bbb"]}) + "\n")
            results = _collect_step_fingerprints(log_dir, step=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].fp, "aaa")


class TestLoadOrBuildIndexMalformedJSON(unittest.TestCase):
    """Tests for graceful handling of malformed JSON in data files."""

    def test_skips_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "data.jsonl")
            with open(data_file, "w") as f:
                f.write(json.dumps({"text": "valid line one"}) + "\n")
                f.write("{BROKEN_JSON\n")
                f.write(json.dumps({"text": "valid line three"}) + "\n")
            cache_dir = os.path.join(tmpdir, "cache")
            index = _load_or_build_index(data_file, data_file, cache_dir)
            self.assertEqual(len(index), 2)
