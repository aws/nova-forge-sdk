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
"""Batch sample tracing tool for gradient spike diagnosis.

Identifies which lines from a customer's input data file were used in a
specific training step's batch.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

import boto3

from amzn_nova_forge.core.constants import DEFAULT_BATCH_TRACE_CACHE_DIR
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.s3_utils import (
    ensure_local,
    is_s3,
    list_s3_prefix,
    read_lines,
)

_DOC_REGEX = re.compile(r" ?\[DOC] ?")
_SECONDS_PER_MINUTE = 60


class BatchTraceError(Exception):
    """Raised when batch tracing encounters an unrecoverable error."""


class _SampleMatch(NamedTuple):
    """A fingerprint from a training step matched to a source data line."""

    fp: str
    dp_rank: int
    line_number: int


class _StepSample(NamedTuple):
    """A fingerprint collected from a training step's batch hash log."""

    fp: str
    dp_rank: int


def _elapsed(t0: float) -> str:
    """Format elapsed time since *t0* as a human-readable string."""
    secs = time.time() - t0
    return f"{secs:.1f}s" if secs < _SECONDS_PER_MINUTE else f"{secs / _SECONDS_PER_MINUTE:.1f}m"


def _normalize_text(text: str) -> str:
    """Canonicalize [DOC] separators for stable fingerprinting.

    The training container uses [DOC] as a document boundary marker in CPT
    data. This normalization ensures consistent spacing so fingerprints
    match regardless of whitespace variations around the marker.
    Must stay in sync with the container-side ``BatchHashLogger``.
    """
    split_docs = re.split(_DOC_REGEX, text)
    split_docs = [el for el in split_docs if el.strip()]
    return " [DOC] ".join(split_docs)


def _fingerprint(content: str) -> str:
    """Return a 16-char hex SHA-256 prefix of *content*.

    The truncation length (16 hex chars = 64 bits) must match the
    training container's batch hash log format.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _fingerprint_line(data: dict) -> str | None:
    """Compute a fingerprint for a single JSONL record.

    Priority: ``text`` field (normalized) > ``conversations``/``messages``
    (deterministic JSON). Returns ``None`` if the record has neither field.
    Must match the training container's ``BatchHashLogger`` hashing logic.
    """
    text = data.get("text", "")
    if text:
        return _fingerprint(_normalize_text(text))
    conv = data.get("conversations") or data.get("messages")
    if conv:
        return _fingerprint(json.dumps(conv, sort_keys=True, ensure_ascii=False))
    return None


def _cache_key(path: str, local_path: str | None = None) -> str:
    key_input = path
    if local_path and os.path.exists(local_path):
        stat = os.stat(local_path)
        key_input += f":{stat.st_mtime_ns}:{stat.st_size}"
    h = hashlib.sha256(key_input.encode()).hexdigest()[:12]
    base = os.path.basename(path.rstrip("/")).split(".")[0]
    return f"{base}_{h}.index.csv"


def _load_or_build_index(
    data_path: str,
    local_path: str,
    cache_dir: str,
) -> dict[str, int]:
    """Build or load a cached fingerprint→line_number index.

    Args:
        data_path: Original path (local or S3) used as the cache key.
        local_path: Already-resolved local file path to index.
        cache_dir: Directory for persisting the index CSV.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _cache_key(data_path, local_path))

    if os.path.exists(cache_path):
        try:
            logger.info("Using cached data index: %s", cache_path)
            mapping: dict[str, int] = {}
            with open(cache_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    mapping[row[0]] = int(row[1])
            logger.info("  %d entries loaded", len(mapping))
            return mapping
        except (StopIteration, IndexError, ValueError, csv.Error) as e:
            logger.warning("Corrupt cache file %s (%s), rebuilding", cache_path, e)
            os.remove(cache_path)

    logger.info("Building data index from %s (one-time, result is cached)...", data_path)

    mapping = {}
    duplicates = 0

    with open(local_path) as f:
        for line_num, raw_line in enumerate(f):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                continue
            fp = _fingerprint_line(data)
            if fp:
                if fp not in mapping:
                    mapping[fp] = line_num
                else:
                    duplicates += 1

    if duplicates:
        logger.warning(
            "%d duplicate lines found — only the first occurrence of each is indexed",
            duplicates,
        )

    tmp_cache = cache_path + ".tmp"
    with open(tmp_cache, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fp", "line"])
        for fp, ln in mapping.items():
            writer.writerow([fp, ln])
    os.replace(tmp_cache, cache_path)

    logger.info("  %d lines indexed, cached to %s", len(mapping), cache_path)
    return mapping


def _collect_step_fingerprints(
    log_dir: str,
    step: int,
    s3_client: Any | None = None,
    cache_dir: str = DEFAULT_BATCH_TRACE_CACHE_DIR,
) -> list[_StepSample]:
    """Collect fingerprints for a specific step from batch hash log files."""
    if is_s3(log_dir):
        if s3_client is None:
            raise BatchTraceError(f"S3 client required to list {log_dir}")
        try:
            log_files = list_s3_prefix(log_dir, s3_client, suffix=".jsonl")
        except Exception as e:
            raise BatchTraceError(f"Failed to list training output: {e}") from e
        log_files = [f for f in log_files if "batch_hashes_dp" in f]
    else:
        log_files = sorted(
            glob.glob(
                os.path.join(glob.escape(log_dir), "**/batch_hashes_dp*.jsonl"), recursive=True
            )
        )

    if not log_files:
        msg = f"No training output log files found in {log_dir}"
        raise BatchTraceError(msg)

    results: list[_StepSample] = []
    ranks_found = 0
    for log_file in log_files:
        try:
            lines = read_lines(log_file, s3_client, cache_dir)
        except Exception as e:
            raise BatchTraceError(f"Failed to read log file {log_file}: {e}") from e

        filename = log_file.rsplit("/", 1)[-1]
        rank_match = re.search(r"batch_hashes_dp(\d+)", filename)
        if not rank_match:
            logger.warning("Could not parse dp_rank from filename: %s", filename)
            continue
        dp_rank = int(rank_match.group(1))
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON in %s", log_file)
                continue
            if record.get("step") == step:
                hashes = record.get("hashes")
                if hashes is None:
                    hashes = [s.get("hash") for s in record.get("samples", [])]
                results.extend(_StepSample(fp=h, dp_rank=dp_rank) for h in hashes if h)
                ranks_found += 1
                break

    logger.info(
        "Found %d training output files, %d had data for step %d",
        len(log_files),
        ranks_found,
        step,
    )
    logger.info("  %d samples in step %d", len(results), step)
    return results


def _extract_matched_lines(
    local_data_path: str,
    matches: list[_SampleMatch],
    output_path: str,
) -> Path:
    """Stream the data file and write matched lines to *output_path*."""
    line_numbers = {m.line_number for m in matches}
    extracted: dict[int, str] = {}

    with open(local_data_path) as infile:
        for ln, line in enumerate(infile):
            if ln in line_numbers:
                extracted[ln] = line
                if len(extracted) == len(line_numbers):
                    break

    with open(output_path, "w") as outfile:
        for ln in sorted(line_numbers):
            if ln in extracted:
                line = extracted[ln]
                outfile.write(line if line.endswith("\n") else line + "\n")

    logger.info("Extracted %d lines to %s", len(extracted), output_path)
    for m in sorted(matches, key=lambda x: x.line_number):
        logger.info("  line %d  (rank %d)", m.line_number, m.dp_rank)

    return Path(output_path)


def run(
    data_path: str,
    log_dir: str,
    step: int,
    output_path: str | None = None,
    s3_client: Any | None = None,
    cache_dir: str = DEFAULT_BATCH_TRACE_CACHE_DIR,
) -> Path | None:
    """Trace which input data lines were used in a specific training step.

    Args:
        data_path: Path to input data file (local or ``s3://``).
        log_dir: Directory containing training batch hash logs (local or ``s3://``).
        step: Training step number to investigate.
        output_path: Output file for matched lines (default: ``step_<N>_samples.jsonl``).
        s3_client: Optional pre-configured boto3 S3 client. Created automatically
            if not provided and S3 paths are used.
        cache_dir: Directory for caching downloaded files and fingerprint indices.
            Defaults to ``~/.nova-forge/batch_trace_cache``.

    Returns:
        Path to the output file containing matched lines, or ``None`` if no
        matches were found.

    Raises:
        BatchTraceError: If batch tracing encounters an unrecoverable error
            (e.g., missing files, missing log files, AWS auth failure).
    """
    cache_dir = str(Path(cache_dir).expanduser())

    needs_s3 = is_s3(data_path) or is_s3(log_dir)
    if needs_s3 and s3_client is None:
        try:
            s3_client = boto3.client("s3")
        except Exception as e:
            raise BatchTraceError(f"AWS authentication failed: {e}") from e

    # Step 1: Collect fingerprints from training output
    t0 = time.time()
    source_label = "local file" if not is_s3(log_dir) else "S3"
    logger.info("[1/4] Loading training output (%s): %s", source_label, log_dir)
    step_fps = _collect_step_fingerprints(log_dir, step, s3_client, cache_dir)
    logger.info("      Done (%s) — %d samples in step %d", _elapsed(t0), len(step_fps), step)
    if not step_fps:
        logger.info("No samples found for step %d", step)
        return None

    # Step 2: Ensure dataset is available locally
    t0 = time.time()
    data_source_label = "local file" if not is_s3(data_path) else "S3"
    logger.info("[2/4] Loading dataset (%s): %s", data_source_label, data_path)
    try:
        local_data_path = ensure_local(data_path, s3_client, cache_dir)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise BatchTraceError(f"Failed to access data file: {e}") from e
    except Exception as e:
        raise BatchTraceError(f"Failed to download data file: {e}") from e
    logger.info("      Done (%s) — %s", _elapsed(t0), local_data_path)

    # Step 3: Build/load fingerprint index
    t0 = time.time()
    logger.info("[3/4] Building/loading hash index...")
    index = _load_or_build_index(data_path, local_data_path, cache_dir)
    logger.info("      Done (%s) — %d entries", _elapsed(t0), len(index))

    # Step 4: Match batch fingerprints against index
    t0 = time.time()
    logger.info("[4/4] Matching batch samples against dataset index...")
    matched: list[_SampleMatch] = []
    unmatched: list[_StepSample] = []
    for sample in step_fps:
        if sample.fp in index:
            matched.append(
                _SampleMatch(
                    fp=sample.fp,
                    dp_rank=sample.dp_rank,
                    line_number=index[sample.fp],
                )
            )
        else:
            unmatched.append(sample)
    logger.info(
        "      Done (%s) — %d matched, %d unmatched", _elapsed(t0), len(matched), len(unmatched)
    )

    logger.info("Step %d batch:", step)
    logger.info("  %d samples from your data", len(matched))
    logger.info("  %d samples from other sources", len(unmatched))

    if not matched:
        logger.info("No lines from your data file matched this step.")
        return None

    # Extract and write matched lines
    dest = output_path or f"step_{step}_samples.jsonl"
    return _extract_matched_lines(local_data_path, matched, dest)
