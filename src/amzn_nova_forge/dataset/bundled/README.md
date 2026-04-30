# AGIDataCurator `.whl` Package

This document describes the filter operations and dependencies included in the `agi_data_curator-1.0.0-py3-none-any.whl` package.

## Data Preparation Filters

AGIDataCurator provides four filter operations for text data preparation: `default_text_filter`, `exact_dedup_filter`, `fuzzy_dedup_filter`, and `language_detection`.

### Default Text Filter (`default_text_filter`)

The `default_text_filter` operation is a composite filter that runs **7 individual text quality filters** in sequence. Each filter evaluates the input documents and removes those that do not meet configurable quality thresholds.

| # | Filter | Description | Defaults |
|---|--------|-------------|----------|
| 1 | **URL Ratio** | Removes documents where URLs comprise more than a configurable fraction of the total text. | Max ratio `0.2`, porn URL filtering `True` |
| 2 | **Alphanumeric Ratio** | Removes documents where the ratio of alphanumeric characters falls below a specified threshold. | Min ratio `0.25`, no upper bound |
| 3 | **Word Repetition** | Removes documents with excessive repeated word n-grams above a configurable threshold. | Max ratio `0.5`, 10-word n-grams |
| 4 | **Character Repetition** | Removes documents with excessive repeated character n-grams using a sqrt-based weighting scheme. | Max ratio `0.5`, 10-char n-grams |
| 5 | **Mojibake Detection** | Removes documents containing encoding corruption detected via ftfy badness scoring or regex patterns. | Max badness `1`, max mojibake ratio `0.05` |
| 6 | **Character Length** | Removes documents with character counts outside a specified min/max range. | Min `50` chars, no max |
| 7 | **Average Line Length** | Removes documents with abnormal average line lengths (e.g., word-per-line dumps or minified code). | Min avg `10` chars/line, no max, requires `2` lines |

### Exact Deduplication Filter (`exact_dedup_filter`)

| # | Filter | Description |
|---|--------|-------------|
| 1 | **Exact Deduplication** | Removes exact duplicate documents based on MD5 hash of text, keeping only the first occurrence. |

### Fuzzy Deduplication Filter (`fuzzy_dedup_filter`)

| # | Filter | Description | Defaults |
|---|--------|-------------|----------|
| 1 | **Fuzzy Deduplication** | Removes near-duplicate documents using MinHash LSH. Catches paraphrases, minor edits, and boilerplate variants that exact dedup misses. | Similarity threshold `0.8`, `256` permutations, `24`-char n-grams, `4` bands per iteration, seed `42`, lowercase `True` |

### Language Detection Filter (`language_detection`)

| # | Filter | Description | Defaults |
|---|--------|-------------|----------|
| 1 | **Language Detection** | Removes documents whose detected language is outside an ISO 639-1 allowlist or below a confidence threshold. Uses FastText's `lid.176` model (~126 MB, MIT-licensed). The output schema matches the input — no extra columns are written. | Allowlist `languages` is required, `min_score` default `0.0`, `keep_undetected` default `False` |

## Dependency License Audit


All dependencies use permissive open-source licenses that are compatible with distribution.

You can use the following command to list out all dependencies in the .whl:
```
unzip -p agi_data_curator-1.0.0-py3-none-any.whl '*.dist-info/METADATA' | grep "^Requires-Dist:"
```

| # | Package | Version Constraint | License | Category | Notes |
|---|---------|-------------------|---------|----------|-------|
| 1 | ray | >=2.9.0 | Apache-2.0 | Permissive | Distributed data processing framework |
| 2 | boto3 | >=1.26.0 | Apache-2.0 | Permissive | AWS SDK |
| 3 | s3fs | >=2024.6.0 | BSD-3-Clause | Permissive | S3 filesystem interface |
| 4 | pydantic | >=2.0.0 | MIT | Permissive | Data validation |
| 5 | pyyaml | >=6.0 | MIT | Permissive | YAML parser |
| 6 | tqdm | >=4.65.0 | MPL-2.0 AND MIT | Weak Copyleft (MPL) + Permissive (MIT) | Progress bars. MPL-2.0 is file-level copyleft only -- does not impose obligations on the larger work |
| 7 | psutil | >=5.9.0 | BSD-3-Clause | Permissive | System utilities |
| 8 | loguru | latest | MIT | Permissive | Logging |
| 9 | fsspec | latest | BSD-3-Clause | Permissive | Filesystem abstraction |
| 10 | requests | latest | Apache-2.0 | Permissive | HTTP client |
| 11 | Pillow | >=9.0.0 | HPND (MIT-like) | Permissive | Image I/O (used by some curator stages) |
| 12 | scipy | >=1.10.0 | BSD-3-Clause | Permissive | Scientific computing support for dedup math |
| 13 | fasttext-wheel | >=0.9.2 | MIT | Permissive | Language identification model runtime (lid.176) |
| 14 | pandas | >=1.3 (runtime extra) | BSD-3-Clause | Permissive | DataFrame library |
| 15 | pyarrow | >=9.0.0 (runtime extra) | Apache-2.0 | Permissive | Arrow/Parquet I/O used by Ray Data for reading and writing Parquet files |
| 16 | numpy | >=1.16.6 (runtime extra) | BSD-3-Clause | Permissive | Numerical computing |

**Summary:** 15 of 16 packages are fully permissive (Apache-2.0, MIT, BSD-3-Clause, or HPND). The one exception, tqdm, is dual-licensed MPL-2.0/MIT -- MPL-2.0 is weak copyleft that only requires modifications to MPL-licensed files themselves to be shared, with no obligations on the surrounding codebase.
