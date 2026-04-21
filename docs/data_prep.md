# Data Preparation Guide

This guide walks through the data preparation workflow using the Nova Forge SDK's dataset loaders.

> **Note:** If you are upgrading from an earlier version of the SDK, some method names and parameters
> have changed. The old names still work but will emit deprecation warnings. See
> [Migrating from Earlier Versions](#migrating-from-earlier-versions) at the end of this guide.

## Operations at a Glance

| Theme | Operator | Purpose | Supported Runtimes |
|---|---|---|---|
| Load | `loader.load(path)` | Ingest data from JSONL, JSON, CSV, Parquet, or Arrow (local or S3) | Local |
| Filter | `loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER)` | Remove records with excessive URL content (web-scraped boilerplate) | AWS Glue |
| Filter | `loader.filter(method=FilterMethod.EXACT_DEDUP)` | Remove exact duplicate records by content hash | AWS Glue |
| Filter | `loader.filter(method=FilterMethod.FUZZY_DEDUP)` | Remove near-duplicates via MinHash LSH similarity | AWS Glue |
| Filter | `loader.filter(method=FilterMethod.INVALID_RECORDS)` | Remove records failing schema/format checks for the target training method | Local |
| Transform | `loader.transform(method=TransformMethod.SCHEMA)` | Convert raw data into the schema required by a training method and model | Local |
| Validate | `loader.validate(method=ValidateMethod.INVALID_RECORDS)` | Check dataset conforms to training requirements (raises on failure) | Local |
| Split | `loader.split(...)` | Divide dataset into train/val/test sets | Local |
| Save | `loader.save(path)` | Write dataset to JSONL or JSON (local or S3) | Local |
| Show | `loader.show(n=10)` | Preview current dataset state | Local |

---

## Overview

The SDK provides dataset loaders for different file formats (`JSONLDatasetLoader`, `JSONDatasetLoader`, `CSVDatasetLoader`, `ParquetDatasetLoader`, `ArrowDatasetLoader`), all sharing the same chainable API. Operations are lazy — they queue up and execute when a terminal operation is called.

```python
from amzn_nova_forge import (
    JSONLDatasetLoader,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.dataset.operations import FilterMethod, TransformMethod, ValidateMethod
```

A typical end-to-end pipeline:

```python
loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/raw_data.jsonl")

# Clean and deduplicate raw text
loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER, text_field="text")
loader.filter(method=FilterMethod.EXACT_DEDUP, text_field="text")

# Convert to training schema
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
)

# Validate and save
loader.validate(method=ValidateMethod.INVALID_RECORDS, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.save("s3://my-bucket/prepared_data.jsonl")
```

Each method returns the loader, so chaining (`loader.load(...).filter(...).transform(...).save(...)`) also works. The sections below cover each operation in detail.

---

## Configuration

### Runtime Managers

Filter operations that run on AWS Glue require a runtime manager. The SDK provides `GlueRuntimeManager` for this purpose — it handles job submission, polling, and artifact staging.

```python
from amzn_nova_forge.manager import GlueRuntimeManager

manager = GlueRuntimeManager(
    glue_role_name="GlueDataPrepExecutionRole",
    num_workers=4,
    worker_type="Z.2X",
    region="us-east-1",
)
```

| Parameter | Default | Description |
|---|---|---|
| `glue_role_name` | `GlueDataPrepExecutionRole` | IAM role the Glue job assumes |
| `s3_artifact_bucket` | Auto-created | Bucket for Glue scripts/wheels. Auto-creates `sagemaker-forge-dataprep-{account_id}-{region}` when `None`. |
| `s3_artifact_prefix` | `nova-forge/glue-artifacts` | S3 key prefix for artifacts |
| `worker_type` | `Z.2X` | Glue worker type (only `Z.2X` supported for Ray) |
| `num_workers` | `2` | Number of Glue workers |
| `region` | Session region | AWS region |
| `poll_interval` | `30` | Seconds between status polls |
| `max_wait_time` | `3600` | Maximum seconds to wait for job completion |
| `kms_key_id` | `None` | Optional KMS key for S3 encryption |

If you don't pass a `runtime_manager` to `filter()`, the SDK creates one from any Glue-specific kwargs on that call (e.g. `num_workers=8`). You can also share a single manager across multiple filter steps.

### IAM Permissions

Two sets of permissions are involved:

1. **Your local credentials** (the caller invoking the SDK) need permission to:
   - Start and monitor Glue jobs (`glue:StartJobRun`, `glue:GetJobRun`, `glue:CreateJob`)
   - Create the artifact S3 bucket if it doesn't exist (`s3:CreateBucket`)
   - Upload scripts/wheels to the artifact bucket (`s3:PutObject`)
   - Download filtered results back to local (`s3:GetObject`, `s3:ListBucket`)

2. **The Glue execution role** (`glue_role_name`) needs the policies defined in [`src/amzn_nova_forge/iam/glue_policies.json`](../src/amzn_nova_forge/iam/glue_policies.json):
   - `trust_policy` — allows `glue.amazonaws.com` to assume the role
   - `glue_base_policy` — Glue job execution (`GetJob`, `GetJobRun`, `StartJobRun`, `BatchStopJobRun`)
   - `glue_s3_policy` — read/write data (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`)
   - `glue_logs_policy` — CloudWatch Logs for job output

### S3 Buckets and Output Path Resolution

When running filter operations, the SDK needs S3 paths for intermediate and final outputs:

- **Artifact bucket** — If `s3_artifact_bucket` is not provided, the SDK auto-creates `sagemaker-forge-dataprep-{account_id}-{region}` to store Glue scripts and wheels. No KMS encryption is applied to this default bucket.
- **Output paths** — If `output_path` is not provided on a filter step, the SDK auto-derives it from the load path using the pattern `<parent>/<input_stem>/<session_id>/<method>/`, where `session_id` is a UTC timestamp. This keeps outputs grouped in a predictable session tree and prevents overwrites across repeated runs. For example, loading from `s3://my-bucket/data/corpus.jsonl` and running `DEFAULT_TEXT_FILTER` produces output at `s3://my-bucket/data/corpus/2026-04-20_14-30-00/default_text_filter/`.
- **Input detection** — When filters are chained, each step automatically uses the previous step's output as its input. Local files are uploaded to S3 automatically before the first Glue filter runs.

---

## Loading Data

`loader.load(path)` — Ingests data from a local file, S3 path, or directory.

- **Input:** File path (local or `s3://`) or directory path
- **Output:** Lazy record stream stored on the loader

```python
loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/data/training_data.jsonl")
```

### Supported Loaders

| Loader | Extensions | Import |
|---|---|---|
| `JSONLDatasetLoader` | `.jsonl` | `from amzn_nova_forge import JSONLDatasetLoader` |
| `JSONDatasetLoader` | `.json` | `from amzn_nova_forge import JSONDatasetLoader` |
| `CSVDatasetLoader` | `.csv` | `from amzn_nova_forge import CSVDatasetLoader` |
| `ParquetDatasetLoader` | `.parquet`, `.pq` | `from amzn_nova_forge import ParquetDatasetLoader` |
| `ArrowDatasetLoader` | `.arrow`, `.feather`, `.ipc` | `from amzn_nova_forge import ArrowDatasetLoader` |

### Path Resolution

All loaders accept relative paths, tilde paths, and `..` segments — resolved to absolute paths automatically:

```python
loader.load("./data/train.jsonl")       # relative to cwd
loader.load("~/datasets/train.jsonl")   # expands ~ to home directory
loader.load("../shared/train.jsonl")    # resolves parent references
```

### Directory Loading

Pass a directory path to load all matching files as a single stream:

```python
loader = JSONLDatasetLoader()
loader.load("/data/training/")          # all .jsonl files

loader = ParquetDatasetLoader()
loader.load("s3://my-bucket/datasets/") # all .parquet files from S3 prefix
```

- Files loaded in lexicographic (sorted) order for deterministic output
- Files starting with `.` or `_` are ignored
- Raises `DataPrepError` if no matching files found or if mismatched extensions exist
- Records yielded lazily, one file at a time

---

## Inspecting Data

`loader.show(n=10)` — Previews the current dataset state. Flushes any pending operations first.

- **Input:** Loaded (and optionally transformed/filtered) dataset
- **Output:** Prints `n` rows to stdout

```python
loader.load("data.jsonl")
loader.show()        # first 10 rows
loader.show(n=3)     # first 3 rows
```

---

## Transforming Data

`loader.transform(method=TransformMethod.*, ...)` — Applies a transformation to the dataset.

- **Input:** Raw dataset with arbitrary column names
- **Output:** Dataset in the target training format

```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
)
```

---

### `TransformMethod.SCHEMA`

Converts raw data into the schema required by a training method and model. If the data is already in the correct format, it's a no-op.

- **Input:** Raw dataset with arbitrary column names
- **Output:** Dataset in the target training format (e.g. Converse schema for SFT)
- **Runs on:** Local

```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `training_method` | `TrainingMethod` | Yes | Target training method (e.g. `SFT_LORA`, `RFT_LORA`, `CPT`). |
| `model` | `Model` | Yes | Target Nova model (e.g. `Model.NOVA_LITE_2`). |
| `column_mappings` | `dict` | No | Maps standard field names to your column names. Format: `{"standard_name": "your_column"}`. |
| `multimodal_data_s3_path` | `str` | No | S3 prefix for image/video assets (required for multimodal OpenAI format). |

**Column mappings by training method:**

| Training Method | Required Mappings | Optional Mappings |
|---|---|---|
| SFT | `question`, `answer` | `system`, `image_format`/`video_format`, `s3_uri`, `bucket_owner` |
| RFT | `question`, `reference_answer` | `system`, `id`, `tools` |
| Evaluation | `query`, `response` | `images`, `metadata` |
| CPT | `text` | — |

**Supported input formats:**

- **Generic Q/A** — flat fields like `question`/`answer`. Requires `column_mappings`.
- **OpenAI Messages** — `messages` array with `role`/`content`. No mappings needed.
- **OpenAI Multimodal** — `messages` with `image_url` content blocks. Pass `multimodal_data_s3_path`.
- **Already in Converse format** — transform is a no-op.

**Examples by training method:**

SFT:
```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
)
```

RFT:
```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.RFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "prompt", "reference_answer": "answer"},
)
```

CPT:
```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.CPT,
    model=Model.NOVA_LITE_2,
    column_mappings={"text": "content"},
)
```

Evaluation:
```python
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.EVALUATION,
    model=Model.NOVA_LITE_2,
    column_mappings={"query": "question", "response": "answer"},
)
```

---

## Validating Data

`loader.validate(method=ValidateMethod.*, ...)` — Checks that the dataset conforms to training requirements. Raises on failure.

- **Input:** Transformed dataset in the target training schema
- **Output:** Raises `DataPrepError` if validation fails; no-op if valid

```python
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
```

---

### `ValidateMethod.INVALID_RECORDS`

Validates dataset structure against the requirements for a training method and model. (`ValidateMethod.SCHEMA` is a deprecated alias that still works but logs a warning.)

- **Input:** Transformed dataset in the target training schema
- **Output:** Raises `DataPrepError` on failure
- **Runs on:** Local

```python
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `training_method` | `TrainingMethod` | Yes | Target training method. |
| `model` | `Model` | Yes | Target Nova model. |
| `eval_task` | `EvaluationTask` | No | Required for evaluation datasets (e.g. `EvaluationTask.GEN_QA`). |

**Checks performed:**

- Required fields present
- Field types and formats correct
- Role alternation in conversations (user/assistant)
- Optional field consistency across samples
- Forbidden keywords in content
- Image/video file size limits
- Image/video count limits per message
- Video duration limit (requires `pymediainfo`)

> **Note:** Row-count checks (e.g. Bedrock min/max sample bounds, dataset size ≥ `global_batch_size`) are enforced only during `train()`, not on the `loader.validate()` path.

**Video duration validation** requires the optional `pymediainfo` dependency as well as the `libmediainfo` package:

1. Install `pymediainfo` from PyPi

```bash
pip install amzn-nova-forge[video]
```

2. Install `libmediainfo` for your OS:

| OS | Command |
|---|---|
| macOS | `brew install libmediainfo` |
| Ubuntu/Debian | `sudo apt-get install libmediainfo-dev` |
| Amazon Linux 2 / RHEL | `sudo yum install libmediainfo` |
| Amazon Linux 2023 | `sudo dnf install libmediainfo` |
| Windows | Bundled with `pymediainfo` — no extra step |

**Evaluation datasets:**

```python
from amzn_nova_forge import EvaluationTask

loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.EVALUATION,
    model=Model.NOVA_LITE_2,
    eval_task=EvaluationTask.GEN_QA,
)
```

---

## Filtering Data

`loader.filter(method=FilterMethod.*, ...)` — Queues a filtering operation. All filters are lazy — execution happens on a terminal operation (`execute()`, `save()`, `show()`, `split()`, `validate()`).

- **Input:** Depends on filter type (see below)
- **Output:** Dataset with non-matching records removed

```python
loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER, text_field="text")
loader.filter(method=FilterMethod.EXACT_DEDUP, text_field="text")
```

---

### `FilterMethod.DEFAULT_TEXT_FILTER`

Removes records with excessive URL content. Useful for cleaning web-scraped data.

| # | Filter | Description | Defaults |
|---|--------|-------------|----------|
| 1 | **URL Ratio** | Removes documents where URLs comprise more than a configurable fraction of the total text. | Max ratio `0.2`, porn URL filtering `True` |
| 2 | **Alphanumeric Ratio** | Removes documents where the ratio of alphanumeric characters falls below a specified threshold. | Min ratio `0.25`, no upper bound |
| 3 | **Word Repetition** | Removes documents with excessive repeated word n-grams above a configurable threshold. | Max ratio `0.5`, 10-word n-grams |
| 4 | **Character Repetition** | Removes documents with excessive repeated character n-grams using a sqrt-based weighting scheme. | Max ratio `0.5`, 10-char n-grams |
| 5 | **Mojibake Detection** | Removes documents containing encoding corruption detected via ftfy badness scoring or regex patterns. | Max badness `1`, max mojibake ratio `0.05` |
| 6 | **Character Length** | Removes documents with character counts outside a specified min/max range. | Min `50` chars, no max |
| 7 | **Average Line Length** | Removes documents with abnormal average line lengths (e.g., word-per-line dumps or minified code). | Min avg `10` chars/line, no max, requires `2` lines |

- **Input:** Raw text data with a flat text field
- **Runs on:** AWS Glue

```python
loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER, text_field="text")
```

---

### `FilterMethod.EXACT_DEDUP`

Removes exact duplicate records by content hash, keeping only unique entries.

- **Input:** Raw text data with a flat text field
- **Runs on:** AWS Glue

```python
loader.filter(method=FilterMethod.EXACT_DEDUP, text_field="text")
```

---

### `FilterMethod.FUZZY_DEDUP`

Removes near-duplicate records using MinHash LSH similarity. Catches paraphrases, minor edits, and boilerplate variants that exact dedup misses.

- **Input:** Raw text data with a flat text field
- **Runs on:** AWS Glue

```python
loader.filter(method=FilterMethod.FUZZY_DEDUP, text_field="text", threshold=0.8)
```

**Fuzzy dedup parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.8` | Jaccard similarity threshold (0.0–1.0). Lower = more aggressive dedup. |
| `num_perm` | `int` | `256` | MinHash permutations. Higher = more accurate, more memory. |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `ngram_size` | `int` | `24` | Character n-gram size for shingling. |
| `num_bands` | `int` | Auto | LSH band count (auto-computed from `threshold`). |
| `rows_per_band` | `int` | Auto | Rows per LSH band (auto-computed from `threshold`). |
| `bands_per_iteration` | `int` | `4` | Bands per memory pass. Higher = faster but more memory. |
| `lowercase` | `bool` | `True` | Lowercase text before comparison. |

---

### `FilterMethod.INVALID_RECORDS`

Removes records that don't conform to schema/format checks for the target training method. Unlike text filters, this runs locally and operates on already-transformed data.

- **Input:** Data in the target training schema (use `transform()` first, or load pre-formatted data)
- **Runs on:** Local

```python
loader.filter(
    method=FilterMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    platform=Platform.SMTJ,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `training_method` | `TrainingMethod` | Yes | Determines which checks apply. |
| `model` | `Model` | Yes | Target Nova model. |
| `platform` | `Platform` | Yes | Target platform (`SMTJ`, `SMHP`, or `BEDROCK`). |
| `eval_task` | `EvaluationTask` | No | For evaluation datasets. |

**Checks applied:**

| Check | Threshold |
|---|---|
| Reserved keywords (`System:`, `USER:`, `Bot:`, `[EOS]`, `<image>`, etc.) | SFT methods only |
| Maximum images per message | 10 |
| Maximum image file size | 10 MB |
| Maximum videos per message | 1 |
| Maximum video file size | 50 MB |
| Maximum video duration | 90 seconds |

---

### Shared Parameters for Text Filters

These apply to `DEFAULT_TEXT_FILTER`, `EXACT_DEDUP`, and `FUZZY_DEDUP`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_path` | `str` | Auto-derived | S3 URI for filtered output. Required for local input. |
| `input_format` | `str` | `"parquet"` | `"parquet"` or `"jsonl"` |
| `output_format` | `str` | `"parquet"` | `"parquet"` or `"jsonl"` |
| `text_field` | `str` | `"text"` | Column name containing the text to process. |
| `extra_args` | `dict` | `None` | Additional kwargs (e.g. `{"max_url_to_text_ratio": 0.3}`). |
| `runtime_manager` | `RuntimeManager` | `None` | Runtime to execute on. Defaults to `GlueRuntimeManager`. |

### Scaling Text Filters

Each text filter can use a different runtime manager for independent scaling:

```python
from amzn_nova_forge.manager import GlueRuntimeManager

text_mgr = GlueRuntimeManager(num_workers=2)
dedup_mgr = GlueRuntimeManager(num_workers=8)

loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER, text_field="text", runtime_manager=text_mgr)
loader.filter(method=FilterMethod.EXACT_DEDUP, text_field="text", runtime_manager=dedup_mgr)
```

If no `runtime_manager` is passed, the SDK creates one from Glue-specific kwargs on that call (e.g. `num_workers=8`).

> **Note:** `runtime_manager` is not auto-chained between filters — pass it explicitly to each step.

---

## Splitting Data

`loader.split(...)` — Divides the dataset into train, validation, and test sets.

- **Input:** Loaded (and optionally transformed/filtered) dataset
- **Output:** Three new loader instances (train, val, test), each fully functional

```python
train, val, test = loader.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
train.save("s3://my-bucket/train.jsonl")
val.save("s3://my-bucket/val.jsonl")
test.save("s3://my-bucket/test.jsonl")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_ratio` | `float` | `0.8` | Fraction for training set. |
| `val_ratio` | `float` | `0.1` | Fraction for validation set. |
| `test_ratio` | `float` | `0.1` | Fraction for test set. |
| `seed` | `int` | `None` | Random seed for reproducibility. |

---

## Saving Data

`loader.save(path)` — Writes the current dataset to a local file or S3 path.

- **Input:** Loaded/transformed/filtered dataset
- **Output:** `.json` or `.jsonl` file at the specified path

```python
loader.save("output/training_data.jsonl")          # local
loader.save("s3://my-bucket/data/training.jsonl")  # S3
```

---

## End-to-End Examples

### SFT from CSV

```python
loader = CSVDatasetLoader()
loader.load("raw_data.csv")
loader.show(n=2)

loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "input", "answer": "output", "system": "system_prompt"},
)
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.save("s3://my-bucket/sft_training_data.jsonl")
```

### SFT from a directory of Arrow files

```python
loader = ArrowDatasetLoader()
loader.load("s3://my-bucket/arrow_shards/")  # scans for .arrow/.feather/.ipc files
loader.show(n=2)

loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "prompt", "answer": "response"},
)
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.save("s3://my-bucket/sft_training_data.jsonl")
```

### RFT with train/test split

```python
loader = JSONLDatasetLoader()
loader.load("rft_data.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.RFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "prompt", "reference_answer": "answer"},
)

train, val, test = loader.split(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05)
train.save("s3://my-bucket/rft_train.jsonl")
val.save("s3://my-bucket/rft_val.jsonl")
```

### OpenAI format with tool calls (Nova 2.0)

```python
loader = JSONLDatasetLoader()
loader.load("openai_with_tools.jsonl")

# Tool calls are only supported on Nova 2.0
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.save("converse_with_tools.jsonl")
```

### Multimodal data with images (OpenAI format)

```python
loader = JSONLDatasetLoader()
loader.load("openai_multimodal.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    multimodal_data_s3_path="s3://my-bucket/images/",
)
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
# Save must use the same bucket as multimodal_data_s3_path
loader.save("s3://my-bucket/multimodal_training.jsonl")
```

### Multimodal data with images (generic format)

```python
loader = JSONLDatasetLoader()
loader.load("image_captions.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={
        "question": "caption_prompt",
        "answer": "caption",
        "image_format": "image_format",
        "s3_uri": "s3_uri",
        "bucket_owner": "bucket_owner",
    },
)
loader.validate(
    method=ValidateMethod.INVALID_RECORDS,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.save("s3://my-bucket/multimodal_training.jsonl")
```

---

## Migrating from Earlier Versions

If you are upgrading from an earlier version of the SDK, the following changes apply. The old syntax continues to work but will emit deprecation warnings.

| Old Syntax | What Changed | New Syntax |
|---|---|---|
| `JSONLDatasetLoader(question="q", answer="a")` | Column mappings moved to `transform()` | `JSONLDatasetLoader()` — pass `column_mappings` to `transform()` instead |
| `loader.transform(TrainingMethod.SFT_LORA, Model.NOVA_LITE)` | `method` param renamed to `training_method`; new `method` param selects transform type | `loader.transform(method=TransformMethod.SCHEMA, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` |
| `loader.transform(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` | Same as above | Same as above |
| `loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)` | `method` param renamed to `training_method`; new `method` param selects validation type | `loader.validate(method=ValidateMethod.INVALID_RECORDS, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` |
| `loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` | Same as above | Same as above |
| `loader.validate(method=ValidateMethod.SCHEMA, ...)` | `ValidateMethod.SCHEMA` deprecated in favour of `INVALID_RECORDS` | `loader.validate(method=ValidateMethod.INVALID_RECORDS, ...)` |
| `loader.save_data("output.jsonl")` | Method renamed | `loader.save("output.jsonl")` |
| `loader.split_data(0.8, 0.1, 0.1)` | Method renamed | `loader.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)` |
