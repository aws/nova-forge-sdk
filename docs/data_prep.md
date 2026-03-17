# Data Preparation Guide

This guide walks through the data preparation workflow using the Nova Forge SDK's dataset loaders.

> **Note:** If you are upgrading from an earlier version of the SDK, some method names and parameters
> have changed. The old names still work but will emit deprecation warnings. See
> [Migrating from Earlier Versions](#migrating-from-earlier-versions) at the end of this guide.

## Overview

The SDK provides three dataset loaders for different file formats, all sharing the same chainable API:

```python
from amzn_nova_forge import (
    CSVDatasetLoader,
    JSONDatasetLoader,
    JSONLDatasetLoader,
    Model,
    TrainingMethod,
)
from amzn_nova_forge.dataset.operations.transform_operation import TransformMethod
from amzn_nova_forge.dataset.operations.validate_operation import ValidateMethod
```

The typical workflow is: **load → transform → validate → save**. Each method returns the loader instance, so you can chain them:

```python
loader = JSONLDatasetLoader()
loader.load("data.jsonl").transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
).validate(
    method=ValidateMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
).save("output.jsonl")
```

---

## Loading Data

### JSONL

```python
loader = JSONLDatasetLoader()
loader.load("training_data.jsonl")

# Also supports S3 paths
loader.load("s3://my-bucket/data/training_data.jsonl")
```

### JSON

```python
loader = JSONDatasetLoader()
loader.load("training_data.json")
```

### CSV

```python
loader = CSVDatasetLoader()
loader.load("training_data.csv")
```

---

## Inspecting Data

Use `show()` to preview the current state of the dataset at any point in the pipeline:

```python
loader.load("data.jsonl")
loader.show()        # Shows first 10 rows
loader.show(n=3)     # Shows first 3 rows
```

---

## Transforming Data

`transform()` applies a transformation to the dataset. The `method` parameter selects which transformation to run.

| Method | Description |
|---|---|
| `TransformMethod.SCHEMA` (default) | Convert between data format schemas (e.g., generic Q/A → Converse, OpenAI → Converse) |

### Schema Transforms

Schema transforms convert your data into the format required by a specific training method and model combination. If the data is already in the correct format, it's a no-op.

#### Column Mappings

When your source data uses different column names than what the SDK expects, provide a `column_mappings` dict to `transform()`. The format is `{"standard_name": "your_column_name"}`.

#### SFT (Supervised Fine-Tuning)

```python
# From plain Q/A format to Converse format
loader = JSONLDatasetLoader()
loader.load("qa_data.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "q", "answer": "a"},
)
```

The SDK also handles OpenAI Messages format automatically — no column mappings needed:

```python
loader = JSONLDatasetLoader()
loader.load("openai_format.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
```

#### RFT (Reinforcement Fine-Tuning)

```python
loader = JSONLDatasetLoader()
loader.load("rft_data.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.RFT_LORA,
    model=Model.NOVA_LITE_2,
    column_mappings={"question": "prompt", "reference_answer": "answer"},
)
```

#### CPT (Continued Pre-Training)

```python
loader = JSONLDatasetLoader()
loader.load("documents.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.CPT,
    model=Model.NOVA_LITE_2,
    column_mappings={"text": "content"},
)
```

#### Evaluation

```python
loader = JSONLDatasetLoader()
loader.load("eval_data.jsonl")
loader.transform(
    method=TransformMethod.SCHEMA,
    training_method=TrainingMethod.EVALUATION,
    model=Model.NOVA_LITE_2,
    column_mappings={"query": "question", "response": "answer"},
)
```

#### Column Mapping Reference

| Training Method | Required Mappings | Optional Mappings |
|---|---|---|
| SFT | `question`, `answer` | `system`, `image_format`/`video_format`, `s3_uri`, `bucket_owner` |
| RFT | `question`, `reference_answer` | `system`, `id`, `tools` |
| Evaluation | `query`, `response` | `images`, `metadata` |
| CPT | `text` | — |

---

## Validating Data

`validate()` checks that the dataset conforms to requirements. The `method` parameter selects which validation to run.

| Method | Description |
|---|---|
| `ValidateMethod.SCHEMA` (default) | Validate dataset structure against the requirements for a training method and model |

### Schema Validation

Run schema validation after transforming to catch issues before submitting a training job.

```python
loader.validate(
    method=ValidateMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
```

Checks include:
- Required fields are present
- Field types and formats are correct
- Role alternation in conversations (user/assistant)
- Optional field consistency across samples
- Forbidden keywords in content

For evaluation datasets, pass the `eval_task` parameter:

```python
from amzn_nova_forge import EvaluationTask

loader.validate(
    method=ValidateMethod.SCHEMA,
    training_method=TrainingMethod.EVALUATION,
    model=Model.NOVA_LITE_2,
    eval_task=EvaluationTask.GEN_QA,
)
```

---

## Splitting Data

`split()` divides the dataset into train, validation, and test sets.

```python
train, val, test = loader.split()  # Default: 80/10/10 split

# Custom ratios
train, val, test = loader.split(
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
)

# Save each split
train.save("train.jsonl")
val.save("val.jsonl")
test.save("test.jsonl")
```

Each returned loader is fully functional — you can call `show()`, `transform()`, `validate()`, or `save()` on it.

---

## Saving Data

`save()` writes the current dataset to a local file or S3 path. Supports `.json` and `.jsonl` formats.

```python
# Save locally
loader.save("output/training_data.jsonl")

# Save to S3
loader.save("s3://my-bucket/data/training_data.jsonl")
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
    method=ValidateMethod.SCHEMA,
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
    method=ValidateMethod.SCHEMA,
    training_method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2,
)
loader.save("converse_with_tools.jsonl")
```

### Multimodal data with images

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
    method=ValidateMethod.SCHEMA,
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
| `loader.validate(TrainingMethod.SFT_LORA, Model.NOVA_LITE)` | `method` param renamed to `training_method`; new `method` param selects validation type | `loader.validate(method=ValidateMethod.SCHEMA, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` |
| `loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)` | Same as above | Same as above |
| `loader.save_data("output.jsonl")` | Method renamed | `loader.save("output.jsonl")` |
| `loader.split_data(0.8, 0.1, 0.1)` | Method renamed | `loader.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)` |
