# Amazon Nova Customization SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across different platforms.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models and Training Methods](#supported-models-and-training-methods)
- [Core Modules Overview](#core-modules-overview)
- [Detailed Module Documentation](#detailed-module-documentation)
  - [Dataset Module](#dataset-module)
  - [Manager Module](#manager-module)
  - [Model Module](#model-module)
  - [Monitor Module](#monitor-module)
- [Examples](#examples)

## Installation

```bash
pip install amzn-nova-customization-sdk
```

* The SDK requires [sagemaker 2.254.1](https://pypi.org/project/sagemaker/2.254.1/), which is automatically set by pip.


## Quick Start

Here's a simple example to get you started with fine-tuning a Nova model:

```python
import time
from amzn_nova_customization_sdk.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_customization_sdk.model.nova_model_customizer import NovaModelCustomizer
from amzn_nova_customization_sdk.model.model_enums import DeployPlatform, Model, TrainingMethod
from amzn_nova_customization_sdk.manager.runtime_manager import SMTJRuntimeManager
from amzn_nova_customization_sdk.model.result import JobStatus
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask
from amzn_nova_customization_sdk.monitor.log_monitor import CloudWatchLogMonitor

# 1. Load and prepare your dataset
loader = JSONLDatasetLoader(question="input", answer="output")
loader.load("s3://your-bucket/training-data.jsonl")
loader.transform(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE)
loader.save_data("s3://your-bucket/prepared-data.jsonl")

# 2. Setup runtime
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4
)

# 3. Initialize customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://your-bucket/prepared-data.jsonl"
)

# 4. Start training
training_result = customizer.train(job_name="my-nova-training")
print(f"Training started: {training_result.job_id}")
training_result.dump() # Save job result as local file so it can be reload after python env shutdown

# 5. Check training job results
training_result.get_job_status()  # InProgress, Completed, Failed

# 6. Monitor job log
customizer.get_logs() # Directly get logs of most recent job from customizer object
# Or create CloudWatchLogMonitor from a job result or job id
training_job_monitor = CloudWatchLogMonitor.from_job_result(training_result)
training_job_monitor.show_logs(limit=10)

# 7. Get trained model for evaluation
# Wait until job succeed
while training_result.get_job_status() != JobStatus.COMPLETED:
    if training_result.get_job_status() == JobStatus.FAILED:
        raise RuntimeError(f"Job failed")
    time.sleep(60)

eval_result = customizer.evaluate(
    job_name='my-mmlu-eval-job',
    eval_task=EvaluationTask.MMLU,
    model_path=training_result.model_artifacts.checkpoint_s3_path # Use trained model path for eval
)
# Save job result to current directory with name of <job_id>_<platform>.json
eval_result.dump()
# Or save job result to certain path and customized name
eval_result.dump(file_path='/volume/path/my-path', file_name='my-name.json') 

# Monitor logs
customizer.get_logs() # Directly get logs of most recent job from customizer object
# Or create CloudWatchLogMonitor from a job result or job id
eval_job_monitor = CloudWatchLogMonitor.from_job_result(eval_result)
eval_job_monitor.show_logs()

# Check eval job status and show results
eval_result.get_job_status()
eval_result.show() # Print eval results

# 8. Deploy model to Bedrock for inference
deployment = customizer.deploy(
  model_artifact_path=training_result.model_artifacts.checkpoint_s3_path,
  deploy_platform=DeployPlatform.BEDROCK_PT,
  pt_units=10
)
```

## Setup

In most cases, the SDK will inform you if the environment lacks the required setup to run a Nova customization job.

Below are some common requirements which you can set up in advance before trying to run a job.

### IAM

The SDK requires certain IAM permissions to perform tasks successfully.  
You can use any role that you like when interacting with the SDK, but that role will need the following permissions:  
_Note that you might not require all permissions depending on your use case._  
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
			"Sid": "ConnectToHyperPodCluster",
			"Effect": "Allow",
			"Action": [
				"eks:DescribeCluster",
				"eks:ListAddons",
				"sagemaker:DescribeCluster"
			],
			"Resource": [
			    "arn:aws:eks:<region>:<account_id>:cluster/*",
			    "arn:aws:sagemaker:<region>:<account_id>:cluster/*"
			]
		},
        {
            "Sid": "StartSageMakerTrainingJob",
            "Effect": "Allow",
            "Action": [
			    "sagemaker:CreateTrainingJob",
			    "sagemaker:DescribeTrainingJob"
			],
            "Resource": "arn:aws:sagemaker:<region>:<account_id>:training-job/*"
        },
        {
            "Sid": "InteractWithSageMakerAndBedrockExecutionRoles",
            "Effect": "Allow",
            "Action": [
                "iam:AttachRolePolicy",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:PassRole",
                "iam:SimulatePrincipalPolicy"
            ],
            "Resource": "arn:aws:iam::<account_id>:role/*"
        },
        {
            "Sid": "CreateSageMakerAndBedrockExecutionRolePolicies",
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:GetPolicy"
            ],
            "Resource": "arn:aws:iam::<account_id>:policy/*"
        },
        {
            "Sid": "HandleTrainingInputAndOutput",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::*"
        },
        {
            "Sid": "AccessCloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogStreams",
                "logs:FilterLogEvents",
                "logs:GetLogEvents"
            ],
            "Resource": "arn:aws:logs:<region>:<account_id>:log-group:*"
        },
        {
            "Sid": "ImportModelToBedrock",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateCustomModel"
            ],
            "Resource": "*"
        },
        {
            "Sid": "DeployModelInBedrock",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateCustomModelDeployment",
                "bedrock:CreateProvisionedModelThroughput",
                "bedrock:GetCustomModel",
                "bedrock:GetCustomModelDeployment",
                "bedrock:GetProvisionedModelThroughput"
            ],
            "Resource": "arn:aws:bedrock:<region>:<account_id>:custom-model/*"
        }
    ]
}
```
- [HyperPod only] If your cluster uses namespace access control, you must have access to the Kubernetes namespace


__Execution Role__  
The execution role is the role that SageMaker assumes to execute training jobs on your behalf.   
___Please see AWS documentation for the recommended set of [execution role permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-createtrainingjob-perms).___  
If performing RFT training, your execution role also must include the following statement:
```
{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:<region>:<account_id>:function:MySageMakerRewardFunction"
}
```

The execution role's trust policy must include the following statement:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

You can optionally set your execution role via:
```
customizer = NovaModelCustomizer(
    infra=SMTJRuntimeManager(
        execution_role='arn:aws:iam::123456789012:role/MyExecutionRole' # Explicitly set execution role
        instance_count=1,
        instance_type='ml.g5.12xlarge',
    ),
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    data_s3_path='s3://input-bucket/input.jsonl'
)
```
If you don’t explicitly set an execution role, the SDK automatically uses the IAM role associated with the credentials you’re using to make the SDK call.

### Instances

Nova customization jobs also require access to enough of the right instance type to run:
- The requested instance type and count should be compatible with the requested job. The SDK will validate your instance configuration for you.
- The [SageMaker account quotas](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) for using the requested instance type in training jobs (for SMTJ) or HyperPod clusters (for SMHP) should allow the requested number of instances.
- (For SMHP) The selected HyperPod cluster should have a [Restricted Instance Group](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-hp-cluster.html) with enough instances of the right type to run the requested job. The SDK will validate that your cluster contains a valid instance group.

### HyperPod CLI

For HyperPod-based customization jobs, the SDK uses the [SageMaker HyperPod CLI](https://github.com/aws/sagemaker-hyperpod-cli/) to connect to HyperPod Clusters and start jobs.

Please use [the `release_v2` branch](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2).

Steps:
1. `git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git` to pull the HyperPod CLI into a local repository
2. If you are using a Python virtual environment to use the Nova Customization SDK, activate that environment with `source <path to venv>/bin/activate`
3. Follow the installation instructions [in the HyperPod CLI README](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2?tab=readme-ov-file#installation) to set up the CLI. As of November 2025, the steps are as follows:
    1. Make sure that `helm` is installed with `helm --help`. If it isn't, use the below script to install it:
    ```
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    ./get_helm.sh
    rm -f ./get_helm.sh
    ```
    2. `cd` into the directory where you cloned the HyperPod CLI
    3. Run `pip install .` to install the CLI
    4. Run `hyperpod --help` to verify that the CLI was installed

## Supported Models and Training Methods

### Models

| Model         | Version | Model Type                     | Context Length |
| ------------- | ------- | ------------------------------ | -------------- |
| `NOVA_MICRO`  | 1.0     | `amazon.nova-micro-v1:0:128k`  | 128k tokens    |
| `NOVA_LITE`   | 1.0     | `amazon.nova-lite-v1:0:300k`   | 300k tokens    |
| `NOVA_LITE_2` | 2.0     | `amazon.nova-2-lite-v1:0:256k` | 256k tokens    |
| `NOVA_PRO`    | 1.0     | `amazon.nova-pro-v1:0:300k`    | 300k tokens    |

### Training Methods

| Method         | Description                         | Supported Models |
| -------------- | ----------------------------------- | ---------------- |
| `SFT_LORA`     | Supervised Fine-tuning with LoRA    | All models       |
| `SFT_FULLRANK` | Full-rank Supervised Fine-tuning    | All models       |
| `RFT_LORA`     | Reinforcement Fine-tuning with LoRA | Nova 2.0 models  |
| `RFT`          | Full Reinforcement Fine-tuning      | Nova 2.0 models  |
| `EVALUATION`   | Model evaluation                    | All models       |

### Platform Support

| Platform | Description             | Models Supported |
| -------- | ----------------------- | ---------------- |
| `SMTJ`   | SageMaker Training Jobs | All models       |
| `SMHP`   | SageMaker HyperPod      | All models       |

## Core Modules Overview

The Nova Customization SDK is organized into the following modules:

| Module             | Purpose                                       | Key Components                             |
| ------------------ | --------------------------------------------- | ------------------------------------------ |
| **Dataset**        | Data loading, transformation, and preparation | `DatasetLoader`, `DatasetTransformer`      |
| **Manager**        | Runtime infrastructure management             | `SMTJRuntimeManager`, `SMHPRuntimeManager` |
| **Model**          | Main SDK entrypoint and orchestration         | `NovaModelCustomizer`                      |
| **Monitor**        | Job monitoring and logging                    | `CloudWatchLogMonitor`                     |

---

## Detailed Module Documentation

### Dataset Module

The Dataset module provides powerful data loading and transformation capabilities for different training formats.

#### Core Classes

**DatasetLoader (Abstract Base Class)**

- **Purpose**: Base class for all dataset loaders
- **Key Methods**:
  - `load(path)`: Load dataset from local or S3 path
  - `show(n=10)`: Display first n rows
  - `split_data(train_ratio, val_ratio, test_ratio)`: Split a provided dataset into randomized train/val/test sets
  - `transform(method, model)`: Transform data to the required format based on the training method a user wants to run
  - `save_data(save_path)`: Save processed data to a local or S3 path

**JSONLDatasetLoader/JSONDatasetLoader/CSVDatasetLoader**

```python
# Import the correct DatasetLoader for your data type.
from amzn_nova_customization_sdk.dataset.dataset_loader import JSONLDatasetLoader

# Column mapping for your dataset structure
# These columns are used for transforming the right columns in your dataset to the right values.
loader = JSONLDatasetLoader(
    question="user_input",      # Maps to your question column
    answer="assistant_response", # Maps to your answer column
    system="system_prompt"      # Optional system message column
)

# Load from local file or S3 so the data can be transformed, split, or saved.
loader.load("path/to/data.jsonl")
```

#### Column Mapping Options

| Column Name       | Purpose               | Required  | Training Method| Notes
|-------------------|-----------------------|-----------|----------------|-------------------------------
| `question`        | User input/query      | ✅        | SFT            | Required field
| `answer`          | Assistant response    | ✅        | SFT            | Required field
| `reasoning_text`  | Chain of thought      | ❌        | SFT            | Optional, 2.0 version only
| `system`          | System prompt         | ❌        | SFT, RFT       | Optional field
| `image_format`    | Image format          | ❌        | SFT            | Optional for multimodal data
| `video_format`    | Video format          | ❌        | SFT            | Optional for multimodal data
| `s3_uri`          | Media S3 URI          | ❌        | SFT            | Required if using media
| `bucket_owner`    | S3 bucket owner       | ❌        | SFT            | Required if using media
| `reference_answer`| Reference response    | ✅        | RFT            | Required field
| `id`              | Identifier            | ❌        | RFT            | Optional field
| `query`           | Evaluation input      | ✅        | Evaluation     | Required field
| `response`        | Evaluation response   | ✅        | Evaluation     | Required field
| `images`          | Image data            | ❌        | Evaluation     | Optional field
| `metadata`        | Additional data       | ❌        | Evaluation     | Optional field

**Note:** These mappings only need to be provided to the DatasetLoader when you want to transform plain JSON/JSONL/CSV data into another format.

#### Data Transformation

* The SDK handles transforming your data to the required format for the training method you plan to use.
  * It can currently transform data from plain CSV and plain JSON/JSONL to SFT.
  * Support for OpenAI 'messages' format to SFT will be added in the future.
* If you're missing any fields, the SDK will let you know what fields are required for the method you want to run.
* You can also refer to the above 'Column Mapping' options to figure out the name of the column you need for a specific method.

```python
from amzn_nova_customization_sdk.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod

loader = JSONLDatasetLoader(
    question="user_input",      # Maps to your question column
    answer="assistant_response", # Maps to your answer column
    system="system_prompt"      # Optional system message column
)

# Load from local file or S3 so the data can be transformed, split, or saved.
loader.load("path/to/data.jsonl")

# Transform for SFT training on Nova 2.0
loader.transform(
    method=TrainingMethod.SFT_LORA,
    model=Model.NOVA_LITE_2
)
```

**Supported Transform Formats:**

- **Converse Format**: For Nova 1.0 and 2.0 SFT training
- **OpenAI Format**: For RFT training
- **Evaluation Format**: For model evaluation tasks

### Manager Module

The Manager module handles setting up runtime infrastructure for training jobs.

#### SMTJRuntimeManager (SageMaker Training Jobs)

```python
from amzn_nova_customization_sdk.manager.runtime_manager import SMTJRuntimeManager

runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4
)
```

**Supported Instance Types:**

__SFT__

| Model    | Run Type        | Allowed Instance Types (Allowed Instance Counts)                                                                                      |
|----------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Micro    | LoRA            | ml.g5.12xlarge (1), ml.g5.48xlarge (1), ml.g6.12xlarge (1), ml.g6.48xlarge (1), ml.p4d.24xlarge (2, 4), ml.p5.48xlarge (2, 4)         |
| Micro    | Full-Rank       | ml.g5.48xlarge (1), ml.g6.48xlarge (1), ml.p4d.24xlarge (2, 4), ml.p5.48xlarge (2, 4)                                                 |
| Lite     | LoRA            | ml.g5.12xlarge (1), ml.g5.48xlarge (1), ml.g6.12xlarge (1), ml.g6.48xlarge (1), ml.p4d.24xlarge (4, 8, 16), ml.p5.48xlarge (4, 8, 16) |
| Lite     | Full-Rank       | ml.p4d.24xlarge (4, 8, 16), ml.p5.48xlarge (4, 8, 16)                                                                                 |
| Lite 2.0 | LoRA, Full-Rank | ml.p5.48xlarge (4, 8, 16), ml.p5en.48xlarge (4, 8, 16)                                                                                |
| Pro      | LoRA            | ml.p4d.24xlarge (6, 12, 24), ml.p5.48xlarge (6, 12, 48)                                                                               |
| Pro      | Full-Rank       | ml.p5.48xlarge (3, 6, 12, 24)                                                                                                         |

__RFT__

| Model    | Run Type        | Allowed Instance Types (Allowed Instance Counts)                                                                                      |
|----------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Lite 2.0 | LoRA, Full-Rank | ml.p5.48xlarge (4), ml.p5en.48xlarge (4)                                                                                              |

__Evaluation__

_All allow 1, 2, 4, 8, or 16 instances_

| Model      | Allowed Instance Types                                                                                                                                                                      |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Micro      | ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.12xlarge, ml.g5.16xlarge, ml.g5.24xlarge, ml.g6.4xlarge, ml.g6.8xlarge, ml.g6.12xlarge, ml.g6.16xlarge, ml.g6.24xlarge, ml.g6.48xlarge, ml.p5.48xlarge  |
| Lite       | ml.g5.12xlarge, ml.g5.24xlarge, ml.g6.12xlarge, ml.g6.24xlarge, ml.g6.48xlarge, ml.p5.48xlarge                                                                                              |
| Lite 2.0   | ml.p4d.24xlarge, ml.p5.48xlarge                                                                                                                                                             |
| Pro        | ml.p5.48xlarge                                                                                                                                                                              |

---------------------

#### SMHPRuntimeManager (SageMaker HyperPod)

```python
from amzn_nova_customization_sdk.manager.runtime_manager import SMHPRuntimeManager

runtime = SMHPRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",
    namespace="kubeflow"
)
```

**Supported Instance Types:**

__SFT__

| Model     | Run Type        | Allowed Instance Types (Allowed Instance Counts)         |
|-----------|-----------------|----------------------------------------------------------|
| Micro     | LoRA, Full-Rank | ml.p5.48xlarge (2, 4, 8)                                 |
| Lite      | LoRA, Full-Rank | ml.p5.48xlarge (4, 8, 16)                                |
| Lite 2.0  | LoRA, Full-Rank | ml.p5.48xlarge (4, 8, 16), ml.p5en.48xlarge (4, 8, 16)   |
| Pro       | LoRA, Full-Rank | ml.p5.48xlarge (6, 12, 48)                               |

__RFT__

| Model     | Run Type        | Allowed Instance Types (Allowed Instance Counts)  |
|-----------|-----------------|---------------------------------------------------|
| Lite 2.0  | LoRA, Full-Rank | ml.p5.48xlarge (2, 4, 8, 16),                     |

__Evaluation__

_All allow 1, 2, 4, 8, or 16 instances_

| Model      | Allowed Instance Types                                                                                                                                                                      |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Micro      | ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.12xlarge, ml.g5.16xlarge, ml.g5.24xlarge, ml.g6.4xlarge, ml.g6.8xlarge, ml.g6.12xlarge, ml.g6.16xlarge, ml.g6.24xlarge, ml.g6.48xlarge, ml.p5.48xlarge  |
| Lite       | ml.g5.12xlarge, ml.g5.24xlarge, ml.g6.12xlarge, ml.g6.24xlarge, ml.g6.48xlarge, ml.p5.48xlarge                                                                                              |
| Lite 2.0   | ml.p4d.24xlarge, ml.p5.48xlarge                                                                                                                                                             |
| Pro        | ml.p5.48xlarge                                                                                                                                                                              |

### Model Module

The Model module is the main entrypoint containing the `NovaModelCustomizer` class.

#### NovaModelCustomizer

**Initialization:**

```python
from amzn_nova_customization_sdk.model.nova_model_customizer import NovaModelCustomizer
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod

customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime_manager,
    data_s3_path="s3://bucket/data.jsonl",
    output_s3_path="s3://bucket/output/",  # Optional
    model_path="custom/model/path",        # Optional
    generated_recipe_dir="directory-path"  # Optional
)
```

#### Core Methods

**1. Training**

```python
result = customizer.train(
    job_name="my-training-job",
    recipe_path="custom-recipe.yaml",  # Optional if you bring your own recipe YAML
    overrides={                        # Optional overrides
        'max_epochs': 3,
        'lr': 5e-6,
        'warmup_steps': 100,
        'loraplus_lr_ratio': 16.0,
        'global_batch_size': 64,
        'max_length': 8192
    },
    rft_lambda_arn="arn:aws:lambda:..."  # For RFT only
)
```

**2. Evaluation**

```python
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask

eval_result = customizer.evaluate(
    job_name="model-evaluation",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://bucket/model-artifacts/",  # Optional model path override
    subtask="abstract_algebra",  # Optional
    processor={ # Optional, only needed for BYOM (Bring your own metric) Eval
        'lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<function-name>', 
        'preprocessing': { # Optional, default to True if not provided
            'enabled': True
        },
        'postprocessing': { # Optional, default to True if not provided
            'enabled': True
        },
        # Optional, Built-in aggregation function (valid options: min, max, average, sum), default to average
        'aggregation': 'average'
    },
    rl_env={ # Optional, only needed for RFT Eval
        'reward_lambda_arn': 'arn:aws:lambda:<region>:<account_id>:function:<reward-function-name>'
    },
    overrides={  # Optional overrides
        'max_new_tokens': 2048,
        'temperature': 0.1,
        'top_p': 0.9
    }
)

eval_result.get_job_status()  # This can be run to check the job status of the current evaluation job.

eval_result.dump() # Save job result to current directory with default naming <job_id>_<platform>.json
eval_result.dump(file_path='my/custom/path', file_name='my-custom-name.json') # Save job result to certain path/name that user specified.

from amzn_nova_customization_sdk.model.result import BaseJobResult
eval_result = BaseJobResult.load('my-path/my-job-result.json') # Load job result from the local file
```

**3. Deployment**

```python
from amzn_nova_customization_sdk.model.model_enums import DeployPlatform

deployment = customizer.deploy(
    model_artifact_path="s3://bucket/model-artifacts/", # Checkpoint s3 path
    deploy_platform=DeployPlatform.BEDROCK_PT,  # or DeployPlatform.BEDROCK_OD
    pt_units=10,                   # For Provisioned Throughput only
    endpoint_name="my-nova-model"
)
```

**4. Batch Inference**

```python
inference_result = customizer.batch_inference(
    job_name="batch-inference",
    input_path="s3://bucket/inference-input.jsonl",
    output_s3_path="s3://bucket/inference-output/",
    model_path="s3://bucket/model-artifacts/" # Optional
)

inference_result.get_job_status() # This can be run to check the job status of the current evaluation job.
inference_result.get("s3://bucket/output/inference_results.jsonl") # After the job status is COMPLETED, this will download a user-friendly "inference_results.jsonl" file to a user-provided s3 location.
```

**5. Log Monitoring**

```python
# View recent logs
customizer.get_logs(limit=100, start_from_head=False)

# View logs from beginning
customizer.get_logs(start_from_head=True)
```

### Monitor Module

Provides CloudWatch log monitoring capabilities.

#### CloudWatchLogMonitor

```python
from amzn_nova_customization_sdk.monitor import CloudWatchLogMonitor
from amzn_nova_customization_sdk.model.model_enums import Platform

eval_result = customizer.evaluate(
    job_name="model-evaluation",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://bucket/model-artifacts/", # Optional model path override
    subtask="abstract_algebra",  # Optional
    overrides={                  # Optional overrides
        'max_new_tokens': 2048,
        'temperature': 0.1,
        'top_p': 0.9
    }
)

# Create from job result
monitor = CloudWatchLogMonitor.from_job_result(
    job_result=my_evaluation_job_result
)
# Or Create from job id 
from datetime import datetime
monitor = CloudWatchLogMonitor.from_job_id(
    job_id="job-id",
    platform=Platform.SMTJ,
    started_time=datetime(year=2025, month=11, day=1, hour=20), # Optional, job start time
    cluster_name="cluster_name", # Optional, SMHP cluster name, only needed when platform is SMHP,
    namespace="namespace", # Optional, SMHP namespace, only needed when platform is SMHP
)

# View logs
monitor.show_logs(limit=50, start_from_head=False)

# Get raw logs as list
logs = monitor.get_logs(limit=100)
```

## Additional features

### Iterative training

The Nova Customization SDK supports iterative fine-tuning of Nova models.

This is done by progressively running fine-tuning jobs on the output checkpoint from the previous job:

``` python
# Stage 1: Initial training on base model
stage1_customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    data_s3_path="s3://bucket/stage1-data.jsonl",
    output_s3_path="s3://bucket/stage1-output"
)

stage1_result = stage1_customizer.train(job_name="stage1-training")
# Wait for completion...
stage1_checkpoint = stage1_result.model_artifacts.checkpoint_s3_path

# Stage 2: Continue training from Stage 1 checkpoint
stage2_customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    data_s3_path="s3://bucket/stage2-data.jsonl",
    output_s3_path="s3://bucket/stage2-output",
    model_path=stage1_checkpoint  # Use previous checkpoint
)

stage2_result = stage2_customizer.train(job_name="stage2-training")
```

**Note:** Iterative fine-tuning requires using the same model and training method (LoRA vs Full-Rank) across all stages.

---

This comprehensive SDK enables end-to-end customization of Amazon Nova models with support for multiple training methods, deployment platforms, and monitoring capabilities. Each module is designed to work together seamlessly while providing flexibility for advanced use cases
