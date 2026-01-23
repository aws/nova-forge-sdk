# Amazon Nova Customization SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across both SageMaker Training Jobs and SageMaker HyperPod.

## Table of Contents

- [Requirements](#requirements)
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

## Requirements
The SDK requires at least Python 3.12.

## Installation

```bash
pip install amzn-nova-customization-sdk
```

* The SDK requires [sagemaker 2.254.1](https://pypi.org/project/sagemaker/2.254.1/), which is automatically set by pip.


## Quick Start

Here's a simple example to get you started with fine-tuning a Nova model:

```python
import time
from amzn_nova_customization_sdk import *

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

# 3. Setup MLflow monitoring (optional)
mlflow_monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",  # Optional, uses default if not provided
    experiment_name="nova-customization",
    run_name="my-training-run"
)

# 4. Initialize customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://your-bucket/prepared-data.jsonl",
    mlflow_monitor=mlflow_monitor  # Enable MLflow tracking
)

# 5. Start training
training_result = customizer.train(job_name="my-nova-training")
print(f"Training started: {training_result.job_id}")
training_result.dump() # Save job result as local file so it can be reloaded after python env shutdown

# 6. Check training job results
training_result.get_job_status()  # InProgress, Completed, Failed

# 7. Monitor job log
customizer.get_logs() # Directly get logs of most recent job from customizer object
# Or create CloudWatchLogMonitor from a job result or job id
training_job_monitor = CloudWatchLogMonitor.from_job_result(training_result)
training_job_monitor.show_logs(limit=10)

# 8. Get trained model for evaluation
# Wait until job succeed
while training_result.get_job_status()[0] != JobStatus.COMPLETED:
    if training_result.get_job_status()[0] == JobStatus.FAILED:
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

# 9. Deploy model to Bedrock for inference
# Note: Defaults to on-demand deployment when `deploy_platform` is not provided
deployment = customizer.deploy(job_result=training_result)
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
        },
        {
            "Sid": "MLflowSagemaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker-mlflow:AccessUI",
				"sagemaker-mlflow:CreateExperiment",
				"sagemaker-mlflow:CreateModelVersion",
				"sagemaker-mlflow:CreateRegisteredModel",
				"sagemaker-mlflow:CreateRun",
				"sagemaker-mlflow:DeleteTag",
				"sagemaker-mlflow:FinalizeLoggedModel",
				"sagemaker-mlflow:Get*",
				"sagemaker-mlflow:ListArtifacts",
				"sagemaker-mlflow:ListLoggedModelArtifacts",
				"sagemaker-mlflow:LogBatch",
				"sagemaker-mlflow:LogInputs",
				"sagemaker-mlflow:LogLoggedModelParams",
				"sagemaker-mlflow:LogMetric",
				"sagemaker-mlflow:LogModel",
				"sagemaker-mlflow:LogOutputs",
				"sagemaker-mlflow:LogParam",
				"sagemaker-mlflow:RenameRegisteredModel",
				"sagemaker-mlflow:RestoreExperiment",
				"sagemaker-mlflow:RestoreRun",
				"sagemaker-mlflow:Search*",
				"sagemaker-mlflow:SetExperimentTag",
				"sagemaker-mlflow:SetLoggedModelTags",
				"sagemaker-mlflow:SetRegisteredModelAlias",
				"sagemaker-mlflow:SetRegisteredModelTag",
				"sagemaker-mlflow:SetTag",
				"sagemaker-mlflow:TransitionModelVersionStage",
				"sagemaker-mlflow:UpdateExperiment",
				"sagemaker-mlflow:UpdateModelVersion",
				"sagemaker-mlflow:UpdateRegister
            ],
			"Resource": "arn:aws:sagemaker:us-east-1:<account_id>:mlflow-tracking-server/*"
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

#### For Non-Forge Customers

Please use [the `release_v2` branch](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2).
1. `git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git` to pull the HyperPod CLI into a local repository
2. If you are using a Python virtual environment to use the Nova Customization SDK, activate that environment with `source <path to venv>/bin/activate`


#### For Forge Customers
1. Download the latest Hyperpod CLI repo with Forge feature support from remote s3.
```
aws s3 cp s3://nova-forge-c7363-206080352451-us-east-1/v1/ ./ --recursive
pip install -e .
```

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

| Method       | Description                         | Supported Models       |
|--------------|-------------------------------------|------------------------|
| `CPT`        | Continued Pre-Training              | All models (SMHP only) |
| `SFT_LORA`   | Supervised Fine-tuning with LoRA    | All models             |
| `SFT_FULL`   | Full-rank Supervised Fine-tuning    | All models             |
| `RFT_LORA`   | Reinforcement Fine-tuning with LoRA | Nova 2.0 models        |
| `RFT_FULL`   | Full Reinforcement Fine-tuning      | Nova 2.0 models        |
| `EVALUATION` | Model evaluation                    | All models             |

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
| **Monitor**        | Job monitoring and logging                    | `CloudWatchLogMonitor`, `MLflowMonitor`   |

---

## Detailed Module Documentation

### Dataset Module

The Dataset module provides powerful data loading and transformation and validation capabilities for different training formats.

#### Core Classes

**DatasetLoader (Abstract Base Class)**

- **Purpose**: Base class for all dataset loaders
- **Key Methods**:
  - `load(path)`: Load dataset from local or S3 path
  - `show(n=10)`: Display first n rows
  - `split_data(train_ratio, val_ratio, test_ratio)`: Split a provided dataset into randomized train/val/test sets
  - `transform(method, model)`: Transform data to the required format based on the training method a user wants to run
  - `validate(method, model, opt: eval_task)`: Validate data is in the required format for your training method and Nova model selection. 
  - `save_data(save_path)`: Save processed data to a local or S3 path

**JSONLDatasetLoader/JSONDatasetLoader/CSVDatasetLoader**

```python
# Import DatasetLoaders and use the correct DatasetLoader for your data type.
from amzn_nova_customization_sdk.dataset import *

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

| Column Name       | Purpose               | Required  | Training Method | Notes
|-------------------|-----------------------|-----------|-----------------|-------------------------------
| `question`        | User input/query      | ✅        | SFT             | Required field
| `answer`          | Assistant response    | ✅        | SFT             | Required field
| `reasoning_text`  | Chain of thought      | ❌        | SFT             | Optional, 2.0 version only
| `system`          | System prompt         | ❌        | SFT, RFT        | Optional field
| `image_format`    | Image format          | ❌        | SFT             | Optional for multimodal data
| `video_format`    | Video format          | ❌        | SFT             | Optional for multimodal data
| `s3_uri`          | Media S3 URI          | ❌        | SFT             | Required if using media
| `bucket_owner`    | S3 bucket owner       | ❌        | SFT             | Required if using media
| `reference_answer`| Reference response    | ✅        | RFT             | Required field
| `id`              | Identifier            | ❌        | RFT             | Optional field
| `query`           | Evaluation input      | ✅        | Evaluation      | Required field
| `response`        | Evaluation response   | ✅        | Evaluation      | Required field
| `images`          | Image data            | ❌        | Evaluation      | Optional field
| `metadata`        | Additional data       | ❌        | Evaluation      | Optional field
| `text`            | Target domain content | ✅        | CPT             | Required field

**Note:** These mappings only need to be provided to the DatasetLoader when you want to transform plain JSON/JSONL/CSV data into another format.

#### Data Transformation

* The SDK handles transforming your data to the required format for the training method you plan to use.
  * It can currently transform data from plain CSV and plain JSON/JSONL to SFT.
  * Support for OpenAI 'messages' format to SFT will be added in the future.
* If you're missing any fields, the SDK will let you know what fields are required for the method you want to run.
* You can also refer to the above 'Column Mapping' options to figure out the name of the column you need for a specific method.

```python
from amzn_nova_customization_sdk.dataset import *
from amzn_nova_customization_sdk.model import *

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

# Validate that the transformed data is correctly formatted. This will print out a validation success method or point you toward potential errors. 
loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
```

If you're validating a BYOD Evaluation dataset, you need to provide another parameter, `eval_task` to the `validate` function. For example:
```
loader.validate(
    method=TrainingMethod.SFT_LORA, 
    model=Model.NOVA_LITE_2, 
    eval_task=EvaluationTask.GEN_QA
) 
```
The list of available EvaluationTasks can be found in `recipe_config/eval_config.py`. 

**Supported Transform Formats:**

- **Converse Format**: For Nova 1.0 and 2.0 SFT and CPT training
- **OpenAI Format**: For RFT training
- **Evaluation Format**: For BYOD model evaluation tasks (excluding LLM-as-Judge)

### Manager Module

The Manager module handles setting up runtime infrastructure for training jobs.

#### SMTJRuntimeManager (SageMaker Training Jobs)

```python
from amzn_nova_customization_sdk.manager import *

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
from amzn_nova_customization_sdk.manager import *

runtime = SMHPRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=4,
    cluster_name="my-hyperpod-cluster",
    namespace="kubeflow"
)
```

**Supported Instance Types:**

__CPT__

| Model    | Allowed Instance Types (Allowed Instance Counts) |
|----------|--------------------------------------------------|
| Micro    | ml.p5.48xlarge (2, 4, 8, 16, 32)                 |
| Lite     | ml.p5.48xlarge (4, 8, 16, 32)                    |
| Lite 2.0 | ml.p5.48xlarge (4, 8, 16, 32)                    |
| Pro      | ml.p5.48xlarge (6, 12, 24)                       |

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
from amzn_nova_customization_sdk.model import *
from amzn_nova_customization_sdk.monitor import *

# Optional: Setup MLflow monitoring
mlflow_monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
    experiment_name="nova-customization",
    run_name="sft-experiment-1"
)

customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime_manager,
    data_s3_path="s3://bucket/data.jsonl",
    output_s3_path="s3://bucket/output/",  # Optional
    model_path="custom/model/path",        # Optional
    generated_recipe_dir="directory-path", # Optional
    mlflow_monitor=mlflow_monitor,         # Optional: Enable MLflow tracking
    data_mixing_enabled=True               # Optional: Enable data mixing (CPT and SFT only on HyperPod)
)
```

#### Data Mixing

Data mixing allows you to blend your custom training data with Nova's high-quality curated datasets, helping maintain the model's broad capabilities while adding your domain-specific knowledge.

**Key Features:**
- Available for CPT and SFT training for Nova 1 and Nova 2 (both LoRA and Full-Rank) on SageMaker HyperPod
- Mix customer data (0-100%) with Nova's curated data
- Nova data categories include general knowledge and code
- Nova data percentages must sum to 100%

**Example Usage:**

```python
# Initialize with data mixing enabled
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMHPRuntimeManager(...),  # Must use HyperPod
    data_s3_path="s3://bucket/data.jsonl",
    output_s3_path="s3://bucket/output/",  # Optional
    data_mixing_enabled=True
)

# Configure data mixing percentages
customizer.set_data_mixing_config({
    "customer_data_percent": 50,  # 50% your data
    "nova_code_percent": 30,      # 30% Nova code data (30% of Nova's 50%)
    "nova_general_percent": 70    # 70% Nova general data (70% of Nova's 50%)
})

# Or use 100% customer data (no Nova mixing)
customizer.set_data_mixing_config({
    "customer_data_percent": 100,
    "nova_code_percent": 0,
    "nova_general_percent": 0
})
```

**Important Notes:**
- The `dataset_catalog` field is system-managed and cannot be set by users
- Data mixing is only available on SageMaker HyperPod platform for Forge customers.
- Refer to the [Get Forge Subscription] page ('https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access') to enable Nova subscription in your account to use this feature.

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
from amzn_nova_customization_sdk.recipe import *

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
from amzn_nova_customization_sdk.model import *

deployment = customizer.deploy(
    model_artifact_path="s3://bucket/model-artifacts/", # Checkpoint s3 path
    deploy_platform=DeployPlatform.BEDROCK_PT,  # or DeployPlatform.BEDROCK_OD
    pt_units=10,                   # For Provisioned Throughput only
    endpoint_name="my-nova-model",
    bedrock_execution_role_name="BedrockDeployModelExecutionRole" # Optional IAM role name for Bedrock deployment
)
```

Optionally, you can provide a Bedrock execution role name to be used in deployment.
Otherwise, a default Bedrock execution role will be created on your behalf.
You can also use the following method to create a Bedrock execution role with scoped down IAM permissions.


```python
from amzn_nova_customization_sdk.util.bedrock import create_bedrock_execution_role

iam_client = boto3.client("iam")

create_bedrock_execution_role(
    iam_client=iam_client, 
    role_name="BedrockDeployModelExecutionRole",
    bedrock_resource="your-model-name", # Optional: Name of the bedrock resources that IAM role should have restricted create and get access to
    s3_resource="s3-bucket" # Optional: S3 resource that IAM role should have restricted read access to such as the training output bucket
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

Provides job monitoring capabilities through CloudWatch logs and MLflow experiment tracking.

#### MLflowMonitor

Enables experiment tracking with MLflow for training jobs.

```python
from amzn_nova_customization_sdk.monitor import *

# Create MLflow monitor with explicit tracking URI
mlflow_monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# Or use default tracking URI (if available)
mlflow_monitor = MLflowMonitor(
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# Use with NovaModelCustomizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime_manager,
    data_s3_path="s3://bucket/data",
    mlflow_monitor=mlflow_monitor
)
```

**MLflow Integration Features:**
- Automatic logging of training metrics
- Model artifact and checkpoint tracking
- Hyperparameter recording
- Support for SageMaker MLflow tracking servers
- Custom MLflow tracking server support (with proper network configuration)

#### CloudWatchLogMonitor

```python
from amzn_nova_customization_sdk.monitor import *
from amzn_nova_customization_sdk.model import *

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

### Dry Run

The Nova Customization SDK supports `dry_run` mode for the following functions: `train()`, `evaluate()`, and `batch_inference()`.

When calling any of the above functions, you can set the `dry_run` parameter to `True`.
The SDK will still generate your recipe and validate your input, but it won't begin a job.
This feature is useful whenever you want to test or validate inputs and still have a recipe generated, without starting a job.

``` python
# Training dry run
customizer.train(
    job_name="train_dry_run",
    dry_run=True,
    ...
)

# Evaluation dry run
customizer.evaluate(
    job_name="evaluate_dry_run",
    dry_run=True,
    ...
)
```

---

This comprehensive SDK enables end-to-end customization of Amazon Nova models with support for multiple training methods, deployment platforms, and monitoring capabilities. Each module is designed to work together seamlessly while providing flexibility for advanced use cases.

For more information, please see the following:
* Notebook with "quick start" examples located at `samples/nova_quickstart.ipynb`
* Specification document located at `docs/spec.md`
