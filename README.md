# Amazon Nova Customization SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across both SageMaker Training Jobs and SageMaker HyperPod.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Supported Models and Training Methods](#supported-models-and-training-methods)
- [Core Modules Overview](#core-modules-overview)
- [Additional Features](#additional-features)
- [Getting Started](#getting-started)

## Installation

```bash
pip install amzn-nova-customization-sdk
```
* The SDK requires [sagemaker 2.254.1](https://pypi.org/project/sagemaker/2.254.1/), which is automatically set by pip.


## Setup

In most cases, the SDK will inform you if the environment lacks the required setup to run a Nova customization job.

Below are some common requirements which you can set up in advance before trying to run a job.

### Python Version
* The SDK also requires at least Python 3.12.

### IAM Roles/Policies

The SDK requires certain IAM permissions to perform tasks successfully. You can use any role that you like when interacting with the SDK, but that role will need the following permissions: 
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
                "s3:PutObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts"
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
- _Note that you might not require all permissions depending on your use case._
- [HyperPod only] If your cluster uses namespace access control, you must have access to the Kubernetes namespace

#### __Execution Role__  
The execution role is the role that SageMaker assumes to execute training jobs on your behalf. This can be separate from the role defined above, which is the role you assume when using the SDK.
___Please see AWS documentation for the recommended set of [execution role permissions](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-createtrainingjob-perms).___

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
If performing RFT training, your execution role also must include the following statement:
```
{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:<region>:<account_id>:function:MySageMakerRewardFunction"
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

1. Please use [the `release_v2` branch](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2). 
```
git clone -b release_v2 https://github.com/aws/sagemaker-hyperpod-cli.git
```
2. If you are using a Python virtual environment to use the Nova Customization SDK, activate that environment with `source <path to venv>/bin/activate`


#### For Forge Customers
1. Download the latest Hyperpod CLI repo with Forge feature support from remote s3.
```
aws s3 cp s3://nova-forge-c7363-206080352451-us-east-1/v1/ ./ --recursive 
mkdir -p src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo
git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git src/hyperpod_cli/sagemaker_hyperpod_recipes/launcher/nemo/nemo_framework_launcher --recursive 
pip install -e .
```

2. Follow the installation instructions [in the HyperPod CLI README](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2?tab=readme-ov-file#installation) to set up the CLI. As of November 2025, the steps are as follows:
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
---
## Supported Models and Training Methods

### Models

| Model         | Version | Model Type                     | Context Length |
| ------------- | ------- | ------------------------------ | -------------- |
| `NOVA_MICRO`  | 1.0     | `amazon.nova-micro-v1:0:128k`  | 128k tokens    |
| `NOVA_LITE`   | 1.0     | `amazon.nova-lite-v1:0:300k`   | 300k tokens    |
| `NOVA_LITE_2` | 2.0     | `amazon.nova-2-lite-v1:0:256k` | 256k tokens    |
| `NOVA_PRO`    | 1.0     | `amazon.nova-pro-v1:0:300k`    | 300k tokens    |

### Training Methods

| Method       | Description                              | Supported Models       |
|--------------|------------------------------------------|------------------------|
| `CPT`        | Continued Pre-Training                   | All models (SMHP only) |
| `DPO_LORA`   | Direct Preference Optimization with LoRA | Nova 1.0 models        |
| `DPO_FULL`   | Full-rank Direct Preference Optimization | Nova 1.0 models        |
| `SFT_LORA`   | Supervised Fine-tuning with LoRA         | All models             |
| `SFT_FULL`   | Full-rank Supervised Fine-tuning         | All models             |
| `RFT_LORA`   | Reinforcement Fine-tuning with LoRA      | Nova 2.0 models        |
| `RFT_FULL`   | Full Reinforcement Fine-tuning           | Nova 2.0 models        |
| `EVALUATION` | Model evaluation                         | All models             |

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

* For detailed API documentation: See `docs/spec.md`  
* For usage examples: See `samples/nova_quickstart.ipynb`

### Dataset Module
Handles data loading, transformation, and validation for training datasets.

**Main Methods:**
- `load()` - Load dataset from local or S3 path
- `transform()` - Transform data to required format for training method
- `validate()` - Validate dataset format and content
- `split_data()` - Split dataset into train/validation/test sets
- `save_data()` - Save processed dataset to local or S3 path
- `show()` - Display sample rows from dataset

**Key Classes:**
- DatasetLoader - Abstract class for dataset loading
- JSONDatasetLoader - For loading JSON data
- JSONLDatasetLoader - For loading JSONL data
- CSVDatasetLoader - For loader CSV data

### Manager Module
Manages runtime infrastructure for executing training and evaluation jobs.
For the allowed instance types for each model/method combination, see `docs/instance_type_spec.md`.

**Main Methods:**
- `execute()` - Start a training or evaluation job
- `cleanup()` - Stop and clean up a running job

**Key Classes:**
- `SMTJRuntimeManager` - For SageMaker Training Jobs
- `SMHPRuntimeManager` - For SageMaker HyperPod clusters

### Model Module
Provides the main SDK entrypoint for orchestrating model customization workflows.

**Main Methods:**
- `train()` - Launch a training job
- `evaluate()` - Launch an evaluation job
- `deploy()` - Deploy trained model to Amazon Bedrock
- `batch_inference()` - Run batch inference on trained model
- `get_logs()` - Retrieve CloudWatch logs for current job
- `get_data_mixing_config()` - Get data mixing configuration
- `set_data_mixing_config()` - Set data mixing configuration

**Key Class:**
- `NovaModelCustomizer` - Main orchestration class

### Monitor Module
Provides job monitoring and experiment tracking capabilities.

**Main Methods:**
- `show_logs()` - Display CloudWatch logs
- `get_logs()` - Retrieve logs as list
- `from_job_result()` - Create monitor from job result
- `from_job_id()` - Create monitor from job ID

**Key Classes:**
- `CloudWatchLogMonitor` - For viewing job logs
- `MLflowMonitor` - For experiment tracking

---
## Additional Features

### Iterative Training

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

### Data Mixing
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
- Refer to the [Get Forge Subscription]('https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access') page to enable Nova subscription in your account to use this feature.


---
## Getting Started
This comprehensive SDK enables end-to-end customization of Amazon Nova models with support for multiple training methods, deployment platforms, and monitoring capabilities. Each module is designed to work together seamlessly while providing flexibility for advanced use cases.

To get started customizing Nova models, please see the following files:
* Notebook with "quick start" examples to start customizing at `samples/nova_quickstart.ipynb`
* Specification document with detailed information about each module at `docs/spec.md`
