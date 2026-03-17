# Amazon Nova Forge SDK

A comprehensive Python SDK for fine-tuning and customizing Amazon Nova models. This SDK provides a unified interface for training, evaluation, deployment, and monitoring of Nova models across both SageMaker Training Jobs and SageMaker HyperPod.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Supported Models and Training Methods](#supported-models-and-training-methods)
- [Core Modules Overview](#core-modules-overview)
- [Additional Features](#additional-features)
- [Getting Started](#getting-started)
- [Security Best Practices for SDK Users](#security-best-practices-for-sdk-users)

## Installation

```bash
pip install amzn-nova-forge
```
* The SDK requires [sagemaker](https://pypi.org/project/sagemaker/), which is automatically set by pip.


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
                "iam:SimulatePrincipalPolicy",
                "iam:PutRolePolicy",
                "iam:TagRole",
                "iam:ListAttachedRolePolicies"

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
            "Sid": "BedrockCustomizationJobs",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelCustomizationJob",
                "bedrock:GetModelCustomizationJob",
                "bedrock:StopModelCustomizationJob"
            ],
            "Resource": [
                "arn:aws:bedrock:<region>:<account_id>:model-customization-job/*",
                "arn:aws:bedrock:<region>:<account_id>:custom-model/*"
            ]
        },
        {
            "Sid": "DeployModelInBedrock",
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateCustomModelDeployment",
                "bedrock:CreateProvisionedModelThroughput",
                "bedrock:GetCustomModel",
                "bedrock:GetCustomModelDeployment",
                "bedrock:GetProvisionedModelThroughput",
                "bedrock:ListCustomModelDeployments"
            ],
            "Resource": "arn:aws:bedrock:<region>:<account_id>:custom-model/*"
        },
        {
            "Sid": "DeployAndInvokeModelInSageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateModel",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:InvokeEndpoint",
                "sagemaker:InvokeEndpointWithResponseStream",
                "sagemaker:UpdateEndpoint"
            ],
            "Resource": [
               "arn:aws:sagemaker:<region>:<account_id>:endpoint/*",
               "arn:aws:sagemaker:<region>:<account_id>:endpoint-config/*",
               "arn:aws:sagemaker:<region>:<account_id>:model/*"
              ]
        },
        {
            "Sid": "MLflowSageMaker",
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
				"sagemaker-mlflow:UpdateRegisteredModel"
            ],
			"Resource": "arn:aws:sagemaker:<region>:<account_id>:mlflow-tracking-server/*"
        }
    ]
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

If performing RFT Multiturn training, you also need the following additional permissions:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SSMCommandsForRFTMultiturn",
            "Effect": "Allow",
            "Action": [
                "ssm:SendCommand",
                "ssm:GetCommandInvocation",
                "ssm:ListCommandInvocations"
            ],
            "Resource": [
                "arn:aws:ec2:<region>:<account_id>:instance/*",
                "arn:aws:ssm:<region>::document/AWS-RunShellScript"
            ]
        },
        {
            "Sid": "ECSTaskManagementForRFTMultiturn",
            "Effect": "Allow",
            "Action": [
                "ecs:DeregisterTaskDefinition",
                "ecs:DescribeTasks",
                "ecs:ListTasks",
                "ecs:RunTask",
                "ecs:StopTask"
            ],
            "Resource": [
                "arn:aws:ecs:<region>:<account_id>:cluster/*",
                "arn:aws:ecs:<region>:<account_id>:task/*",
                "arn:aws:ecs:<region>:<account_id>:task-definition/*"
            ]
        },
        {
            "Sid": "RFTMultiturnInfraDiscovery",
            "Effect": "Allow",
            "Action": [
                "cloudformation:ListStacks",
                "ecs:DescribeClusters"
            ],
            "Resource": "*"
        }
    ]
}
```
> **Note:** `cloudformation:ListStacks` and `ecs:DescribeClusters` are read-only actions that do not support resource-level permissions in IAM, so `Resource` must be `"*"`. They cannot create, modify, or delete any resources.

For SMTJ jobs you can set your execution role via:
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

#### __EKS Cluster Access (HyperPod Only)__
After creating your execution role, you must grant it access to your HyperPod cluster's EKS cluster. This is required for the SDK to submit jobs to HyperPod.

**Step 1: Create an access entry for your execution role**
```bash
aws eks create-access-entry \
  --cluster-name <your-cluster-name> \
  --principal-arn arn:aws:iam::<account_id>:role/<your-execution-role-name>
```

**Step 2: Associate the cluster admin policy**
```bash
aws eks associate-access-policy \
  --cluster-name <your-cluster-name> \
  --principal-arn arn:aws:iam::<account_id>:role/<your-execution-role-name> \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
  --access-scope type=cluster
```

Replace the following placeholders:
- `<your-cluster-name>`: Your HyperPod cluster's EKS cluster name (e.g., `sagemaker-my-cluster-eks`)
- `<account_id>`: Your AWS account ID
- `<your-execution-role-name>`: The name of your execution role (e.g., `NovaForgeSdkExecutionRole`)


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
2. If you are using a Python virtual environment to use the Nova Forge SDK, activate that environment with `source <path to venv>/bin/activate`


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
| `CPT`                | Continued Pre-Training                   | All models (SMHP only) |
| `DPO_LORA`           | Direct Preference Optimization with LoRA | Nova 1.0 models        |
| `DPO_FULL`           | Full-rank Direct Preference Optimization | Nova 1.0 models        |
| `SFT_LORA`           | Supervised Fine-tuning with LoRA         | All models             |
| `SFT_FULL`           | Full-rank Supervised Fine-tuning         | All models             |
| `RFT_LORA`           | Reinforcement Fine-tuning with LoRA      | Nova 2.0 models        |
| `RFT_FULL`           | Full Reinforcement Fine-tuning           | Nova 2.0 models        |
| `RFT_MULTITURN_LORA` | RFT Multiturn with LoRA                  | Nova 2.0 models        |
| `RFT_MULTITURN_FULL` | Full RFT Multiturn                       | Nova 2.0 models        |
| `EVALUATION`         | Model evaluation                         | All models             |

### Platform Support

| Platform  | Description                      | Models Supported |
| --------- | -------------------------------- | ---------------- |
| `SMTJ`    | SageMaker Training Jobs          | All models       |
| `SMHP`    | SageMaker HyperPod               | All models       |
| `BEDROCK` | Amazon Bedrock (Managed Service) | All models       |

## Core Modules Overview

The Nova Forge SDK is organized into the following modules:

| Module             | Purpose                                       | Key Components                                                   |
| ------------------ | --------------------------------------------- | ---------------------------------------------------------------- |
| **Dataset**        | Data loading, transformation, and preparation | `JSONLDatasetLoader`, `JSONDatasetLoader`, `CSVDatasetLoader`    |
| **Manager**        | Runtime infrastructure management             | `SMTJRuntimeManager`, `SMHPRuntimeManager`, `BedrockRuntimeManager` |
| **Model**          | Main SDK entrypoint and orchestration         | `NovaModelCustomizer`                                             |
| **Monitor**        | Job monitoring and logging                    | `CloudWatchLogMonitor`, `MLflowMonitor`                          |
| **RFT Multiturn**  | Reinforcement fine-tuning infrastructure      | `RFTMultiturnInfrastructure`                                      |

* For detailed API documentation: See [`docs/spec.md`](docs/spec.md)
* For usage examples: See [`samples/nova_quickstart.ipynb`](samples/nova_quickstart.ipynb)
* For RFT Singleturn examples: See [`samples/rft_singleturn_quickstart.ipynb`](samples/rft_singleturn_quickstart.ipynb)
* For RFT Multiturn documentation: See [`docs/rft_multiturn.md`](docs/rft_multiturn.md)
* For RFT Multiturn examples: See [`samples/rft_multiturn_quickstart.ipynb`](samples/rft_multiturn_quickstart.ipynb)

### Dataset Module
Handles data loading, transformation, validation, and persistence for training datasets. Supports JSONL, JSON, and CSV formats from local files or S3.

```python
loader = JSONLDatasetLoader()
loader.load("data.jsonl")
loader.transform(method=TransformMethod.SCHEMA, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.validate(method=ValidateMethod.SCHEMA, training_method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
loader.save("output.jsonl")
```

For the full data preparation guide including column mappings, splitting, and end-to-end examples, see [docs/data_prep.md](docs/data_prep.md).

### Manager Module
Manages runtime infrastructure for executing training and evaluation jobs.
For the allowed instance types for each model/method combination, see `docs/instance_type_spec.md`.

**Main Methods:**
- `execute()` - Start a training or evaluation job
- `cleanup()` - Stop and clean up a running job

**Key Classes:**
- `SMTJRuntimeManager` - For SageMaker Training Jobs
- `SMHPRuntimeManager` - For SageMaker HyperPod clusters
- `BedrockRuntimeManager` - For Amazon Bedrock managed service

### Model Module
Provides the main SDK entrypoint for orchestrating model customization workflows.

**Main Methods:**
- `train()` - Launch a training job
- `evaluate()` - Launch an evaluation job
- `deploy()` - Deploy trained model to Amazon SageMaker or Bedrock
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
- `MLflowMonitor` - For experiment tracking with presigned URL generation

### RFT Multiturn Module
Manages infrastructure for reinforcement fine-tuning with multi-turn conversational tasks.

**Main Methods:**
- `setup()` - Deploy SAM stack and validate platform
- `start_training_environment()` - Start training environment
- `start_evaluation_environment()` - Start evaluation environment
- `get_logs()` - Retrieve environment logs
- `kill_task()` - Stop running task
- `cleanup()` - Clean up infrastructure resources
- `check_all_queues()` - Check message counts in all queues
- `flush_all_queues()` - Purge all messages from queues

**Key Classes:**
- `RFTMultiturnInfrastructure` - Main infrastructure management class
- `CustomEnvironment` - For creating custom reward environments

**Supported Platforms:**
- `LOCAL` - Local development environment
- `EC2` - Amazon EC2 instances
- `ECS` - Amazon ECS clusters

**Built-in Environments:**
- `VFEnvId.WORDLE` - Wordle game environment
- `VFEnvId.TERMINAL_BENCH` - Terminal benchmark environment

---
### Iterative Training

The Nova Forge SDK supports iterative fine-tuning of Nova models.

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

The Nova Forge SDK supports `dry_run` mode for the following functions: `train()`, `evaluate()`, and `batch_inference()`.

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

---
## Security Best Practices for SDK Users

### 1. IAM and Access Management

**Execution Roles**

- Use dedicated execution roles for SageMaker training jobs with minimal required permissions
- Avoid using admin roles - follow the principle of least privilege
- Regularly audit role permissions and remove unused policies

```python
# Good: Explicit execution role
runtime = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=2,
    execution_role="arn:aws:iam::123456789012:role/SageMakerNovaTrainingRole"
)

# Avoid: Using default role without validation
```

**Required Permissions**

The SDK requires specific IAM permissions. Review the [IAM](#iam-rolespolicies) section and:

* Grant only the minimum permissions needed for your use case
* Use condition statements to restrict resource access
* Regularly review and rotate access keys

### 2. Credential Management

**AWS Credentials**

* Never hardcode credentials in code or configuration files
* Use IAM roles instead of access keys when possible
* Rotate credentials regularly
* Use AWS Secrets Manager for application secrets
* Enable credential monitoring through AWS Config

**MLflow Integration**

* Secure MLflow tracking URIs with proper authentication
* Use encrypted connections to MLflow servers
* Implement access controls on experiment data
* Regularly audit MLflow access logs

### 3. Data Security and Privacy

**Training Data Protection**

- Encrypt data at rest in S3 using KMS keys
- Use S3 bucket policies to restrict access
- Validate data sources before processing

```python
# Ensure your S3 buckets have proper encryption and access controls
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime,
    data_s3_path="s3://secure-training-bucket/encrypted-data/",
    output_s3_path="s3://secure-output-bucket/results/"
)
```

### 4. Network Security

**VPC Configuration**

* Deploy in private subnets when possible
* Use VPC endpoints for AWS service access
* Implement security groups with minimal required ports
* Enable VPC Flow Logs for network monitoring

### 5. Secure Communication

- Always use HTTPS endpoints
- Never disable SSL certificate verification
- Keep TLS libraries updated

### 6. Input Validation

- Always validate user inputs before passing to SDK
- Sanitize data that will be stored or processed
- Check resource quotas before job submission
- Sanitize job names and resource identifiers


```python
# The SDK includes built-in validation
loader = JSONLDatasetLoader(question="input", answer="output")
loader.load("s3://your-bucket/training-data.jsonl")
# Always validate your data format
loader.validate(method=TrainingMethod.SFT_LORA, model=Model.NOVA_LITE_2)
```

### 7. Monitoring & Logging

- Enable CloudTrail for API audit logs
- Use CloudWatch for operational monitoring
- Never log sensitive data (tokens, credentials, PII)
- Monitor job logs through CloudWatch
- Set up alerts for suspicious activities

**Security Monitoring**
- Monitor failed authentication attempts
- Track unusual resource access patterns
- Log all model deployment activities

### 8. Deployment Security

**Bedrock Deployment**

- Use least privilege policies for Bedrock access
- Implement endpoint access controls
- Monitor model inference patterns
- Enable request/response logging when appropriate

### 9. Validation

The SDK includes built-in validation:

- IAM permission validation before job execution
- Input sanitization for user-provided parameters

Validation is enabled by default.