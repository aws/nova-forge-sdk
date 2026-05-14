# Enums and Configuration
### Model Enum
Supported Nova models with their configurations.
**Values:**
- `Model.NOVA_MICRO`: Amazon Nova Micro (Version 1)
  - `model_type`: "amazon.nova-micro-v1:0:128k"
  - `model_path`: "nova-micro/prod"
  - `version`: Version.ONE
- `Model.NOVA_LITE`: Amazon Nova Lite (Version 1)
  - `model_type`: "amazon.nova-lite-v1:0:300k"
  - `model_path`: "nova-lite/prod"
  - `version`: Version.ONE
- `Model.NOVA_LITE_2`: Amazon Nova Lite (Version 2)
  - `model_type`: "amazon.nova-2-lite-v1:0:256k"
  - `model_path`: "nova-lite-2/prod"
  - `version`: Version.TWO
- `Model.NOVA_PRO`: Amazon Nova Pro (Version 1)
  - `model_type`: "amazon.nova-pro-v1:0:300k"
  - `model_path`: "nova-pro/prod"
  - `version`: Version.ONE

**Methods:**
##### `from_model_type()`
Gets Model enum from model type string.

**Signature:**
```python
@classmethod
def from_model_type(
 cls,
 model_type: str
) -> "Model"
```

**Example:**
```python
model = Model.from_model_type("amazon.nova-micro-v1:0:128k")
```
---
### TrainingMethod Enum
Supported training methods.

**Values:**
- `TrainingMethod.CPT`: Continued Pre-Training
- `TrainingMethod.DPO_LORA`: Direct Preference Optimization with LoRA
- `TrainingMethod.DPO_FULL`: Direct Preference Optimization (full rank)
- `TrainingMethod.SFT_LORA`: Supervised Fine-Tuning with LoRA
- `TrainingMethod.SFT_FULL`: Supervised Fine-Tuning (full rank)
- `TrainingMethod.RFT_LORA`: Reinforcement Fine-Tuning with LoRA
- `TrainingMethod.RFT_FULL`: Full reinforcement Fine-Tuning
- `TrainingMethod.EVALUATION`: Evaluation only
---
### DeployPlatform Enum
Supported deployment platforms.

**Values:**
- `DeployPlatform.BEDROCK_OD`: Amazon Bedrock On-Demand
- `DeployPlatform.BEDROCK_PT`: Amazon Bedrock Provisioned Throughput
- `DeployPlatform.SAGEMAKER`: Amazon SageMaker
---
### DeploymentMode Enum
Deployment behavior when an endpoint with the same name already exists.

**Values:**
- `DeploymentMode.FAIL_IF_EXISTS`: Raise an error if endpoint already exists (safest, default)
- `DeploymentMode.UPDATE_IF_EXISTS`: Try in-place update only, fail if not supported (PT only)

**Note:** Only `FAIL_IF_EXISTS` and `UPDATE_IF_EXISTS` modes are currently supported. 
`UPDATE_IF_EXISTS` is only applicable for Bedrock Provisioned Throughput (PT) deployments.

---
### EvaluationTask Enum
Supported evaluation tasks.
Common values include:
- `EvaluationTask.MMLU`: Massive Multitask Language Understanding
- `EvaluationTask.GPQA`: General Physics Question Answering
- `EvaluationTask.MATH`: Mathematical Problem Solving
- `EvaluationTask.GEN_QA`: Custom Dataset Evaluation
- The full list of available tasks can be found here: [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-benchmark)
---
### Platform Enum
Infrastructure platforms.

**Values:**
- `Platform.SMTJ`: SageMaker Training Jobs
- `Platform.SMHP`: SageMaker HyperPod
- `Platform.BEDROCK`: Amazon Bedrock
---
### JobStatus Enum
Job execution status.

**Values:**
- `JobStatus.IN_PROGRESS`: Job is running
- `JobStatus.COMPLETED`: Job completed successfully
- `JobStatus.FAILED`: Job failed

---

