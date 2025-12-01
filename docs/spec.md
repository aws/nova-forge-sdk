# Nova Customization SDK - API Specification
## Table of Contents
1. [NovaModelCustomizer](#novamodelcustomizer)
2. [Runtime Managers](#runtime-managers)
3. [Dataset Loaders](#dataset-loaders)
4. [Job Results](#job-results)
5. [Enums and Configuration](#enums-and-configuration)
---
## NovaModelCustomizer
The main entrypoint class for customizing and training Nova models.
### Constructor
#### `__init__()`
Initializes a NovaModelCustomizer instance.
**Signature:**
```python
def __init__(
 self,
 model: Model,
 method: TrainingMethod,
 infra: RuntimeManager,
 data_s3_path: str,
 output_s3_path: Optional[str] = None,
 model_path: Optional[str] = None,
)
```
**Parameters:**
- `model` (Model): The Nova model to be trained (e.g., `Model.NOVA_MICRO`, `Model.NOVA_LITE`, `Model.NOVA_LITE_2`, `Model.NOVA_PRO`)
- `method` (TrainingMethod): The fine-tuning method (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.RFT`)
- `infra` (RuntimeManager): Runtime infrastructure manager (e.g., `SMTJRuntimeManager` or `SMHPRuntimeManager`)
- `data_s3_path` (str): S3 path to the training dataset
- `output_s3_path` (Optional[str]): S3 path for output artifacts. If not provided, will be auto-generated
- `model_path` (Optional[str]): S3 path for model path
- `validation_config` (Optional[Dict[str, bool]]): Optional dict to control validation.
**Raises:**
- `ValueError`: If region is unsupported or model is invalid
**Example:**
```python
from amzn_nova_customization_sdk import NovaModelCustomizer, Model, TrainingMethod
from amzn_nova_customization_sdk.manager import SMTJRuntimeManager
infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)
customizer = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=infra,
 data_s3_path="s3://my-bucket/training-data/",
 output_s3_path="s3://my-bucket/output/"
)
```
---
### Methods
#### `train()`
Generates the recipe YAML, configures runtime, and launches a training job.
**Signature:**
```python
def train(
 self,
 job_name: str,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 rft_lambda_arn: Optional[str] = None,
) -> TrainingResult
```
**Parameters:**
- `job_name` (str): User-defined name for the training job
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of configuration overrides. Example overrides below:
 - `max_epochs` (int): Maximum number of training epochs
 - `lr` (float): Learning rate
 - `warmup_steps` (int): Number of warmup steps
 - `loraplus_lr_ratio` (float): LoRA+ learning rate ratio
 - `global_batch_size` (int): Global batch size
 - `max_length` (int): Maximum sequence length
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only used for RFT training methods)
**Returns:**
- `TrainingResult`: Metadata object containing:
 - `job_id` (str): The training job identifier
 - `method` (TrainingMethod): The training method used
 - `started_time` (datetime): Job start timestamp
 - `model_artifacts` (ModelArtifacts): Paths to model checkpoints and outputs
   - `checkpoint_s3_path` (str, Optional): Path to the model checkpoint/trained model. 
   - `output_s3_path` (str): Path to the metrics and output tar file. 
   **Raises:**
- `Exception`: If job execution fails
- `ValueError`: If training method is not supported
**Example:**
```python
result = customizer.train(
 job_name="my-training-job",
 overrides={
 'max_epochs': 10,
 'lr': 5e-6,
 'warmup_steps': 20,
 'global_batch_size': 128
 }
)
print(f"Training job started: {result.job_id}")
print(f"Checkpoint path: {result.model_artifacts.checkpoint_s3_path}")
```
---
#### `evaluate()`
Generates the recipe YAML, configures runtime, and launches an evaluation job.

**Signature:**
```python
def evaluate(
 self,
 job_name: str,
 eval_task: EvaluationTask,
 model_path: Optional[str] = None,
 subtask: Optional[str] = None,
 data_s3_path: Optional[str] = None,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
) -> BaseJobResult
```
**Parameters:**
- `job_name` (str): User-defined name for the evaluation job
- `eval_task` (EvaluationTask): The evaluation task to be performed (e.g., `EvaluationTask.MMLU`)
  - The list of available tasks can be found here: [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-benchmark)
- `model_path` (Optional[str]): S3 path for model to evaluate
- `subtask` (Optional[str]): Subtask for evaluation (task-specific)
  - The list of available subtasks per task can be found here: [Subtasks](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-subtasks)
- `data_s3_path` (Optional[str]): S3 URI for the dataset
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of inference configuration overrides
  - `max_new_tokens` (int): Maximum tokens to generate
  - `top_k` (int): Top-k sampling parameter
  - `top_p` (float): Top-p (nucleus) sampling parameter
  - `temperature` (float): Temperature for sampling
- `processor` (Optional[Dict[str, Any]]): Optional, Bring Your Own Metrics/RFT lambda Configuration
- `rl_env` (Optional[Dict[str, Any]]): Optional, Bring your own reinforcement learning environment config
- 
**Returns:**
- `EvaluationResult(BaseJobResult)`: Metadata object (either `SMTJEvaluationResult` or `SMHPEvaluationResult`) containing:
  - `job_id` (str): The evaluation job identifier
  - `started_time` (datetime): Job start timestamp
  - `eval_output_path` (str): S3 path to evaluation results
  - `eval_task` (EvaluationTask): The Evaluation task

**Example:**

```python
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask

# General eval task (with overrides)
eval_result = customizer.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://my-bucket/checkpoints/my-model/",
    overrides={
        'max_new_tokens': 2048,
        'temperature': 0,
        'top_p': 1.0
    }
)
print(f"Evaluation job started: {eval_result.job_id}")

# BYOM eval task (by providing processor config)
byom_eval_result = customizer.evaluate(
    job_name='yuhag-eval-test-byom',
    eval_task=EvaluationTask.GEN_QA,
    data_s3_path="s3://905418167188-data/yuhag-dev/eval/gen_qa.jsonl",
    processor={
        "lambda_arn": "arn:aws:lambda:us-east-1:905418167188:function:yuhag-eval-simple-byom-lambda"
    }
)
```
---
#### `deploy()`
Creates a custom model and deploys it to Amazon Bedrock.
**Signature:**
```python
def deploy(
 self,
 model_artifact_path: str,
 deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
 pt_units: Optional[int] = None,
 endpoint_name: Optional[str] = None,
) -> DeploymentResult
```
**Note:** If DeployPlatform.BEDROCK_PT is selected, you must include a value for pt_units. 
**Parameters:**
- `model_artifact_path` (str): S3 path to the trained model, usually the escrow/checkpoint bucket. 
  - If you're using this after the 'train' function, use the checkpoint_s3_path for this variable. 
- `deploy_platform` (DeployPlatform): Platform to deploy the model to
 - `DeployPlatform.BEDROCK_OD`: Bedrock On-Demand
 - `DeployPlatform.BEDROCK_PT`: Bedrock Provisioned Throughput
- `pt_units` (Optional[int]): Number of Provisioned Throughput units (required only for Bedrock PT)
- `endpoint_name` (Optional[str]): Name of the deployed model's endpoint (auto-generated if not provided)

**Returns:**
- `DeploymentResult`: Contains:
 - `endpoint` (EndpointInfo): Endpoint information
 - `platform` (DeployPlatform): Deployment platform
 - `endpoint_name` (str): Endpoint name
 - `uri` (str): Model ARN
 - `model_artifact_path` (str): S3 path to artifacts
 - `created_at` (datetime): Deployment creation timestamp
**Raises:**
- `Exception`: When unable to successfully deploy the model
- `ValueError`: If platform is not supported
**Example:**
```python
deployment = customizer.deploy(
 model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
 deploy_platform=DeployPlatform.BEDROCK_OD,
 endpoint_name="my-custom-nova-model"
)
print(f"Model deployed: {deployment.endpoint.uri}")
print(f"Endpoint: {deployment.endpoint.endpoint_name}")
```
---
#### `batch_inference()`
Launches a batch inference job on a trained model.
**Signature:**
```python
def batch_inference(
 self,
 job_name: str,
 input_path: str,
 output_s3_path: str,
 model_path: Optional[str] = None,
 endpoint: Optional[EndpointInfo] = None,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
) -> BaseJobResult
```
**Parameters:**
- `job_name` (str): Name for the batch inference job
- `input_path` (str): S3 path to input data for inference
- `output_s3_path` (str): S3 path for inference outputs
- `model_path` (Optional[str]): S3 path to the model
- `endpoint` (Optional[EndpointInfo]): Endpoint information for SageMaker or Bedrock inference
- `recipe_path` (Optional[str]): Path for a YAML recipe file
- `overrides` (Optional[Dict[str, Any]]): Configuration overrides for inference
 - `max_new_tokens` (int): Maximum tokens to generate
 - `top_k` (int): Top-k sampling parameter
 - `top_p` (float): Top-p (nucleus) sampling parameter
 - `temperature` (float): Temperature for sampling
 - `top_logprobs` (int): Number of top log probabilities to return
**Returns:**
- `BaseJobResult`: Metadata object (`SMTJBatchInferenceResult`) containing:
 - `job_id` (str): Batch inference job identifier
 - `started_time` (datetime): Job start timestamp
 - `inference_output_path` (str): S3 path to inference results
**Example:**
```python
inference_result = customizer.batch_inference(
 job_name="batch-inference-job",
 input_path="s3://my-bucket/inference-input/",
 output_s3_path="s3://my-bucket/inference-output/",
 model_path="s3://my-bucket/trained-model/"
)
print(f"Batch inference started: {inference_result.job_id}")
```
In a separate notebook cell, you can run the following commands to get the job status and download a formatted result file when the jobs completes.
```python
inference_result.get_job_status() # Gets the job status.
inference_result.get("s3://my-bucket/save-location/file-name.jsonl") # Uploads a formatted inference_results.jsonl file to the given s3 location. 
```
---
#### `get_logs()`
Retrieves and displays CloudWatch logs for the current job.
**Signature:**
```python
def get_logs(
 self,
 limit: Optional[int] = None,
 start_from_head: bool = False
)
```
**Parameters:**
- `limit` (Optional[int]): Maximum number of log lines to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs; if False, start from the end
**Returns:**
- None (prints logs to console)
**Example:**
```python
# After starting a training job
customizer.train(job_name="my-job")
customizer.get_logs(limit=100, start_from_head=True)
```
---
#### `validate()`
Validates the specified training job without actually launching a job.

**Signature:**
```python
def validate(
 self,
 job_name: str,
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 rft_lambda_arn: Optional[str] = None,
) -> None:
```
**Parameters:**
- `job_name` (str): User-defined name for the training job
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of configuration overrides. Example overrides below:
 - `max_epochs` (int): Maximum number of training epochs
 - `lr` (float): Learning rate
 - `warmup_steps` (int): Number of warmup steps
 - `loraplus_lr_ratio` (float): LoRA+ learning rate ratio
 - `global_batch_size` (int): Global batch size
 - `max_length` (int): Maximum sequence length
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only used for RFT training methods)
**Returns:**
- None (throws an error if the job is known to be invalid)

**Example:**
```python

try:
    customizer.validate(
     job_name="my-training-job",
     overrides={
     'max_epochs': 10,
     'lr': 5e-6,
     'warmup_steps': 20,
     'global_batch_size': 128
     }
    )

    print("this is a valid job")
except ValueError as e:
    logger.info(f"Job configuration was invalid. Exception: {e}")
```
---
## Runtime Managers
Runtime managers handle the infrastructure for executing training and evaluation jobs.
### SMTJRuntimeManager
Manages SageMaker Training Jobs.
#### Constructor
**Signature:**
```python
def __init__(
 self,
 instance_type: str,
 instance_count: int
)
```
**Parameters:**
- `instance_type` (str): EC2 instance type (e.g., "ml.p5.48xlarge", "ml.p4d.24xlarge")
- `instance_count` (int): Number of instances to use
**Example:**
```python
from amzn_nova_customization_sdk.manager import SMTJRuntimeManager
infra = SMTJRuntimeManager(
 instance_type="ml.p5.48xlarge",
 instance_count=2
)
```
#### Properties
- `instance_type` (str): Returns the instance type
- `instance_count` (int): Returns the number of instances
#### Methods
##### `execute()`
Starts a SageMaker training job.
**Signature:**
```python
def execute(
 self,
 job_name: str,
 data_s3_path: Optional[str],
 output_s3_path: str,
 image_uri: str,
 recipe: str,
 input_s3_data_type: str = "Converse",
) -> str
```
**Returns:**
- `str`: Training job name/ID
##### `cleanup()`
Stops and cleans up a training job.
**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```
---
### SMHPRuntimeManager
Manages SageMaker HyperPod jobs.
#### Constructor
**Signature:**
```python
def __init__(
 self,
 instance_type: str,
 instance_count: int,
 cluster_name: str,
 namespace: str
)
```
**Parameters:**
- `instance_type` (str): EC2 instance type
- `instance_count` (int): Number of instances
- `cluster_name` (str): HyperPod cluster name
- `namespace` (str): Kubernetes namespace
**Example:**
```python
from amzn_nova_customization_sdk.manager import SMHPRuntimeManager
infra = SMHPRuntimeManager(
 instance_type="ml.p5.48xlarge",
 instance_count=4,
 cluster_name="my-hyperpod-cluster",
 namespace="default"
)
```
#### Properties
- `instance_type` (str): Returns the instance type
- `instance_count` (int): Returns the number of instances
#### Methods
##### `execute()`
Starts a SageMaker HyperPod job.
**Signature:**
```python
def execute(
 self,
 job_name: str,
 data_s3_path: Optional[str],
 output_s3_path: str,
 image_uri: str,
 recipe: str,
 input_s3_data_type: str = "Converse",
) -> str
```
**Returns:**
- `str`: HyperPod job ID
##### `cleanup()`
Cancels and cleans up a HyperPod job.
**Signature:**
```python
def cleanup(
 self,
 job_name: str
) -> None
```
---
## Dataset Loaders
Dataset loaders handle loading, transforming, and saving datasets in various formats.
### Base Class: DatasetLoader
Abstract base class for all dataset loaders.
#### Constructor
**Signature:**
```python
def __init__(
 self,
 **column_mappings
)
```
**Parameters:**
- `**column_mappings`: Keyword arguments mapping standard column names to dataset column names
 - Example: `question="input"` where "question" is the standard name and "input" is your column name
---
### JSONLDatasetLoader
Loads datasets from JSONL (JSON Lines) files.
#### Methods
##### `load()`
Loads dataset from a JSONL file (local or S3).
**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```
**Parameters:**
- `path` (str): Path to JSONL file (local path or S3 URI)
**Returns:**
- `DatasetLoader`: Self (for method chaining)
**Example:**
```python
from amzn_nova_customization_sdk.dataset import JSONLDatasetLoader
loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/data/training.jsonl")
```
---
### JSONDatasetLoader
Loads datasets from JSON files.
#### Methods
##### `load()`
Loads dataset from a JSON file (local or S3).
**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```
**Parameters:**
- `path` (str): Path to JSON file (local path or S3 URI)
**Returns:**
- `DatasetLoader`: Self (for method chaining)
**Example:**
```python
from amzn_nova_customization_sdk.dataset import JSONDatasetLoader
loader = JSONDatasetLoader()
loader.load("data/training.json")
```
---
### CSVDatasetLoader
Loads datasets from CSV files.
#### Methods
##### `load()`
Loads dataset from a CSV file.
**Signature:**
```python
def load(
 self,
 path: str
) -> "DatasetLoader"
```
**Parameters:**
- `path` (str): Path to CSV file (local path or S3 URI)
**Returns:**
- `DatasetLoader`: Self (for method chaining)
**Example:**
```python
from amzn_nova_customization_sdk.dataset import CSVDatasetLoader
loader = CSVDatasetLoader(question="user_query", answer="bot_response")
loader.load("data/conversations.csv")
```
---
### Common DatasetLoader Methods
These methods are available on all DatasetLoader subclasses.
#### `show()`
Displays the first n rows of the dataset.
**Signature:**
```python
def show(
 self,
 n: int = 10
) -> None
```
**Parameters:**
- `n` (int): Number of rows to display (default: 10)
**Example:**
```python
loader.show(5) # Show first 5 rows
```
---
#### `split_data()`
Splits dataset into train, validation, and test sets.
**Signature:**
```python
def split_data(
 self,
 train_ratio: float = 0.8,
 val_ratio: float = 0.1,
 test_ratio: float = 0.1,
 seed: int = 42,
) -> Tuple["DatasetLoader", "DatasetLoader", "DatasetLoader"]
```
**Parameters:**
- `train_ratio` (float): Proportion of data for training (default: 0.8)
- `val_ratio` (float): Proportion of data for validation (default: 0.1)
- `test_ratio` (float): Proportion of data for testing (default: 0.1)
- `seed` (int): Random seed for reproducibility (default: 42)
**Returns:**
- `Tuple[DatasetLoader, DatasetLoader, DatasetLoader]`: Three DatasetLoader objects (train, val, test)
**Raises:**
- `DataPrepError`: If ratios don't sum to 1.0 or dataset is empty
**Example:**
```python
train_loader, val_loader, test_loader = loader.split_data(
 train_ratio=0.7,
 val_ratio=0.2,
 test_ratio=0.1
)
```
---
#### `transform()`
Transforms dataset to the required format for a specific training method and model.
**Signature:**
```python
def transform(
 self,
 method: TrainingMethod,
 model: Model
) -> "DatasetLoader"
```
**Parameters:**
- `method` (TrainingMethod): The training method (e.g., `TrainingMethod.SFT_LORA`)
- `model` (Model): The Nova model version (e.g., `Model.NOVA_LITE`)
**Returns:**
- `DatasetLoader`: Self (for method chaining)
**Raises:**
- `ValueError`: If method/model combination is not supported
- `DataPrepError`: If transformation fails
**Example:**
```python
loader.transform(
 method=TrainingMethod.SFT_LORA,
 model=Model.NOVA_MICRO
)
```
---
#### `save_data()`
Saves the dataset to a local or S3 location.
**Signature:**
```python
def save_data(
 self,
 save_path: str
) -> str
```
**Parameters:**
- `save_path` (str): Path where to save the file (local or S3, must end in .json or .jsonl)
**Returns:**
- `str`: Path where the file was saved
**Raises:**
- `DataPrepError`: If save fails or format is unsupported
**Example:**
```python
# Save locally
loader.save_data("output/training_data.jsonl")
# Save to S3
loader.save_data("s3://my-bucket/data/training_data.jsonl")
```
---
## Job Results
Job result classes provide methods to check status and retrieve results from training, evaluation, and inference jobs.
### Base Classes
#### BaseJobResult
Abstract base class for all job results.

**Attributes:**
- `job_id` (str): Job identifier
- `started_time` (datetime): Job start timestamp

**Methods:**
##### `get_job_status()`
Gets the current status of the job.

**Signature:**
```python
def get_job_status(
 self
) -> tuple[JobStatus, str]
```

**Returns:**
- `tuple[JobStatus, str]`: A tuple of (status enum, raw status string)
 - `JobStatus.IN_PROGRESS`: Job is running
 - `JobStatus.COMPLETED`: Job completed successfully
 - `JobStatus.FAILED`: Job failed

**Example:**
```python
status, raw_status = result.get_job_status()
if status == JobStatus.COMPLETED:
 print("Job finished!")
```

##### `dump(file_path: Optional[str] = None)`
Save the job result to file_path path

**Signature:**
```python
def dump(
 self, 
 file_path: Optional[str] = None
) -> None
```

**Example:**
```python
result.dump()
# Result will be saved to ./{job_id}_{platform}.json under current dir
result.dump('/customized/path/customized_name.jsob')
# Result will be saved to /customized/path/customized_name.jsob
```

##### `load(file_path: str)`
Load the job result from file_path path

**Signature:**
```python
@classmethod
def load(
 cls, 
 file_path: str
) -> "BaseJobResult":
```

**Returns:**
- JobResultObject. The instance of subclass of BaseJobResult such as SMTJEvaluationResult

**Example:**
```python
job_result = BaseJobResult.load('./my_job_result.json')
```

---
### EvaluationResult(ABC)
Result object for SageMaker Training Job evaluation tasks.

**Attributes:**
- `job_id` (str): Job identifier
- `started_time` (datetime): Job start timestamp
- `eval_task` (EvaluationTask): Evaluation task performed
- `eval_output_path` (str): S3 path to evaluation results

#### Subclasses
- SMTJEvaluationResult
- SMHPEvaluationResult

#### Methods
##### `get()`
Downloads and returns evaluation results as a dictionary.

**Signature:**
```python
def get(
 self
) -> Dict
```

**Returns:**
- `Dict`: Evaluation results (empty dict if job not completed)

**Example:**
```python
eval_result = customizer.evaluate(...)
# Wait for job to complete
results = eval_result.get()
print(results)
```

---
##### `show()`
Prints evaluation results to console.

**Signature:**
```python
def show(
 self
) -> None
```

**Example:**
```python
eval_result.show()
```

---
##### `upload_tensorboard_results()`
Uploads TensorBoard results to S3.

**Signature:**
```python
def upload_tensorboard_results(
 self,
 tensorboard_s3_path: Optional[str] = None
) -> None
```

**Parameters:**
- `tensorboard_s3_path` (Optional[str]): Target S3 path (auto-generated if not provided)

**Example:**
```python
eval_result.upload_tensorboard_results(
 tensorboard_s3_path="s3://my-bucket/tensorboard/"
)
```

---
##### `clean()`
Cleans up local cached results.

**Signature:**
```python
def clean(
 self
) -> None
```

---
### SMTJBatchInferenceResult
Result object for batch inference jobs.
**Attributes:**
- `job_id` (str): Job identifier
- `started_time` (datetime): Job start timestamp
- `inference_output_path` (str): S3 path to inference outputs
#### Methods
##### `get()`
Downloads and returns inference results, optionally saving to S3.
**Signature:**
```python
def get(
 self,
 s3_path: Optional[str] = None
) -> Dict
```
**Parameters:**
- `s3_path` (Optional[str]): S3 path to save formatted results
**Returns:**
- `Dict`: Dictionary containing list of inference results
 - Each result has: `system`, `query`, `gold_response`, `inference_response`, `metadata`
**Example:**
```python
inference_result = customizer.batch_inference(...)
# Wait for job to complete
results = inference_result.get(s3_path="s3://my-bucket/formatted-results.jsonl")
```
---
##### `show()`
Prints inference results to console.
**Signature:**
```python
def show(
 self
) -> None
```
---
##### `clean()`
Cleans up local cached results.
**Signature:**
```python
def clean(
 self
) -> None
```
---
## Enums and Configuration
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
- `TrainingMethod.SFT_LORA`: Supervised Fine-Tuning with LoRA
- `TrainingMethod.SFT_FULLRANK`: Supervised Fine-Tuning (full rank)
- `TrainingMethod.RFT_LORA`: Reinforcement Fine-Tuning with LoRA
- `TrainingMethod.RFT`: Reinforcement Fine-Tuning
- `TrainingMethod.EVALUATION`: Evaluation only
---
### DeployPlatform Enum
Supported deployment platforms.
**Values:**
- `DeployPlatform.BEDROCK_OD`: Amazon Bedrock On-Demand
- `DeployPlatform.BEDROCK_PT`: Amazon Bedrock Provisioned Throughput
- `DeployPlatform.SAGEMAKER`: Amazon SageMaker (not yet implemented)
---
### EvaluationTask Enum
Supported evaluation tasks.
Common values include:
- `EvaluationTask.MMLU`: Massive Multitask Language Understanding
- `EvaluationTask.HELLASWAG`: Commonsense reasoning benchmark
- `EvaluationTask.TRUTHFULQA`: Truthfulness evaluation
- `EvaluationTask.GSM8K`: Grade school math problems
- And many more...
---
### Platform Enum
Infrastructure platforms.
**Values:**
- `Platform.SMTJ`: SageMaker Training Jobs
- `Platform.SMHP`: SageMaker HyperPod
---
### JobStatus Enum
Job execution status.
**Values:**
- `JobStatus.IN_PROGRESS`: Job is running
- `JobStatus.COMPLETED`: Job completed successfully
- `JobStatus.FAILED`: Job failed
---
## Complete Usage Example

```python
from amzn_nova_customization_sdk import NovaModelCustomizer
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod
from amzn_nova_customization_sdk.manager import SMTJRuntimeManager
from amzn_nova_customization_sdk.dataset import JSONLDatasetLoader
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask

# 1. Prepare dataset
loader = JSONLDatasetLoader()
loader.load("s3://my-bucket/raw-data.jsonl")
loader.transform(method=TrainingMethod.SFT_LORA, model=Model.NOVA_MICRO)
# Split into train/val/test
train, val, test = loader.split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
train.save_data("s3://my-bucket/train.jsonl")
val.save_data("s3://my-bucket/val.jsonl")
# 2. Set up infrastructure
infra = SMTJRuntimeManager(
    instance_type="ml.p5.48xlarge",
    instance_count=1
)
# 3. Initialize customizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    data_s3_path="s3://my-bucket/train.jsonl",
    output_s3_path="s3://my-bucket/output/"
)
# 4. Train model
training_result = customizer.train(
    job_name="my-training-job",
    overrides={
        'max_epochs': 5,
        'lr': 5e-6,
        'global_batch_size': 64
    }
)
# Monitor logs
customizer.get_logs(limit=50)
# 5. Evaluate model
eval_result = customizer.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    model_path=training_result.model_artifacts.checkpoint_s3_path
)
# Wait for completion and get results
results = eval_result.get()
eval_result.show()
# 6. Deploy model
deployment = customizer.deploy(
    model_artifact_path=training_result.model_artifacts.checkpoint_s3_path,
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-model"
)
print(f"Model deployed at: {deployment.endpoint.uri}")
```
---
## Error Handling
All SDK functions may raise exceptions. It's recommended to wrap calls in try-except blocks:
```python
try:
 result = customizer.train(job_name="my-job")
except ValueError as e:
 print(f"Configuration error: {e}")
except Exception as e:
 print(f"Training failed: {e}")
```
Common exceptions:
- `ValueError`: Invalid parameters or configuration
- `DataPrepError`: Dataset preparation errors
- `Exception`: General job execution or AWS API errors
---
## Best Practices
1. **Always validate your data** using `loader.show()` before training
2. **Use overrides sparingly** - start with defaults and tune as needed
3. **Monitor logs** during training using `get_logs()`
4. **Check job status** before calling `.get()` on results
5. **Clean up resources** when done to avoid unnecessary costs
6. **Use descriptive job names** to help track and organize your experiments
7. **Save results incrementally** during long-running jobs
8. **Test with small datasets** before scaling up to full training
---
## Additional Resources
- AWS Documentation: [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- AWS Documentation: [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)
- SDK GitHub Repository: Check for updates and examples
- Support: Use AWS Support for technical assistance---
_Last Updated: November 21, 2025_

