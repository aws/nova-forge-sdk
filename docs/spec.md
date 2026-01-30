# Nova Customization SDK - API Specification

## Table of Contents
1. [NovaModelCustomizer](#novamodelcustomizer)
2. [Runtime Managers](#runtime-managers)
3. [Dataset Loaders](#dataset-loaders)
4. [Job Results](#job-results)
5. [Monitoring](#monitoring)
6. [Enums and Configuration](#enums-and-configuration)
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
 data_s3_path: Optional[str] = None,
 output_s3_path: Optional[str] = None,
 model_path: Optional[str] = None,
 validation_config: Optional[Dict[str, bool]] = None,
 generated_recipe_dir: Optional[str] = None,
 mlflow_monitor: Optional[MLflowMonitor] = None,
 deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
 data_mixing_enabled: bool = False,
)
```
**Parameters:**
- `model` (Model): The Nova model to be trained (e.g., `Model.NOVA_MICRO`, `Model.NOVA_LITE`, `Model.NOVA_LITE_2`, `Model.NOVA_PRO`)
- `method` (TrainingMethod): The fine-tuning method (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.RFT`)
- `infra` (RuntimeManager): Runtime infrastructure manager (e.g., `SMTJRuntimeManager` or `SMHPRuntimeManager`)
- `data_s3_path` (Optional[str]): S3 path to the training dataset
- `output_s3_path` (Optional[str]): S3 path for output artifacts. If not provided, will be auto-generated
- `model_path` (Optional[str]): S3 path for model path
- `validation_config` (Optional[Dict[str, bool]]): Optional dict to control validation. Defaults to `{'iam': True, 'infra': True}`
- `generated_recipe_dir` (Optional[str]): Optional local path to save the generated recipe
- `mlflow_monitor` (Optional[MLflowMonitor]): Optional MLflow monitoring configuration for experiment tracking
- `deployment_mode` (DeploymentMode): Behavior when deploying to existing endpoint name. Options: FAIL_IF_EXISTS (default), UPDATE_IF_EXISTS
- `data_mixing_enabled` (bool): Enable data mixing feature for CPT and SFT training on SageMaker HyperPod. Default is False
  - **Note:** The `data_mixing_enabled` parameter must be set to `True` during initialization to use data mixing features. 
  - **Note:** Datamixing is only supported for CPT, SFT_LORA, and SFT_FULL methods on SageMaker HyperPod (SMHP). 

**Raises:**
- `ValueError`: If region is unsupported or model is invalid

**Example:**
```python
from amzn_nova_customization_sdk import *

infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

# Without MLflow monitoring
customizer = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=infra,
 data_s3_path="s3://my-bucket/training-data/",
 output_s3_path="s3://my-bucket/output/"
)

# With MLflow monitoring
mlflow_monitor = MLflowMonitor(
 tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
 experiment_name="nova-customization",
 run_name="sft-run-1"
)

customizer_with_mlflow = NovaModelCustomizer(
 model=Model.NOVA_MICRO,
 method=TrainingMethod.SFT_LORA,
 infra=infra,
 data_s3_path="s3://my-bucket/training-data/",
 output_s3_path="s3://my-bucket/output/",
 mlflow_monitor=mlflow_monitor
)
```
---
### Methods

#### `get_data_mixing_config()`
Get the current data mixing configuration.

**Signature:**
```python
def get_data_mixing_config(
 self
) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Dictionary containing the data mixing configuration

**Example:**
```python
config = customizer.get_data_mixing_config()
print(config)
# Output: {'customer_data_percent': 50, 'nova_code_percent': 30, 'nova_general_percent': 70}
```

---

#### `set_data_mixing_config()`
Set the data mixing configuration.

**Signature:**
```python
def set_data_mixing_config(
 self,
 config: Dict[str, Any]
) -> None
```

**Parameters:**
- `config` (Dict[str, Any]): Dictionary containing the data mixing configuration
  - `customer_data_percent` (int/float): Percentage of customer data (0-100)
  - `nova_code_percent` (int/float): Percentage of Nova code data (0-100)
  - `nova_general_percent` (int/float): Percentage of Nova general data (0-100)
  - Nova percentages must sum to 100%

**Raises:**
- `ValueError`: If data mixing is not enabled or configuration is invalid

**Example:**
```python
# Must initialize with data_mixing_enabled=True
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=SMHPRuntimeManager(...),
    data_s3_path="s3://bucket/data.jsonl",
    data_mixing_enabled=True
)

# Set data mixing configuration
customizer.set_data_mixing_config({
    "customer_data_percent": 50,
    "nova_code_percent": 30,
    "nova_general_percent": 70
})
```

---

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
 validation_data_s3_path: Optional[str] = None,
 dry_run: Optional[bool] = False       
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
  - A full list of available overrides can be found via the [Nova Customization public documentation](https://docs.aws.amazon.com/nova/latest/userguide/customize-fine-tune-sagemaker.html) or by referencing the training recipes [here](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-recipes.html). 
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only used for RFT training methods)
- `validation_data_s3_path` (Optional[str]): Validation S3 path, only applicable for CPT (but is still optional for CPT)
- `dry_run` (Optional[bool]): Actually starts a job if False, otherwise just performs validation.

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
 processor: Optional[Dict[str, Any]] = None,
 rl_env: Optional[Dict[str, Any]] = None,
 dry_run: Optional[bool] = False,
 job_result: Optional[TrainingResult] = None
) -> EvaluationResult | None
```

**Parameters:**
- `job_name` (str): User-defined name for the evaluation job
- `eval_task` (EvaluationTask): The evaluation task to be performed (e.g., `EvaluationTask.MMLU`)
  - The list of available tasks can be found here: [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-benchmark)
- `model_path` (Optional[str]): S3 path for model to evaluate. If not provided, will attempt to extract from `job_result` or the customizer's most recent training job.
- `data_s3_path` (Optional[str]): S3 URI for the dataset. Only required for BYOD (Bring Your Own Data) evaluation tasks.
- `subtask` (Optional[str]): Subtask for evaluation (task-specific)
  - The list of available subtasks per task can be found here: [Subtasks](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-evaluation.html#nova-model-evaluation-subtasks)
- `recipe_path` (Optional[str]): Path for a YAML recipe file (both S3 and local paths are accepted)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of inference configuration overrides
  - `max_new_tokens` (int): Maximum tokens to generate
  - `top_k` (int): Top-k sampling parameter
  - `top_p` (float): Top-p (nucleus) sampling parameter
  - `temperature` (float): Temperature for sampling
- `processor` (Optional[Dict[str, Any]]): Optional, Bring Your Own Metrics/RFT lambda Configuration
- `rl_env` (Optional[Dict[str, Any]]): Optional, Bring your own reinforcement learning environment config
- `dry_run` (Optional[bool]): Actually starts a job if False, otherwise just performs validation.
- `job_result` (Optional[TrainingResult]): Optional TrainingResult object to extract checkpoint path from. If provided and `model_path` is None, will automatically extract the checkpoint path from the training job's output and validate platform compatibility.

**Returns:**
- `EvaluationResult(BaseJobResult)`: Metadata object (either `SMTJEvaluationResult` or `SMHPEvaluationResult`) containing:
  - `job_id` (str): The evaluation job identifier
  - `started_time` (datetime): Job start timestamp
  - `eval_output_path` (str): S3 path to evaluation results
  - `eval_task` (EvaluationTask): The Evaluation task
- Returns `None` if `dry_run=True`

**Example:**

```python
from amzn_nova_customization_sdk.recipe import *

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

Deployment behavior when endpoint already exists is controlled by the `deployment_mode`
parameter set during NovaModelCustomizer initialization:
- FAIL_IF_EXISTS: Raise error (default, safest)
- UPDATE_IF_EXISTS: Try in-place update, fail if not supported (PT only)

**Signature:**
```python
def deploy(
 self,
 model_artifact_path: Optional[str] = None,
 deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
 pt_units: Optional[int] = None,
 endpoint_name: Optional[str] = None,
 job_result: Optional[TrainingResult] = None,
 bedrock_execution_role_name: str = BEDROCK_EXECUTION_ROLE_NAME
) -> DeploymentResult
```

* **Note:** If DeployPlatform.BEDROCK_PT is selected, you must include a value for pt_units. 

* **Note:** If `model_artifact_path` is provided, we will NOT attempt to resolve `model_artifact_path` from `job_result` or the enclosing `NovaModelCustomizer` object.

**Parameters:**
- `model_artifact_path` (Optional[str]): S3 path to the trained model checkpoint. If not provided, will attempt to extract from job_result or the `job_id` field of the Customizer.
- `deploy_platform` (DeployPlatform): Platform to deploy the model to
 - `DeployPlatform.BEDROCK_OD`: Bedrock On-Demand
 - `DeployPlatform.BEDROCK_PT`: Bedrock Provisioned Throughput
- `pt_units` (Optional[int]): Number of Provisioned Throughput units (required only for Bedrock PT)
- `endpoint_name` (Optional[str]): Name of the deployed model's endpoint (auto-generated if not provided)
- `job_result` (Optional[TrainingResult]): Training job result object to use for extracting checkpoint path and validating job completion. Also used to retrieve job_id if it's not provided.
- `bedrock_execution_role_name`:  Optional IAM execution role name for Bedrock, defaults to BedrockDeployModelExecutionRole. If this role does not exist, it will be created.

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
from amzn_nova_customization_sdk.model import *

deployment = customizer.deploy(
 model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
 deploy_platform=DeployPlatform.BEDROCK_OD,
 endpoint_name="my-custom-nova-model"
)
print(f"Model deployed: {deployment.endpoint.uri}")
print(f"Endpoint: {deployment.endpoint.endpoint_name}")
print(f"Status: {deployment.status}")
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
 recipe_path: Optional[str] = None,
 overrides: Optional[Dict[str, Any]] = None,
 dry_run: Optional[bool] = False
) -> InferenceResult
```
**Parameters:**
- `job_name` (str): Name for the batch inference job
- `input_path` (str): S3 path to input data for inference
- `output_s3_path` (str): S3 path for inference outputs
- `model_path` (Optional[str]): S3 path to the model
- `recipe_path` (Optional[str]): Path for a YAML recipe file
- `overrides` (Optional[Dict[str, Any]]): Configuration overrides for inference
 - `max_new_tokens` (int): Maximum tokens to generate
 - `top_k` (int): Top-k sampling parameter
 - `top_p` (float): Top-p (nucleus) sampling parameter
 - `temperature` (float): Temperature for sampling
 - `top_logprobs` (int): Number of top log probabilities to return
- `dry_run` (Optional[bool]): Actually starts a job if False, otherwise just performs validation.

**Returns:**
- `InferenceResult`: Metadata object (`SMTJBatchInferenceResult`) containing:
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
 start_from_head: bool = False,
 end_time: Optional[str] = None
)
```

**Parameters:**
- `limit` (Optional[int]): Maximum number of log lines to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs; if False, start from the end
- `end_time` (Optional[str]): Optionally specify an end time for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
# After starting a training job
customizer.train(job_name="my-job")
customizer.get_logs(limit=100, start_from_head=True)
```
---
## Runtime Managers
Runtime managers handle the infrastructure for executing training and evaluation jobs,
leveraging the `JobConfig` dataclass to do so:
```python
@dataclass
class JobConfig:
    job_name: str
    image_uri: str
    recipe_path: str
    output_s3_path: Optional[str] = None
    data_s3_path: Optional[str] = None
    input_s3_data_type: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
```
* The specific instance types that can be used with the runtime managers (SMTJ, SMHP) can be found in `docs/instance_type_spec.md`. This file also defines which instance types can be used with a specific model and method.
### SMTJRuntimeManager
Manages SageMaker Training Jobs.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    instance_type: str,
    instance_count: int,
    execution_role: Optional[str] = None,
    kms_key_id: Optional[str] = None,
    encrypt_inter_container_traffic: bool = False,
    subnets: Optional[list[str]] = None,
    security_group_ids: Optional[list[str]] = None,
)
```

**Parameters:**
- `instance_type` (str): EC2 instance type (e.g., "ml.p5.48xlarge", "ml.p4d.24xlarge")
- `instance_count` (int): Number of instances to use
- `execution_role` (Optional[str]): The execution role for the training job
- `kms_key_id` (Optional[str]): Optional KMS Key Id to use in S3 Bucket encryption, training jobs and deployments.
- `encrypt_inter_container_traffic` (bool): Boolean that determines whether to encrypt inter-container traffic. Default value is False.
- `subnets` (Optional[list[str]]): Optional list of strings representing subnets. Default value is None.
- `security_group_ids` (Optional[list[str]]): Optional list of strings representing security group IDs. Default value is None.

**Example:**
```python
from amzn_nova_customization_sdk.manager import *
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
 job_config: JobConfig
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
 namespace: str,
 kms_key_id: Optional[str] = None,
)
```

**Parameters:**
- `instance_type` (str): EC2 instance type
- `instance_count` (int): Number of instances
- `cluster_name` (str): HyperPod cluster name
- `namespace` (str): Kubernetes namespace
- `kms_key_id` (Optional[str]): Optional KMS Key Id to use in S3 Bucket encryption

**Example:**
```python
from amzn_nova_customization_sdk.manager import *
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
 job_config=JobConfig
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
#### Column Mappings
If you are transforming a plain JSON, JSONL, or CSV file from a generic format (e.g. 'input/output') to another format (e.g. Converse for SFT), you need to provide "column mappings" to connect your generic column/field name to the expected ones in the transformation function.

For example, if your plain dataset has "input" and "output" columns, and you want to transform it for SFT (which requrires "question" and "answer"), you would provide the following:
```python
loader = JSONDatasetLoader(
    question="input",
    answer="output"
)
```
Below is a list of accepted column mapping parameters for transformations. 
* SFT: `question`, `answer`
  * Optional: `system`, [image/video required options]: `image_format`/`video_format`, `s3_uri`, `bucket_owner`
  * 2.0: `reasoning_text`, `tools`/`toolsConfig`*
* RFT: `question`, `reference_answer`
  * Optional: `system`, `id`, `tools`*
* Eval: `query`, `response`
  * Optional: `images`, `metadata`
* CPT: `text`

Additional Notes: 
* If you're providing multimodal data in a generic format, you need to provide ALL three of the following fields:
  * `image_format` OR `video_format` + `s3_uri`, `bucket_owner`
* *`tools/toolsConfig` (SFT 2.0) and `tools` (RFT) parameters can *only* be provided when transforming from OpenAI Messages format to Converse or OpenAI. A generic format *cannot* be provided for this transformation to work.

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
from amzn_nova_customization_sdk.dataset import *
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
from amzn_nova_customization_sdk.dataset import *
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
from amzn_nova_customization_sdk.dataset import *
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
Transforms dataset to the required format for a specific training method and model. Currently the following transformations are supported:
* Q/A-formatted CSV/JSON/JSONL to SFT 1.0, SFT 2.0 (without reasoningContent, Tools), RFT, Eval, CPT
* OpenAI Messages format to SFT 1.0 and SFT 2.0 (with Tools)

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
#### `validate()`
Validates dataset when given the user's intended training method and model. 

**Signature:**
```python
def validate(
 self,
 method: TrainingMethod,
 model: Model,
 eval_task: EvaluationTask (Optional)
) -> None
```
**Parameters:**
- `method` (TrainingMethod): The training method (e.g., `TrainingMethod.SFT_LORA`)
- `model` (Model): The Nova model version (e.g., `Model.NOVA_LITE`)
- `eval_task` (EvaluationTask): The evaluation task (e.g., `EvaluationTask.GEN_QA`)

**Returns:**
- None

**Raises:**
- `ValueError`: If method/model combination is not supported or validation is unsuccessful.

**Example:**
```python
loader.validate(
 method=TrainingMethod.SFT_LORA,
 model=Model.NOVA_MICRO
)
```
If you're validating a BYOD Evaluation dataset, you need to provide another parameter, `eval_task` to the `validate` function. For example:
```
loader.validate(
    method=TrainingMethod.EVALUATION, 
    model=Model.NOVA_LITE_2, 
    eval_task=EvaluationTask.GEN_QA
)

>> Validation succeeded for 22 samples on an Evaluation BYOD dataset
```
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

##### `dump(file_path: Optional[str] = None, file_name: Optional[str] = None)`
Save the job result to file_path path

**Signature:**
```python
def dump(
 self, 
 file_path: Optional[str] = None,
 file_name: Optional[str] = None
) -> Path
```

**Parameters:**
- `file_path` (Optional[str]): Directory path to save the result. Saves to current directory if not provided
- `file_name` (Optional[str]): The file name of the result. Default to `<job_id>_<platform>.json` if not provided

**Returns:**
- `Path`: The full result file path

**Example:**
```python
result.dump()
# Result will be saved to ./{job_id}_{platform}.json under current dir
result.dump(file_path='/customized/path', file_name='customized_name.json')
# Result will be saved to /customized/path/customized_name.json
```

##### `load(file_path: str)`
Load the job result from the `file_path` path

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
### EvaluationResult (ABC)
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
## Monitoring

### MLflowMonitor

MLflow monitoring configuration for Nova model training. This class provides experiment tracking capabilities through MLflow integration.

**MLflow Integration Features:**
- Automatic logging of training metrics
- Model artifact and checkpoint tracking
- Hyperparameter recording
- Support for SageMaker MLflow tracking servers
- Custom MLflow tracking server support (with proper network configuration)


#### Constructor

**Signature:**
```python
def __init__(
 self,
 tracking_uri: Optional[str] = None,
 experiment_name: Optional[str] = None,
 run_name: Optional[str] = None,
)
```

**Parameters:**
- `tracking_uri` (Optional[str]): MLflow tracking server URI or SageMaker MLflow app ARN. If not provided, attempts to use a default SageMaker MLflow tracking server if one exists
- `experiment_name` (Optional[str]): Name of the MLflow experiment. If not provided, will use the job name
- `run_name` (Optional[str]): Name of the MLflow run. If not provided, will be auto-generated

**Raises:**
- `ValueError`: If MLflow configuration validation fails

**Example:**
```python
from amzn_nova_customization_sdk.monitor import *

# With explicit tracking URI
monitor = MLflowMonitor(
    tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# With default tracking URI (if available)
monitor = MLflowMonitor(
    experiment_name="nova-customization",
    run_name="sft-run-1"
)

# Use with NovaModelCustomizer
customizer = NovaModelCustomizer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.SFT_LORA,
    infra=runtime_manager,
    data_s3_path="s3://bucket/data",
    mlflow_monitor=monitor
)
```

#### Methods

##### `to_dict()`

Converts MLflow configuration to dictionary format for use in recipe overrides.

**Signature:**
```python
def to_dict(
 self
) -> dict
```

**Returns:**
- `dict`: Dictionary with mlflow_* keys for recipe configuration. Returns empty dict if no tracking URI is available

**Example:**
```python
monitor = MLflowMonitor(
 tracking_uri="arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
 experiment_name="nova-customization"
)

config_dict = monitor.to_dict()
# Returns: {
#   "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456:mlflow-app/app-xxx",
#   "mlflow_experiment_name": "nova-customization"
# }
```

#### MLflow Integration Notes

When MLflow monitoring is enabled:
1. Training metrics will be automatically logged to the specified MLflow tracking server
2. Model artifacts and checkpoints will be tracked in MLflow
3. Hyperparameters and configuration will be recorded as MLflow parameters
4. You can view experiment results in the MLflow UI

The MLflow integration supports:
- SageMaker MLflow tracking servers
- Custom MLflow tracking servers (with appropriate network configuration)
- Automatic experiment and run creation
- Metric logging during training
- Artifact tracking

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
- `DeployPlatform.SAGEMAKER`: Amazon SageMaker (not yet implemented)
---
### DeploymentMode Enum
Deployment behavior when an endpoint with the same name already exists.

**Values:**
- `DeploymentMode.FAIL_IF_EXISTS`: Raise an error if endpoint already exists (safest, default)
- `DeploymentMode.UPDATE_IF_EXISTS`: Try in-place update only, fail if not supported (PT only)

**Note:** Only `FAIL_IF_EXISTS` and `UPDATE_IF_EXISTS` modes are currently supported. `UPDATE_IF_EXISTS` is only applicable for Bedrock Provisioned Throughput (PT) deployments.

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
---
### JobStatus Enum
Job execution status.

**Values:**
- `JobStatus.IN_PROGRESS`: Job is running
- `JobStatus.COMPLETED`: Job completed successfully
- `JobStatus.FAILED`: Job failed

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
- Support: Use AWS Support for technical assistance 
---
_Last Updated: January 23, 2026_
