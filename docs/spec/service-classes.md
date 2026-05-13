# Service Classes

The modular service classes are the recommended API for Nova model customization. Each class handles a single concern — training, evaluation, deployment, or inference — and can be used independently.

All service classes accept an optional `ForgeConfig` dataclass for shared configuration (KMS keys, output paths, caching, etc.).

### ForgeConfig

Shared configuration dataclass for all service classes.

**Signature:**
```python
@dataclass
class ForgeConfig:
    kms_key_id: Optional[str] = None
    output_s3_path: Optional[str] = None
    generated_recipe_dir: Optional[str] = None
    validation_config: Optional[ValidationConfig] = None
    image_uri: Optional[str] = None
    mlflow_monitor: Optional[MLflowMonitor] = None
    enable_job_caching: bool = False
    job_cache_dir: str = "~/.nova-forge/cache"
    job_caching_config: Optional[JobCachingConfig] = None
```

**Parameters:**
- `kms_key_id` (Optional[str]): KMS key ID for S3 encryption
- `output_s3_path` (Optional[str]): S3 path for output artifacts. Auto-generated if not provided
- `generated_recipe_dir` (Optional[str]): Local path to save generated recipe files
- `validation_config` (Optional[ValidationConfig]): Controls pre-flight validation. Fields: `iam` (bool), `infra` (bool), `recipe` (bool) — all default to True
- `image_uri` (Optional[str]): Custom container image URI override
- `mlflow_monitor` (Optional[MLflowMonitor]): MLflow monitoring configuration (SageMaker only)
- `enable_job_caching` (bool): Enable caching of completed job results for reuse. Default: False
- `job_cache_dir` (str): Directory for cached job results. Default: `~/.nova-forge/cache`
- `job_caching_config` (Optional[JobCachingConfig]): Advanced caching configuration. Fields: `include_core` (bool), `include_recipe` (bool), `include_infra` (bool), `include_params` (List[str]), `exclude_params` (List[str]), `allowed_statuses` (List[JobStatus])

**Example:**
```python
from amzn_nova_forge.core import ForgeConfig, ValidationConfig
from amzn_nova_forge.monitor import MLflowMonitor

config = ForgeConfig(
    output_s3_path="s3://my-bucket/output",
    kms_key_id="my-kms-key-id",
    validation_config=ValidationConfig(iam=True, infra=True, recipe=True),
    mlflow_monitor=MLflowMonitor(
        tracking_uri="arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-xxx",
        experiment_name="nova-customization"
    ),
    enable_job_caching=True
)
```
---

### ForgeTrainer

Handles training job configuration and execution for Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    model: Model,
    method: TrainingMethod,
    infra: RuntimeManager,
    training_data_s3_path: Optional[str] = None,
    model_s3_path: Optional[str] = None,
    data_mixing_enabled: bool = False,
    holdout_data_s3_path: Optional[str] = None,
    val_check_interval: Optional[int] = None,
    config: Optional[ForgeConfig] = None,
    region: Optional[str] = None,
    is_multimodal: Optional[bool] = None,
    hub_content_version: Optional[str] = None,
    enable_batch_sample_tracing: bool = False,
)
```

**Parameters:**
- `model` (Model): The Nova model to train (e.g., `Model.NOVA_MICRO`, `Model.NOVA_LITE_2`)
- `method` (TrainingMethod): The fine-tuning method (e.g., `TrainingMethod.SFT_LORA`, `TrainingMethod.RFT`)
- `infra` (RuntimeManager): Runtime infrastructure manager (e.g., `SMTJRuntimeManager`, `SMHPRuntimeManager`, `BedrockRuntimeManager`)
- `training_data_s3_path` (Optional[str]): S3 path to the training dataset
- `model_s3_path` (Optional[str]): S3 path for the base or previously trained model
- `data_mixing_enabled` (bool): Enable data mixing for CPT and SFT training on SMHP, and SFT text-only on Nova Lite 2 on SMTJServerless. Default: False
- `holdout_data_s3_path` (Optional[str]): S3 path to holdout/validation data (optional, used for CPT and SFT on SMTJ/SMTJServerless/SMHP, or any method on Bedrock)
- `val_check_interval` (Optional[int]): How often (in training steps) to run validation. Defaults to 2500 if omitted. Only used when `holdout_data_s3_path` is provided.
- `config` (Optional[ForgeConfig]): Shared configuration. If not provided, defaults are used
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `is_multimodal` (Optional[bool]): Explicitly set multimodal mode when `data_mixing_enabled=True`. If None, auto-detects from data
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version
- `enable_batch_sample_tracing` (bool): Activate per-step batch hashing during training, which enables `trace_batch()` post-training. Supported platform/method combinations are validated at construction time. Default: False

**Raises:**
- `ValueError`: If `enable_batch_sample_tracing=True` is used with an unsupported platform or training method

**Example:**
```python
from amzn_nova_forge.trainer import ForgeTrainer
from amzn_nova_forge.core import ForgeConfig
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model, TrainingMethod

infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

trainer = ForgeTrainer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://my-bucket/training-data/data.jsonl",
    config=ForgeConfig(output_s3_path="s3://my-bucket/output")
)
```
---

#### Methods

##### `train()`
Generates the recipe YAML, configures the runtime, and launches a training job.

**Signature:**
```python
def train(
    self,
    job_name: str,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    rft_lambda_arn: Optional[str] = None,
    dry_run: bool = False,
    rft_multiturn_infra=None,
) -> Optional[TrainingResult]
```

**Parameters:**
- `job_name` (str): User-defined name for the training job
- `recipe_path` (Optional[str]): Path for a YAML recipe file (S3 or local)
- `overrides` (Optional[Dict[str, Any]]): Dictionary of configuration overrides (e.g., `max_epochs`, `lr`, `warmup_steps`, `global_batch_size`)
- `rft_lambda_arn` (Optional[str]): Rewards Lambda ARN (only for RFT methods). Takes priority over `rft_lambda_arn` on the RuntimeManager
- `dry_run` (bool): If True, performs validation only without starting a job. Default: False
- `rft_multiturn_infra`: Optional RFTMultiturnInfrastructure for RFT multiturn training

**Returns:**
- `TrainingResult`: Metadata object containing `job_id`, `method`, `started_time`, `model_artifacts`, and `model_type`. Returns `None` if `dry_run=True`

**Raises:**
- `Exception`: If job execution fails
- `ValueError`: If training method is not supported

**Example:**
```python
result = trainer.train(
    job_name="my-training-job",
    overrides={
        "max_epochs": 10,
        "lr": 5e-6,
        "warmup_steps": 20,
        "global_batch_size": 128
    }
)
print(f"Training job started: {result.job_id}")
print(f"Checkpoint path: {result.model_artifacts.checkpoint_s3_path}")
```
---

##### `get_logs()`
Retrieves and displays CloudWatch logs for a training job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[TrainingResult]): Job result to retrieve logs for. If not provided, uses `job_id`
- `job_id` (Optional[str]): Job identifier. Used if `job_result` is not provided
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines to retrieve
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
trainer.get_logs(job_result=result, limit=100, start_from_head=True)
```
---

##### `trace_batch()`
Extracts the lines from your input training data that were used in a specific training step's batch. Useful for diagnosing gradient spikes or training anomalies — given a step number, it matches the container's batch hash logs against your source data and writes the matched lines to an output file.

The training job must have been launched with `enable_batch_sample_tracing=True` so that batch hash logs are written during training. Supported platform/method combinations are validated at `ForgeTrainer` construction time. If you create a new `ForgeTrainer` instance to trace a previously-launched job, the flag is not strictly required on the new instance — but a warning will be emitted.

**Signature:**
```python
def trace_batch(
    self,
    training_result: TrainingResult,
    step: int,
    output_path: str | None = None,
    cache_dir: str = "~/.nova-forge/batch_trace_cache",
) -> Path | None
```

**Parameters:**
- `training_result` (TrainingResult): Result from a completed training job. The method extracts `training_result.model_artifacts.output_s3_path` and `training_result.job_id` to locate the batch hash logs at `{output_s3_path}/{job_id}/batch_tracing/`.
- `step` (int): Training step number to investigate (must match the step numbers in the container's batch hash logs).
- `output_path` (Optional[str]): Path for the output file containing matched lines. Default: `step_<N>_samples.jsonl` in the current working directory.
- `cache_dir` (str): Directory for caching downloaded files and fingerprint indices. Default: `~/.nova-forge/batch_trace_cache`. The cache stores downloaded S3 files (training data, hash logs) and a CSV fingerprint index. For large datasets the cache may grow to match the source data size. The cache is keyed by S3 path — if you replace the file at an existing S3 URI, delete the cache directory to avoid stale matches.

**Returns:**
- `Path | None`: Path to the output JSONL file containing the matched lines (verbatim copies from your source data, sorted by line number). Returns `None` if either the step had no logged batch data (step out of range or job still running) or the step's batch contained no samples from your file.

**Raises:**
- `ValueError`: If `training_data_s3_path` or `training_result.model_artifacts.output_s3_path` is not available
- `BatchTraceError`: If batch tracing encounters an unrecoverable error (e.g., missing log files, AWS auth failure)
  Import: `from amzn_nova_forge.trainer.utils import BatchTraceError`

**Example:**
```python
trainer = ForgeTrainer(
    model=Model.NOVA_LITE_2,
    method=TrainingMethod.CPT,
    infra=infra,
    training_data_s3_path="s3://my-bucket/data.jsonl",
    enable_batch_sample_tracing=True,
)

result = trainer.train(job_name="my-cpt-job")

# After job completes, investigate step 42.
# Output writes to ./step_42_samples.jsonl by default.
matched_file = trainer.trace_batch(result, step=42)
if matched_file:
    print(f"Matched lines written to: {matched_file}")

# Explicit output path:
matched_file = trainer.trace_batch(result, step=42, output_path="/tmp/flagged.jsonl")
```
---

### ForgeEvaluator

Handles evaluation job configuration and execution for Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    model: Model,
    infra: RuntimeManager,
    data_s3_path: Optional[str] = None,
    config: Optional[ForgeConfig] = None,
    region: Optional[str] = None,
    hub_content_version: Optional[str] = None,
)
```

**Parameters:**
- `model` (Model): The Nova model to evaluate
- `infra` (RuntimeManager): Runtime infrastructure manager
- `data_s3_path` (Optional[str]): S3 path to evaluation data (required for BYOD evaluation tasks)
- `config` (Optional[ForgeConfig]): Shared configuration
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version

**Example:**
```python
from amzn_nova_forge.evaluator import ForgeEvaluator
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model

infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

evaluator = ForgeEvaluator(
    model=Model.NOVA_MICRO,
    infra=infra,
    data_s3_path="s3://my-bucket/eval-data/data.jsonl"
)
```
---

#### Methods

##### `evaluate()`
Generates the recipe YAML, configures the runtime, and launches an evaluation job.

**Signature:**
```python
def evaluate(
    self,
    job_name: str,
    eval_task: EvaluationTask,
    model_path: Optional[str] = None,
    task_config: Optional[EvalTaskConfig] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    job_result: Optional[TrainingResult] = None,
    rft_multiturn_infra=None,
) -> Optional[EvaluationResult]
```

**Parameters:**
- `job_name` (str): User-defined name for the evaluation job
- `eval_task` (EvaluationTask): The evaluation task (e.g., `EvaluationTask.MMLU`)
- `model_path` (Optional[str]): S3 path to the model to evaluate. If not provided, extracted from `job_result`
- `task_config` (Optional[EvalTaskConfig]): Task-specific configuration. Fields: `subtask`, `processor`, `rl_env`, `override_data_s3_path`
- `recipe_path` (Optional[str]): Path for a YAML recipe file (S3 or local)
- `overrides` (Optional[Dict[str, Any]]): Inference configuration overrides (e.g., `max_new_tokens`, `temperature`, `top_p`)
- `dry_run` (bool): If True, performs validation only. Default: False
- `job_result` (Optional[TrainingResult]): Training result to extract checkpoint path from
- `rft_multiturn_infra`: Optional RFTMultiturnInfrastructure for RFT evaluation

**Returns:**
- `EvaluationResult`: Metadata object containing `job_id`, `started_time`, `eval_output_path`, and `eval_task`. Returns `None` if `dry_run=True`

**Example:**
```python
from amzn_nova_forge.core import EvaluationTask

eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    model_path="s3://my-bucket/checkpoints/my-model",
    overrides={
        "max_new_tokens": 2048,
        "temperature": 0,
        "top_p": 1.0
    }
)
print(f"Evaluation job started: {eval_result.job_id}")

# Chain from training result
eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    job_result=training_result
)
```
---

##### `get_logs()`
Retrieves and displays CloudWatch logs for an evaluation job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[EvaluationResult]): Job result to retrieve logs for
- `job_id` (Optional[str]): Job identifier
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
evaluator.get_logs(job_result=eval_result, limit=50)
```
---

### ForgeDeployer

Handles model deployment to Amazon Bedrock and SageMaker endpoints.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    region: str,
    model: Model,
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
    config: Optional[ForgeConfig] = None,
    method: Optional[TrainingMethod] = None,
)
```

**Parameters:**
- `region` (str): AWS region for deployment
- `model` (Model): The Nova model being deployed
- `deployment_mode` (DeploymentMode): Behavior when endpoint already exists. Default: `FAIL_IF_EXISTS`
- `config` (Optional[ForgeConfig]): Shared configuration
- `method` (Optional[TrainingMethod]): Training method used (needed for SageMaker deployment image selection)

**Example:**
```python
from amzn_nova_forge.deployer import ForgeDeployer
from amzn_nova_forge.model.model_enums import Model, DeploymentMode

deployer = ForgeDeployer(
    region="us-east-1",
    model=Model.NOVA_MICRO,
    deployment_mode=DeploymentMode.FAIL_IF_EXISTS
)
```
---

#### Methods

##### `deploy()`
Creates a custom model and deploys it to Amazon Bedrock or SageMaker in a single step.

**Signature:**
```python
def deploy(
    self,
    model_artifact_path: str,
    deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
    endpoint_name: Optional[str] = None,
    unit_count: int = 1,
    execution_role_name: Optional[str] = None,
    sagemaker_instance_type: str = "ml.p5.48xlarge",
    sagemaker_environment: Optional[SageMakerEndpointEnvironment] = None,
    skip_model_reuse: bool = False,
) -> DeploymentResult
```

**Parameters:**
- `model_artifact_path` (str): S3 path to the trained model checkpoint
- `deploy_platform` (DeployPlatform): Platform to deploy to (`BEDROCK_OD`, `BEDROCK_PT`, or `SAGEMAKER`). Default: `BEDROCK_OD`
- `endpoint_name` (Optional[str]): Name of the endpoint (auto-generated if not provided)
- `unit_count` (int): Number of PT units (Bedrock PT) or instances (SageMaker). Default: 1
- `execution_role_name` (Optional[str]): IAM role name. If omitted, the SDK creates a default role
- `sagemaker_instance_type` (str): Instance type for SageMaker deployment. Default: `"ml.p5.48xlarge"`
- `sagemaker_environment` (Optional[SageMakerEndpointEnvironment]): SageMaker endpoint environment config. Fields:
  - `CONTEXT_LENGTH` (int, default: 4000), `MAX_CONCURRENCY` (int, default: 1)
  - Optional generation defaults: `DEFAULT_TEMPERATURE` (0–2), `DEFAULT_TOP_P` (1e-10–1), `DEFAULT_TOP_K` (-1 to disable, or ≥1), `DEFAULT_MAX_NEW_TOKENS` (≥1), `DEFAULT_LOGPROBS` (1–20)
  - Optional speculative decoding: `SPECULATIVE_DECODING_METHOD` (`"eagle3"` or `"suffix"`), `DISABLE_SPECULATIVE_DECODING` (`"true"` or `"false"`), `NUM_SPECULATIVE_TOKENS` (1–10), `SUFFIX_DECODING_MAX_TREE_DEPTH`, `SUFFIX_DECODING_MAX_CACHED_REQUESTS`, `SUFFIX_DECODING_MAX_SPEC_FACTOR`, `SUFFIX_DECODING_MIN_TOKEN_PROB`
  - Optional memory/quantization: `KV_CACHE_DTYPE` (`"fp8"`), `QUANTIZATION_DTYPE` (`"fp8"`)
- `skip_model_reuse` (bool): If True, always create a new model. Default: False

**Returns:**
- `DeploymentResult`: Contains `endpoint` (EndpointInfo), `platform`, `endpoint_name`, `uri`, `model_artifact_path`, and `created_at`

**Raises:**
- `Exception`: When unable to deploy the model
- `ValueError`: If platform is not supported

**Example:**
```python
deployment = deployer.deploy(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/",
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-custom-nova-model"
)
print(f"Model deployed: {deployment.endpoint.uri}")
```
---

##### `create_custom_model()`
Creates a Bedrock custom model from S3 artifacts without deploying to an endpoint.

**Signature:**
```python
def create_custom_model(
    self,
    model_artifact_path: str,
    endpoint_name: Optional[str] = None,
    execution_role_name: Optional[str] = None,
    tags: Optional[List[Dict[str, str]]] = None,
    skip_model_reuse: bool = False,
) -> ModelDeployResult
```

**Parameters:**
- `model_artifact_path` (str): S3 path to trained model checkpoint
- `endpoint_name` (Optional[str]): Optional name prefix for the model name
- `execution_role_name` (Optional[str]): IAM role name for Bedrock
- `tags` (Optional[List[Dict[str, str]]]): Optional list of `{"key": str, "value": str}` dicts for tracking
- `skip_model_reuse` (bool): If True, always create a new model. Default: False

**Returns:**
- `ModelDeployResult`: Contains `model_arn`, `model_name`, `escrow_uri`, and `created_at`

**Example:**
```python
publish_result = deployer.create_custom_model(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
print(f"Model ARN: {publish_result.model_arn}")
publish_result.dump(file_path="./results/")
```
---

##### `deploy_to_bedrock()`
Deploys a published Bedrock custom model to an endpoint.

**Signature:**
```python
def deploy_to_bedrock(
    self,
    model_deploy_result: Optional[ModelDeployResult] = None,
    model_arn: Optional[str] = None,
    deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
    pt_units: Optional[int] = None,
    endpoint_name: Optional[str] = None,
) -> DeploymentResult
```

**Parameters:**
- `model_deploy_result` (Optional[ModelDeployResult]): Result from `create_custom_model()`. Cannot be combined with `model_arn`
- `model_arn` (Optional[str]): Direct model ARN. Cannot be combined with `model_deploy_result`
- `deploy_platform` (DeployPlatform): `BEDROCK_OD` (default) or `BEDROCK_PT`
- `pt_units` (Optional[int]): Number of PT units (required for `BEDROCK_PT`)
- `endpoint_name` (Optional[str]): Endpoint name (auto-generated if not provided)

**Returns:**
- `DeploymentResult`: Contains `endpoint`, `created_at`, and `model_publish`

**Raises:**
- `ValueError`: When both `model_deploy_result` and `model_arn` are provided, or when no model ARN is available
- `RuntimeError`: When deployment creation fails

**Example:**
```python
# Two-step deploy: create model, then deploy
publish_result = deployer.create_custom_model(
    model_artifact_path="s3://escrow-bucket/my-model-artifacts/"
)
deployment = deployer.deploy_to_bedrock(
    model_deploy_result=publish_result,
    endpoint_name="my-endpoint"
)

# Or deploy from an existing model ARN
deployment = deployer.deploy_to_bedrock(
    model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
)
```
---

##### `find_published_model()`
Finds an existing published model by platform and escrow path to enable model reuse.

**Signature:**
```python
def find_published_model(
    self,
    platform: str,
    escrow_path: str,
    skip_model_reuse: bool = False,
) -> Optional[str]
```

**Parameters:**
- `platform` (str): Target platform (`"bedrock"` or `"sagemaker"`)
- `escrow_path` (str): S3 path of the model artifacts
- `skip_model_reuse` (bool): If True, always returns None (skips lookup). Default: False

**Returns:**
- `Optional[str]`: Existing model ARN if found, otherwise None

**Example:**
```python
existing_arn = deployer.find_published_model(
    platform="bedrock",
    escrow_path="s3://escrow-bucket/my-model-artifacts/"
)
if existing_arn:
    print(f"Reusing existing model: {existing_arn}")
```
---

##### `get_status()`
Gets the deployment status for a DeploymentResult.

**Signature:**
```python
def get_status(self, result: DeploymentResult) -> JobStatus
```

**Parameters:**
- `result` (DeploymentResult): The deployment result to check

**Returns:**
- `JobStatus`: Current status (`IN_PROGRESS`, `COMPLETED`, or `FAILED`)

---

##### `get_status_by_arn()`
Gets the deployment status by endpoint ARN and platform.

**Signature:**
```python
def get_status_by_arn(
    self,
    endpoint_arn: str,
    platform: DeployPlatform,
) -> Optional[JobStatus]
```

**Parameters:**
- `endpoint_arn` (str): The endpoint ARN to check
- `platform` (DeployPlatform): The deployment platform

**Returns:**
- `Optional[JobStatus]`: Current status, or None if status cannot be determined

---

##### `get_logs()`
Retrieves and displays logs for a deployment.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    endpoint_arn=None,
    platform=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[DeploymentResult]): Deployment result to retrieve logs for
- `endpoint_arn` (Optional[str]): Endpoint ARN (used if `job_result` is not provided)
- `platform` (Optional[DeployPlatform]): Deployment platform (used with `endpoint_arn`)

**Returns:**
- None (prints logs to console)

---

### ForgeInference

Handles single and batch inference on trained Nova models.

#### Constructor

**Signature:**
```python
def __init__(
    self,
    region: Optional[str] = None,
    model: Optional[Model] = None,
    infra: Optional[RuntimeManager] = None,
    config: Optional[ForgeConfig] = None,
    method: Optional[TrainingMethod] = None,
    hub_content_version: Optional[str] = None,
)
```

**Parameters:**
- `region` (Optional[str]): AWS region. Auto-detected if not provided
- `model` (Optional[Model]): The Nova model (required for batch inference)
- `infra` (Optional[RuntimeManager]): Runtime infrastructure manager (required for batch inference)
- `config` (Optional[ForgeConfig]): Shared configuration
- `method` (Optional[TrainingMethod]): Training method (used for batch inference recipe generation)
- `hub_content_version` (Optional[str]): Version of the hub content to retrieve from SageMaker Hub. If None, uses the latest version

**Example:**
```python
from amzn_nova_forge.inference import ForgeInference

# For single inference (minimal setup)
inference = ForgeInference(region="us-east-1")

# For batch inference
inference = ForgeInference(
    region="us-east-1",
    model=Model.NOVA_MICRO,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=1),
    method=TrainingMethod.SFT_LORA
)
```
---

#### Methods

##### `invoke()`
Invokes a single inference on a deployed model endpoint.

**Signature:**
```python
def invoke(
    self,
    endpoint_arn: str,
    request_body: Dict[str, Any],
) -> Any
```

**Parameters:**
- `endpoint_arn` (str): Endpoint ARN to invoke
- `request_body` (Dict[str, Any]): Inference request body

**Returns:**
- `Any`: Inference response

**Example:**
```python
response = inference.invoke(
    endpoint_arn="arn:aws:bedrock:us-east-1:123456789012:endpoint/my-endpoint",
    request_body={
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "max_tokens": 100,
        "stream": False
    }
)
```
---

##### `invoke_batch()`
Launches a batch inference job on a trained model.

**Signature:**
```python
def invoke_batch(
    self,
    job_name: str,
    input_path: str,
    output_s3_path: str,
    model_path: Optional[str] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    job_result: Optional[TrainingResult] = None,
) -> Optional[InferenceResult]
```

**Parameters:**
- `job_name` (str): Name for the batch inference job
- `input_path` (str): S3 path to input data
- `output_s3_path` (str): S3 path for inference outputs
- `model_path` (Optional[str]): S3 path to the model checkpoint
- `recipe_path` (Optional[str]): Path for a YAML recipe file
- `overrides` (Optional[Dict[str, Any]]): Inference configuration overrides (e.g., `max_new_tokens`, `temperature`, `top_p`)
- `dry_run` (bool): If True, performs validation only. Default: False
- `job_result` (Optional[TrainingResult]): Training result to extract checkpoint path from

**Returns:**
- `InferenceResult`: Metadata object containing `job_id`, `started_time`, and `inference_output_path`. Returns `None` if `dry_run=True`

**Example:**
```python
inference_result = inference.invoke_batch(
    job_name="batch-inference-job",
    input_path="s3://my-bucket/inference-input",
    output_s3_path="s3://my-bucket/inference-output",
    model_path="s3://my-bucket/trained-model"
)
print(f"Batch inference started: {inference_result.job_id}")
```
---

##### `get_logs()`
Retrieves and displays CloudWatch logs for an inference job.

**Signature:**
```python
def get_logs(
    self,
    job_result=None,
    job_id=None,
    started_time=None,
    limit=None,
    start_from_head: bool = False,
    end_time=None,
) -> None
```

**Parameters:**
- `job_result` (Optional[InferenceResult]): Job result to retrieve logs for
- `job_id` (Optional[str]): Job identifier
- `started_time` (Optional[datetime]): Job start time to filter logs
- `limit` (Optional[int]): Maximum number of log lines
- `start_from_head` (bool): If True, start from the beginning of logs. Default: False
- `end_time` (Optional[int]): End time in epoch milliseconds for searching a log time range

**Returns:**
- None (prints logs to console)

**Example:**
```python
inference.get_logs(job_result=inference_result, limit=100)
```
---

### End-to-End Example (Service Classes)

```python
from amzn_nova_forge.trainer import ForgeTrainer
from amzn_nova_forge.evaluator import ForgeEvaluator
from amzn_nova_forge.deployer import ForgeDeployer
from amzn_nova_forge.inference import ForgeInference
from amzn_nova_forge.core import ForgeConfig
from amzn_nova_forge.manager import SMTJRuntimeManager
from amzn_nova_forge.model.model_enums import Model, TrainingMethod, DeployPlatform
from amzn_nova_forge.core import EvaluationTask

# Shared configuration
config = ForgeConfig(
    output_s3_path="s3://my-bucket/output",
    enable_job_caching=True
)
infra = SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2)

# 1. Train
trainer = ForgeTrainer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=infra,
    training_data_s3_path="s3://my-bucket/data.jsonl",
    config=config
)
train_result = trainer.train(job_name="my-training-job")

# 2. Evaluate
evaluator = ForgeEvaluator(model=Model.NOVA_MICRO, infra=infra, config=config)
eval_result = evaluator.evaluate(
    job_name="my-eval-job",
    eval_task=EvaluationTask.MMLU,
    job_result=train_result
)

# 3. Deploy
deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_MICRO)
deployment = deployer.deploy(
    model_artifact_path=train_result.model_artifacts.checkpoint_s3_path,
    deploy_platform=DeployPlatform.BEDROCK_OD,
    endpoint_name="my-nova-endpoint"
)

# 4. Inference
inference_client = ForgeInference(region="us-east-1")
response = inference_client.invoke(
    endpoint_arn=deployment.endpoint.uri,
    request_body={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
```

---
