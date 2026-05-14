# Job Results
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
- JobResultObject. The instance of subclass of BaseJobResult such as SMTJEvaluationResult, SMHPEvaluationResult, BedrockEvaluationResult, SMTJTrainingResult, SMHPTrainingResult, or BedrockTrainingResult

**Example:**
```python
job_result = BaseJobResult.load('./my_job_result.json')
```

---

##### `enable_job_notifications()`
Enable email notifications for when a job reaches a terminal state (Completed, Failed, or Stopped).

**Signature:**
```python
def enable_job_notifications(
    self,
    emails: list[str],
    output_s3_path: Optional[str] = None,
    region: Optional[str] = "us-east-1",
    **platform_kwargs
) -> None
```

**Parameters:**
- `emails` (list[str]): List of email addresses to notify
- `output_s3_path` (Optional[str]): S3 path where job outputs are stored. 
  - Only required if the SDK cannot automatically extract it from the job result's `model_artifacts` attribute.
  - For most training jobs, this parameter is automatically populated and does not need to be provided explicitly.
- `region` (Optional[str]): AWS region for notification infrastructure (default: "us-east-1")
- `**platform_kwargs`: Platform-specific parameters:
  - **For SMTJ:**
    - `kms_key_id` (Optional[str]): Customer KMS key ID (not full ARN) for SNS topic encryption
  - **For SMHP:**
    - `namespace` (str): Kubernetes namespace where the PyTorchJob runs (e.g., "kubeflow", "default") (Required)
    - `kubectl_layer_arn` (str): ARN of the lambda-kubectl layer (Required)
    - `eks_cluster_arn` (Optional[str]): EKS cluster ARN (auto-detected if not provided)
    - `vpc_id` (Optional[str]): VPC ID (auto-detected if not provided)
    - `subnet_ids` (Optional[list[str]]): List of subnet IDs for Lambda (auto-detected if not provided)
    - `security_group_id` (Optional[str]): Security group ID for Lambda (auto-detected if not provided)
    - `polling_interval_minutes` (Optional[int]): How often to check job status in minutes (default: 5)
    - `kms_key_id` (Optional[str]): Customer KMS key ID (not full ARN) for SNS topic encryption

**Returns:**
- None

**Raises:**
- `ValueError`: If required parameters are missing or invalid
- `NotificationManagerInfraError`: If infrastructure setup fails

**How It Works:**
1. Creates AWS infrastructure (CloudFormation stack) if it doesn't exist:
   - DynamoDB table to store job notification configurations
   - SNS topic for email notifications
   - Lambda function to monitor job status
   - EventBridge rule (SMTJ) or scheduled rule (SMHP) to trigger Lambda
   - (SMHP only) VPC endpoints for DynamoDB and S3 if needed
2. Stores job configuration in DynamoDB (including namespace for SMHP)
3. Subscribes email addresses to SNS topic (users must confirm subscription)
4. Monitors job status and sends email when job completes, fails, is stopped, or becomes degraded (SMHP only)

**Email Confirmation:**
Users will receive a confirmation email from AWS SNS and must click the confirmation link before receiving job notifications.

**Examples:**

SMTJ (SageMaker Training Jobs):
```python
# Basic usage - output_s3_path is automatically extracted
result = customizer.train(job_name="my-job")
result.enable_job_notifications(
    emails=["user@example.com", "team@example.com"]
)

# With customer KMS encryption
result.enable_job_notifications(
    emails=["user@example.com"],
    kms_key_id="abc-123-def-456"  # Just the key ID, not full ARN
)

# With custom region
result.enable_job_notifications(
    emails=["user@example.com"],
    region="us-west-2"
)
```

SMHP (SageMaker HyperPod):
```python
# Basic usage (with auto-detection)
result = customizer.train(job_name="my-job")
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",  # Required
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1"  # Required
)

# With custom polling interval
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1",
    polling_interval_minutes=10  # Check every 10 minutes instead of default 5
)

# With explicit VPC configuration of the cluster where jobs are being monitored.
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1",
    eks_cluster_arn="arn:aws:eks:<region>:123456789012:cluster/my-cluster",
    vpc_id="vpc-12345",
    subnet_ids=["subnet-1", "subnet-2"],
    security_group_id="sg-12345"
)
```

**Important Notes:**
- For SMHP, requires deploying a kubectl Lambda layer from AWS Serverless Application Repository
- For SMHP, the user will need to manually grant the Lambda function access to your EKS cluster (access-entry).
    - Please refer to [`docs/user-guides/job_notifications.md`](../user-guides/job_notifications.md) for the commands to run to set this up. 
- See [`docs/user-guides/job_notifications.md`](../user-guides/job_notifications.md) for detailed setup instructions, troubleshooting, and advanced usage

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
- BedrockEvaluationResult

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
### IAM Role Creation SDK
This SDK provides utility functions for creating IAM roles with specific permissions for AWS Bedrock and SageMaker services.

**Methods**

#### `create_bedrock_execution_role()`
Creates an IAM role with permissions for Bedrock model creation and deployment.

**Signature:**
```python
def create_bedrock_execution_role(
    iam_client, 
    role_name: str, 
    bedrock_resource: str = "*", 
    s3_resource: str = "*"
) -> Dict
```

**Parameters:**
- `iam_client`: Boto3 IAM client
- `role_name` (str): Name of the IAM role to create
- `bedrock_resource` (Optional[str]): Specific Bedrock resource to restrict access. Defaults to "*" (all resources)
- `s3_resource` (Optional[str]): Specific S3 resource to restrict access. Defaults to "*" (all resources)

**Returns:**
- `Dict`: IAM role details

**Example:**
```python
import boto3
from amzn_nova_forge.iam import create_bedrock_execution_role

iam_client = boto3.client("iam")
create_bedrock_execution_role(iam_client, "role-name", "bedrock_resource", "s3_resource")
```

### `create_sagemaker_execution_role()`
Creates an IAM role with permissions for SageMaker model creation and deployment.

**Signature:**
```python
def create_sagemaker_execution_role(
    iam_client,
    role_name: str,
    s3_resource: str = "*",
    kms_resource: str = "*",
    ec2_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_metric_condition: Optional[Dict[str, Any]] = None,
    cloudwatch_logstream_resource: str = "*",
    cloudwatch_loggroup_resource: str = "*"
) -> Dict
```

**Parameters:**
- `iam_client`: Boto3 IAM client
- `role_name` (str): Name of the IAM role to create
- `s3_resource` (Optional[str]): Specific S3 resource to restrict access
- `kms_resource` (Optional[str]): Specific KMS resource to restrict access
- `ec2_condition` (Optional[Dict]): Conditional access for EC2 resources
- `cloudwatch_metric_condition` (Optional[Dict]): Conditional access for CloudWatch metrics
- `cloudwatch_logstream_resource` (Optional[str]): Specific CloudWatch log stream resource
- `cloudwatch_loggroup_resource` (Optional[str]): Specific CloudWatch log group resource

**Returns:**
- `Dict`: IAM role details

**Example:**
```python
import boto3
from amzn_nova_forge.iam import create_sagemaker_execution_role

iam_client = boto3.client("iam")
create_sagemaker_execution_role(
        iam_client,
        role_name="role-name",
        s3_resource="example-bucket",
        kms_resource="encryption-key",
        ec2_condition={
            "ArnLike": {
                "ec2:Vpc": "arn:aws:ec2:*:*:vpc/example"
            }
        },
        cloudwatch_metric_condition={
            "StringEquals": {
                "cloudwatch:namespace": ["example-namespace"]
            }
        },
        cloudwatch_loggroup_resource="example-loggroup",
        cloudwatch_logstream_resource="example-logstream"
    )
```
---
#### `ModelDeployResult`
Result of creating a Bedrock or Sagemaker model.

**Fields:**
- `model_arn` (str): The Bedrock custom model ARN or SageMaker model ARN.
- `model_name` (str): The model name.
- `escrow_uri` (str): S3 artifacts path used to create the model.
- `created_at` (datetime): UTC timestamp when the model was created.

**Properties:**
- `platform` (Optional[str]): Returns `"bedrock"` or `"sagemaker"` based on strict ARN format validation, or `None` for unrecognized ARNs.
- `status` (ModelStatus): Queries the model's current status via live API call. Returns `CREATING`, `ACTIVE`, `FAILED`, or `UNKNOWN`. Logs a warning if no AWS client is available.

**Class Methods:**
- `from_arn(model_arn, bedrock_client=None, sagemaker_client=None)` — Reconstruct from an existing model ARN. Detects platform from ARN format, calls the appropriate describe API, and recovers `escrow_uri` from tags. Works for both Bedrock and SageMaker ARNs.
- `load(file_path)` — Load from a JSON file saved by `dump()`. Automatically creates fresh AWS clients for status checking.

**Instance Methods:**
- `dump(file_path=None, file_name=None)` — Save to JSON file for later use with `load()`.

**Example:**
```python
# Create and persist
publish = customizer.create_custom_model(job_result=training_result)
publish.dump(file_path="./results/")

# Load and check status
loaded = ModelDeployResult.load("./results/my-model_deploy_result.json")
print(loaded.platform)  # "bedrock" or "sagemaker"
print(loaded.status)     # ModelStatus.ACTIVE

# Reconstruct from ARN
result = ModelDeployResult.from_arn(
  "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
)
```

#### `ModelStatus`
Platform-independent model status enum.

**Values:**
- `ModelStatus.CREATING` — Model is still being created (Bedrock only).
- `ModelStatus.ACTIVE` — Model is ready for deployment.
- `ModelStatus.FAILED` — Model creation failed.
- `ModelStatus.UNKNOWN` — Status could not be determined (no client, unrecognized ARN, or unexpected API response).
---

