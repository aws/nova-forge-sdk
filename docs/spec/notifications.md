# Job Notifications

The Nova Forge SDK provides automated email notifications for training jobs when they reach terminal states (Completed, Failed, or Stopped). This feature helps you monitor long-running jobs without constantly checking their status.

### Overview

Job notifications are managed through platform-specific notification managers that automatically set up and manage the required AWS infrastructure:

- **SMTJNotificationManager**: For SageMaker Training Jobs (SMTJ)
- **SMHPNotificationManager**: For SageMaker HyperPod (SMHP)

### How It Works

When you enable notifications for a job, the SDK automatically:

1. **Creates AWS Infrastructure** (if it doesn't exist):
   - CloudFormation stack with all required resources
   - DynamoDB table to store job notification configurations
   - SNS topic for email notifications
   - Lambda function to handle job state changes
   - EventBridge rule to monitor job status
   - IAM roles and policies with appropriate permissions

2. **Configures Job Monitoring**:
   - Stores job configuration in DynamoDB
   - Subscribes email addresses to SNS topic
   - Monitors job status via EventBridge

3. **Sends Notifications**:
   - Detects when job reaches terminal state
   - Validates output artifacts (for SMTJ, checks for manifest.json in output.tar.gz)
   - Sends email notification with job details and console link

### Using Job Notifications

The simplest way to enable notifications is through the job result object:

```python
from amzn_nova_forge import *

# Start a training job
customizer = NovaModelCustomizer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2),
    data_s3_path="s3://my-bucket/training-data/data.jsonl",
    output_s3_path="s3://my-bucket/output"
)

result = customizer.train(job_name="my-training-job")

# Enable notifications
result.enable_job_notifications(
    emails=["user@example.com", "team@example.com"],
    region="us-west-2", # Optional
    kms_key_id="1234abcd-12ab-34cd-56ef-1234567890ab", # Optional customer KMS key
    output_s3_path="s3://my-bucket/custom-output-path" # Optional output path
)
```
**Note:** Only provide `output_s3_path` if the 'JobResult' object doesn't have 'model_artifacts' (will be called out when you run the function).

### Email Confirmation

When you enable notifications:
1. Each email address receives a confirmation email from AWS SNS
2. Users must click the confirmation link in the email
3. After confirmation, they'll receive notifications for all jobs using that SNS topic
4. Confirmation is only needed once per email address per region

### Notification Content

Email notifications include:
- Job ID and platform (SMTJ/SMHP)
- Job status (Completed, Failed, or Stopped)
- Timestamp
- Link to AWS Console for the job
- For completed jobs: Validation status of output artifacts
- For failed jobs: Failure reason (if available)

### Infrastructure Details (SMTJ)

#### CloudFormation Stack

The notification infrastructure is managed as a CloudFormation stack:
- **Stack Name**: `NovaForgeSDK-SMTJ-JobNotifications`
- **Region**: Specified when enabling notifications (default: us-east-1)
- **Resources**: DynamoDB table, SNS topic, Lambda function, EventBridge rule, IAM roles

#### DynamoDB Table

Stores job notification configurations:
- **Table Name**: `NovaForgeSDK-SMTJ-JobNotifications`
- **Primary Key**: `job_id` (String)
- **Attributes**: `emails` (String Set), `output_s3_path` (String), `created_at` (String), `ttl` (Number)
- **TTL**: Automatically deletes entries after 30 days

#### SNS Topic

Manages email subscriptions:
- **Topic Name**: `NovaForgeSDK-SMTJ-Notifications`
- **Encryption**: Optional KMS encryption
- **Subscriptions**: Email protocol with confirmation required

#### Lambda Function

Handles job state change events:
- **Function Name**: `NovaForgeSDK-SMTJ-NotificationHandler`
- **Runtime**: Python 3.12
- **Timeout**: 180 seconds
- **Triggers**: EventBridge rule for SageMaker Training Job state changes

#### EventBridge Rule

Monitors job status:
- **Rule Name**: `NovaForgeSDK-SMTJ-Job-State-Change`
- **Event Pattern**: SageMaker Training Job State Change events
- **States Monitored**: Completed, Failed, Stopped

### Infrastructure Details (SMHP)

#### CloudFormation Stack

The notification infrastructure is managed as a CloudFormation stack:
- **Stack Name**: `NovaForgeSDK-SMHP-JobNotifications-{ClusterName}`
- **Region**: Specified when enabling notifications (default: us-east-1)
- **Resources**: DynamoDB table, SNS topic, Lambda function, EventBridge rule, IAM roles, VPC endpoints

#### DynamoDB Table

Stores job notification configurations:
- **Table Name**: `NovaForgeSDK-SMHP-JobNotifications-{ClusterName}`
- **Primary Key**: `job_id` (String)
- **Attributes**: `output_s3_path` (String), `namespace` (String), `ttl` (Number)
- **TTL**: Automatically deletes entries after 30 days
- **Point-in-Time Recovery**: Enabled

#### SNS Topic

Manages email subscriptions:
- **Topic Name**: `NovaForgeSDK-SMHP-Notifications-{ClusterName}`
- **Encryption**: Optional KMS encryption
- **Subscriptions**: Email protocol with confirmation required

#### Lambda Function

Handles job status polling:
- **Function Name**: `NovaForgeSDK-SMHP-NotificationHandler-{ClusterName}`
- **Runtime**: Python 3.12
- **Timeout**: 300 seconds
- **Memory**: 512 MB
- **VPC Configuration**: Deployed in VPC with access to EKS cluster
- **Layers**: kubectl layer for Kubernetes API access
- **Triggers**: EventBridge scheduled rule (default: every 5 minutes)

#### EventBridge Rule

Periodically checks job status:
- **Rule Name**: `NovaForgeSDK-SMHP-Job-Check-{ClusterName}`
- **Schedule**: Rate-based (default: every 5 minutes, configurable)
- **Target**: Lambda function for polling PyTorchJob status

#### VPC Endpoints

Enable private AWS service access for Lambda:
- **DynamoDB Gateway Endpoint**: `NovaForgeSDK-SMHP-DynamoDB-{ClusterName}`
- **SNS Interface Endpoint**: `NovaForgeSDK-SMHP-SNS-{ClusterName}`
- **S3 Gateway Endpoint**: `NovaForgeSDK-SMHP-S3-{ClusterName}`

### Limitations and Notes

1. **Email Confirmation**: Users must confirm their email subscription before receiving notifications.

2. **Region-Specific**: Notification infrastructure is created per region. Jobs in different regions require separate infrastructure.

3. **Stack Creation Restrictions**: For SMTJ, one notification stack is created per region. 
For SMHP, one notification stack is created per cluster per region.

4. **KMS Key Requirements**: If using KMS encryption:
   - Provide only the key ID, not the full ARN
   - The Lambda function automatically receives permissions to use the key
   - The key must be in the same region as the notification infrastructure

5. **Output Path Required**: The `output_s3_path` is required for manifest validation. The SDK will attempt to extract it from `model_artifacts` if not provided explicitly.

6. **Hard-coded CloudFormation Stack Names**: When the CF stack is created, it will have one of the following names: `NovaForgeSDK-SMTJ-JobNotifications` or `NovaForgeSDK-SMHP-JobNotifications-{HP-Cluster}`. 

### Troubleshooting

#### Notifications Not Received

1. **Check email confirmation**: Ensure you clicked the confirmation link in the AWS SNS email
2. **Check spam folder**: SNS emails may be filtered as spam
3. **Verify job status**: Notifications only sent for terminal states (Completed, Failed, Stopped)
4. **Check CloudWatch Logs**: View Lambda function logs for errors

#### Stack Creation Failures

If CloudFormation stack creation fails:
1. Check IAM permissions for CloudFormation, DynamoDB, SNS, Lambda, EventBridge, and IAM
2. Verify no resource name conflicts exist
3. Check CloudFormation console for detailed error messages

### API Reference

See the [BaseJobResult.enable_job_notifications()](#enable_job_notifications) method documentation for detailed parameter information.


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
