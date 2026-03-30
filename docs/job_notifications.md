# Job Notifications

## Overview

The Nova Forge SDK provides automated email notifications for training and evaluation jobs. When enabled, you'll receive emails when your jobs reach terminal states (Completed, Failed, or Stopped). The SDK automatically sets up and manages all required AWS infrastructure.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Platform-Specific Configuration](#platform-specific-configuration)
  - [SMTJ (SageMaker Training Jobs)](#smtj-sagemaker-training-jobs)
  - [SMHP (SageMaker HyperPod)](#smhp-sagemaker-hyperpod)
- [Setup Requirements](#setup-requirements)
- [Manual CloudFormation Deployment](#manual-cloudformation-deployment)
- [Advanced Usage](#advanced-usage)
- [Notification Content](#notification-content)
- [SMHP Pod Health Monitoring](#smhp-pod-health-monitoring)
- [Limitations and Considerations](#limitations-and-considerations)
- [Troubleshooting](#troubleshooting)


## Features

- **Automatic Infrastructure Setup**: CloudFormation, DynamoDB, SNS, Lambda, and EventBridge are configured automatically. 
    - A user can still take the CloudFormation templates and manually deploy them in the account as well.
- **Email Notifications**: Get notified when jobs complete, fail, or are stopped
- **Artifact Validation**: For successful jobs, validates that output artifacts (manifest.json) exist
- **Optional KMS Encryption**: Encrypt SNS topics with your own KMS key
- **Platform Support**: Works with both SMTJ (SageMaker Training Jobs) and SMHP (SageMaker HyperPod)
- **SMHP Pod Health Monitoring**: Detects when a master pod enters a CrashLoopBackOff loop and sends a notification.

## Quick Start

### Basic Usage

```python
from amzn_nova_forge import *

# Start a training job
customizer = NovaModelCustomizer(
    model=Model.NOVA_MICRO,
    method=TrainingMethod.SFT_LORA,
    infra=SMTJRuntimeManager(instance_type="ml.p5.48xlarge", instance_count=2),
    data_s3_path="s3://my-bucket/data.jsonl",
    output_s3_path="s3://my-bucket/output/"
)

result = customizer.train(job_name="my-training-job")

# Enable notifications
result.enable_job_notifications(
    emails=["user@example.com", "team@example.com"]
)
```

### With KMS Encryption

```python
result.enable_job_notifications(
    emails=["user@example.com"],
    kms_key_id="1234abcd-12ab-34cd-56ef-1234567890ab"  # Just the key ID, not full ARN
)
```

## Platform-Specific Configuration

### SMTJ (SageMaker Training Jobs)

SMTJ notifications require minimal configuration:

```python
result.enable_job_notifications(
    emails=["user@example.com"],
    kms_key_id="1234abcd-12ab-34cd-56ef-1234567890ab"  # Optional
)
```

**Infrastructure Created:**
- CloudFormation Stack: `NovaForgeSDK-SMTJ-JobNotifications`
- DynamoDB Table: Stores job notification configurations
- SNS Topic: Sends email notifications
- Lambda Function: Monitors job state changes via EventBridge
- EventBridge Rule: Triggers Lambda on SageMaker job state changes

### SMHP (SageMaker HyperPod)

SMHP notifications require additional configuration for Kubernetes access:

```python
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",  # REQUIRED: Kubernetes namespace where job runs
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1",  # REQUIRED
    kms_key_id="1234abcd-12ab-34cd-56ef-1234567890ab"  # Optional
)
```

**Required Parameters:**
- `namespace`: Kubernetes namespace where your PyTorchJob runs (e.g., "kubeflow", "default")
- `kubectl_layer_arn`: ARN of the lambda-kubectl layer (see [Setup](#kubectl-lambda-layer) below)

**Optional Parameters (Auto-detected if not provided):**
- `eks_cluster_arn`: EKS cluster ARN
- `vpc_id`: VPC ID where HyperPod cluster runs
- `subnet_ids`: List of subnet IDs for Lambda function
- `security_group_id`: Security group ID for Lambda function
- `polling_interval_minutes`: How often to check job status (default: 5 minutes)

**Infrastructure Created:**
- CloudFormation Stack: `NovaForgeSDK-SMHP-JobNotifications-{cluster-name}`
- DynamoDB Table: Stores job notification configurations with namespace
- SNS Topic: Sends email notifications
- Lambda Function: Polls Kubernetes for PyTorchJob status using kubectl
- EventBridge Rule: Triggers Lambda on schedule (every N minutes)
    - Can be disabled to stop job tracking while leaving the infrastructure in place.
- VPC Endpoints: Gateway endpoints for DynamoDB and S3 (if needed)

## Setup Requirements
- If you prefer to deploy the infrastructure manually instead of using the SDK, see the [Manual CloudFormation Deployment](#manual-cloudformation-deployment) section below.
- If you deploy this stack and are no longer monitoring SMHP job notifications, delete the infrastructure following the steps [here](#deleting-notification-infrastructure). 
  - If you want to keep the infrastructure for later use, disable the EventBridge Scheduled Rule via the AWS Console instead of deleting the stack.

### Email Subscription Confirmation

After enabling notifications, each email address will receive an AWS SNS confirmation email. Users must click the confirmation link to start receiving notifications.

### kubectl Lambda Layer

For SMHP notifications, you need a Lambda layer containing the kubectl binary:

1. Deploy the layer from AWS Serverless Application Repository:
   - Go to: https://serverlessrepo.aws.amazon.com/applications/arn:aws:serverlessrepo:us-east-1:903779448426:applications~lambda-layer-kubectl
   - Click "Deploy"
   - Select the region you want to deploy the layer to from the top bar (e.g. "United States (Oregon)")
   - Name your lambda layer and click "Deploy"
   - Note the layer ARN from the generated outputs

2. Use the layer ARN when enabling notifications:
```python
kubectl_layer_arn = "arn:aws:lambda:<region>:123456789012:layer:kubectl:1"
```

### EKS Access Entry (SMHP Only)

After the notification infrastructure is created, you need to grant the Lambda function access to your EKS cluster. The Lambda role needs permission to query Kubernetes resources (PyTorchJobs, Pods) to monitor job status.

1. Get the Lambda role ARN from the CloudFormation stack outputs:
```bash
aws cloudformation describe-stacks \
  --stack-name NovaForgeSDK-SMHP-JobNotifications-YOUR-CLUSTER-NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`LambdaRoleArn`].OutputValue' \
  --output text
```

2. Get your EKS cluster name:
```bash
aws sagemaker describe-cluster \
  --cluster-name YOUR-HYPERPOD-CLUSTER-NAME \
  --query 'Orchestrator.Eks.ClusterArn' \
  --output text | awk -F'/' '{print $NF}'
```

3. Create an EKS access entry for the Lambda role:
```bash
aws eks create-access-entry \
  --cluster-name YOUR-EKS-CLUSTER-NAME \
  --principal-arn arn:aws:iam::ACCOUNT-ID:role/NovaForgeSDK-SMHP-NotifLambdaRole-HP-CLUSTER \
  --type STANDARD \
  --region YOUR-REGION
```

4. Associate the access policy to grant read permissions:

```bash
aws eks associate-access-policy \
  --cluster-name YOUR-EKS-CLUSTER-NAME \
  --principal-arn arn:aws:iam::ACCOUNT-ID:role/NovaForgeSDK-SMHP-NotifLambdaRole-HP-CLUSTER \
  --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
  --access-scope type=cluster \
  --region YOUR-REGION
```
**Note:** The `AmazonEKSClusterAdminPolicy` is needed so that the Lambda can monitor PyTorchJobs resources which aren't included in the ViewPolicy. 


**To delete an access entry** (e.g., when cleaning up or recreating the stack):
```bash
aws eks delete-access-entry \
  --cluster-name YOUR-EKS-CLUSTER-NAME \
  --principal-arn arn:aws:iam::ACCOUNT-ID:role/NovaForgeSDK-SMHP-NotifLambdaRole-HP-CLUSTER \
  --region YOUR-REGION
```

### KMS Key Permissions (Optional)

If using a customer-managed KMS key for SNS encryption, the CloudFormation stack automatically grants the Lambda role IAM permissions to use the key (`kms:Decrypt` and `kms:GenerateDataKey`).

**However**, if your KMS key has a restrictive key policy that doesn't allow IAM permissions to take effect, you'll need to manually grant access:

```bash
aws kms create-grant \
  --key-id YOUR-KEY-ID \
  --grantee-principal arn:aws:iam::ACCOUNT:role/NovaForgeSDK-SMHP-NotifLambdaRole-HP-CLUSTER \
  --operations Decrypt GenerateDataKey
```

**When is this needed?**
- Your KMS key policy explicitly denies IAM-based access
- Your KMS key policy doesn't include a statement allowing IAM policies to grant access
- You see "KMS Access Denied" errors in Lambda logs despite the IAM policy being in place

**Important:** Provide only the key ID (e.g., `1234abcd-12ab-34cd-56ef-1234567890ab`), not the full ARN. The SDK constructs the full ARN automatically.

## Manual CloudFormation Deployment

If you prefer to deploy the notification infrastructure manually (without using the SDK), you can use the CloudFormation templates directly.

### SMTJ Manual Deployment

The SMTJ template has minimal configuration and can be deployed directly:

1. Navigate to the template:
```bash
src/NovaCustomizationSDK/src/amzn_nova_forge/notifications/templates/smtj_notification_cf_stack.yaml
```

2. Deploy via AWS Console:
   - Go to CloudFormation → Create Stack
   - Upload the template file
   - Configure parameters:
     - `KmsKeyId` (Optional): Your KMS key ID for SNS encryption (just the ID, not full ARN)
   - Create the stack

3. OR Deploy via AWS CLI:
```bash
aws cloudformation create-stack \
  --stack-name NovaForgeSDK-SMTJ-JobNotifications \
  --template-body file://smtj_notification_cf_stack.yaml \
  --parameters ParameterKey=KmsKeyId,ParameterValue=YOUR-KEY-ID \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

4. Post-deployment:
   - Subscribe email addresses to the SNS topic (from stack outputs)
   - Manually add job entries to DynamoDB table if not using SDK

### SMHP Manual Deployment

The SMHP template supports both SDK-driven and manual deployment. For manual deployment:

1. Copy the template locally:
```bash
# Template location in SDK installation
cp /path/to/amzn_nova_forge/notifications/templates/smhp_notification_cf_stack.yaml .
```

2. Edit the template and uncomment the `Default:` lines under Parameters, replacing placeholders:
   - `ClusterName`: Your HyperPod cluster name
   - `EksClusterArn`: Get with `aws sagemaker describe-cluster --cluster-name <name> --query Orchestrator.Eks.ClusterArn`
   - `VpcId`: The VPC ID associated with your cluster (e.g., vpc-0123456789abcdef0)
   - `SubnetIds`: Comma-separated private subnet IDs with NAT gateway
   - `SecurityGroupId`: Security group allowing outbound HTTPS to EKS API
   - `RouteTableIds`: Route table IDs for your subnets (see parameter description for helper command)
   - `KubectlLayerArn`: Deploy from [AWS Serverless App Repository](https://serverlessrepo.aws.amazon.com/applications/arn:aws:serverlessrepo:us-east-1:903779448426:applications~lambda-layer-kubectl)
   - `CreateDynamoDBEndpoint`: True/False (check parameter description for helper command)
   - `CreateS3Endpoint`: True/False (check parameter description for helper command)
   - `PollingIntervalMinutes`: How often to check for job status (default: 5)
   - `KmsKeyId`: (Optional) Your KMS key ID

3. Deploy via AWS Console:
   - Go to CloudFormation -> Create Stack
   - Upload your edited template file
   - Review parameters (should already be filled in from Default values)
   - Create the stack with `CAPABILITY_NAMED_IAM`

4. Deploy via AWS CLI:
```bash
aws cloudformation create-stack \
  --stack-name NovaForgeSDK-SMHP-JobNotifications-YOUR-CLUSTER \
  --template-body file://smhp_notification_cf_stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

5. Post-deployment (REQUIRED):
   - Get Lambda role ARN from stack outputs
   - Create EKS access entry (see [EKS Access Entry](#eks-access-entry-smhp-only) section)
   - Subscribe email addresses to SNS topic
   - Manually add job entries to DynamoDB table if not using SDK


## Advanced Usage

### Using Notification Manager Directly

For jobs started outside the SDK or to enable notifications on existing jobs:

```python
from amzn_nova_forge.notifications import SMTJNotificationManager, SMHPNotificationManager

# For SMTJ jobs
smtj_manager = SMTJNotificationManager(region="us-east-1")
smtj_manager.enable_notifications(
    job_name="existing-training-job",
    emails=["user@example.com"],
    output_s3_path="s3://bucket/output/"
)

# For SMHP jobs
smhp_manager = SMHPNotificationManager(cluster_name="my-cluster", region="us-east-1")
smhp_manager.enable_notifications(
    job_name="existing-pytorch-job",
    emails=["user@example.com"],
    output_s3_path="s3://bucket/output/",
    namespace="kubeflow",
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1"
)
```

### Deleting Notification Infrastructure

To remove all notification infrastructure:

```python
smtj_manager = SMTJNotificationManager()
# Delete SMTJ infrastructure
smtj_manager.delete_notification_stack()

smhp_manager = SMHPNotificationManager(
    cluster_name="cluster-name", 
    region="us-east-1"
)
# Delete SMHP infrastructure
smhp_manager.delete_notification_stack()
```

**Note:** Stack deletion is initiated immediately but may take several minutes to complete (especially for SMHP with VPC resources). Check the CloudFormation console to monitor progress.

### Custom Polling Interval (SMHP Only)

Adjust how often the Lambda function checks job status:

```python
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1"
    polling_interval_minutes=10  # Check every 10 minutes instead of default 5
)
```

## Notification Content

### Email Format

Notifications include:
- **Subject**: `[SMTJ] Job Succeeded: job-name` or `[SMHP] Job Failed: job-name`
- **Body**:
  - Job ID
  - Platform (SMTJ or SMHP)
  - Cluster Name (SMHP only)
  - Namespace (SMHP only)
  - Status (Succeeded, Failed, Stopped, or Running (Degraded))
  - Timestamp
  - Artifact validation result (for successful jobs)
    - Training jobs: Validates manifest.json exists
    - Evaluation jobs: Note that manifest.json is not expected (evaluation jobs produce results_*.json files)
  - Pod health information (for degraded SMHP jobs)

### Example Email

Training job success:
```
SageMaker HyperPod Job Status Update
    - Job ID: my-training-job-abc123
    - Platform: SMHP
    - Cluster: my-cluster
    - Namespace: kubeflow
    - Status: Succeeded
    - Timestamp: 2026-03-13T10:30:00Z

Job completed successfully. manifest.json found.

---
This notification was sent by Forge SDK Job Notifications.
```

Evaluation job success:
```
SageMaker Training Job Status Update
- Job ID: my-eval-job-xyz789
- Platform: SMTJ
- Status: Completed
- Timestamp: 2026-03-13T10:30:00Z
- View in Console: https://console.aws.amazon.com/sagemaker/...

Job completed. manifest.json was not found in output.tar.gz.
Note: This is expected for evaluation jobs (which produce results_*.json instead).
For training jobs, please check your logs for details.

---
This notification was sent by Forge SDK Job Notifications.
```

## SMHP Pod Health Monitoring

For SMHP jobs, the notification system monitors the health of the master pod and can detect issues before the job fails:

### Monitored Conditions

1. **CrashLoopBackOff**: Pod is repeatedly crashing
2. **Excessive Restarts**: Pod has restarted more than 5 times

### Degraded Status Notifications

If a running job's master pod is unhealthy, you'll receive a notification with status `Running (Degraded - {reason})`:

```
SageMaker HyperPod Job Status Update
- Job ID: my-training-job-abc123
- Platform: SMHP
- Cluster: my-cluster
- Namespace: kubeflow
- Status: Running (Degraded - CrashLoopBackOff)
- Timestamp: 2026-03-13T10:30:00Z
- Master Pod Restarts: 8
- Issue: CrashLoopBackOff

---
This notification was sent by Forge SDK Job Notifications.
```
### SMHP: "kubectl error: Unauthorized"

If you see errors like:
- `kubectl error: Unauthorized`
- `User "arn:aws:sts::..." cannot get resource "pytorchjobs" in API group "kubeflow.org"`

**Solutions:**

1. **Check EKS access entry exists:**
```bash
aws eks list-access-entries --cluster-name YOUR-CLUSTER --region YOUR-REGION
```

2. **Verify the access policy is associated:**
```bash
aws eks list-associated-access-policies \
  --cluster-name YOUR-CLUSTER \
  --principal-arn arn:aws:iam::ACCOUNT:role/NovaForgeSDK-SMHP-NotifLambdaRole-HP-CLUSTER \
  --region YOUR-REGION
```

3. **If using AmazonEKSViewPolicy, it doesn't include PyTorchJob permissions:**
   - Switch to `AmazonEKSClusterAdminPolicy` (simpler), OR
   - Add custom Kubernetes RBAC for PyTorchJobs (see [EKS Access Entry](#eks-access-entry-smhp-only) section Option B)

4. **Verify kubectl layer:** Ensure layer ARN is correct and deployed (include the version number)

5. **Check VPC configuration:** Lambda needs network access to EKS API
### General
- **Email Confirmation Required**: Users must confirm their email subscription before receiving notifications
- **One Stack Per Region/Cluster**: Infrastructure is shared across all jobs in the same region (SMTJ) or cluster (SMHP)
- **TTL**: Job configurations are automatically deleted from DynamoDB 30 days after creation
- **Terminal States Only**: Notifications are sent only when jobs reach Completed, Failed, Stopped, or Degraded states

### SMTJ-Specific
- **EventBridge Latency**: Notifications may have a slight delay (typically < 1 minute) after job state change
- **Training Jobs Only**: Only supports SageMaker Training Jobs

### SMHP-Specific
- **Polling Interval**: Job status is checked on a schedule (default: every 5 minutes), not in real-time
- **Kubernetes Access Required**: Lambda function needs network access to EKS API server
- **VPC Configuration**: Lambda runs in VPC and requires proper subnet/security group configuration
- **kubectl Layer**: Requires deploying a Lambda layer with kubectl binary
- **Namespace Required**: Must specify the Kubernetes namespace where the job runs
- **PyTorchJob Only**: Currently only monitors PyTorchJob resources, not other Kubernetes job types
- **Master Pod Monitoring**: Health checks only monitor the master pod (pod-0), not worker pods
- **Manual Notification Disablement**: If you want to keep the infrastructure but are not currently tracking any jobs, *disable* the EventBridge Scheduled rule via the Console to stop it from running incrementally

### Cost Considerations
- **Lambda Invocations**: SMHP Lambda is invoked every N minutes (default: 5) per cluster
    - Remember to disable the EventBridge rule when you're not tracking any current SMHP jobs.
- **DynamoDB**: Pay-per-request pricing for job configuration storage
- **SNS**: Minimal cost for email notifications
- **VPC Endpoints**: Gateway endpoints (DynamoDB, S3) have no hourly charges

## Troubleshooting

### Emails Not Received

1. **Check spam folder**: SNS emails may be filtered
2. **Confirm subscription**: Click the confirmation link in the initial SNS email
3. **Check SNS topic**: Verify email is subscribed in AWS Console → SNS → Topics

### SMHP: "kubectl error: Unauthorized"

1. **Check EKS access entry**: Lambda role needs EKS cluster access (refer to [EKS Access Entry](#eks-access-entry-smhp-only) for more information)
2. **Verify kubectl layer**: Ensure layer ARN is correct and deployed (include the version number)
3. **Check VPC configuration**: Lambda needs network access to EKS API

### SMHP: "namespace is required" Error

Provide the `namespace` parameter when enabling notifications:
```python
result.enable_job_notifications(
    emails=["user@example.com"],
    namespace="kubeflow",  # Add this
    kubectl_layer_arn="arn:aws:lambda:<region>:123456789012:layer:kubectl:1"
)
```

### KMS Access Denied

If using a customer KMS key, ensure the Lambda role has permissions:
```bash
aws kms create-grant \
  --key-id YOUR-KEY-ID \
  --grantee-principal arn:aws:iam::ACCOUNT:role/NovaForgeSDK-*-NotificationLambda-Role-* \
  --operations Decrypt GenerateDataKey
```

### SMHP: "ResourceInUseException" for CreateAccessEntry
```
An error occurred (ResourceInUseException) when calling the CreateAccessEntry operation: The specified access entry resource is already in use on this cluster.
```
This occurs when someone has already created the infrastructure stack but didn't delete the previous EKS access entry. You need to delete this old EKS access entry before creating a new one.
1. Navigate to the AWS Console -> Find the job monitoring Lambda function. 
2. Go to the "Access" tab and find the lambda role that was created by the CloudFormation stack. 
    - Includes: "NovaForgeSDK-PLATFORM-NotificationLambda-Role"
3. Delete this entry and run the steps in the [EKS Access Entry](#eks-access-entry-smhp-only) section again.

### General Tip: Debugging job monitoring errors

If you are running into any errors listed here or not, you can reference the lambda's CloudWatch logs to learn more about what isn't sending/set up correctly. 
1. Navigate to the lambda function (names listed below) in your AWS Console.
2. Go to the "Monitor" tab and click "View CloudWatch Logs".
3. The top log stream is the newest, so start there and try to find the error there.

**Lambda Function:**
* **SMTJ:** NovaForgeSDK-SMTJ-NotificationHandler (one per region)
* **SMHP:** NovaForgeSDK-SMHP-NotificationHandler-CLUSTER-NAME (one per cluster)

**Note:** If no CloudWatch logs are being produced even though you enabled job notifications, check that the EventBridge "Scheduled Rule" is "Enabled".