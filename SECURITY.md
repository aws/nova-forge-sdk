Reporting Security Issues
----------------------------------------------------------------------------------------------------------
We take all security reports seriously. When we receive such reports, we will investigate and  subsequently address any potential vulnerabilities as quickly as possible. 
If you discover a potential security issue in this project, please notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/) or
directly via email to [AWS Security](mailto:aws-security@amazon.com).
Please do not create a public GitHub issue in this project.

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

The SDK requires specific IAM permissions. Review the [IAM](README.md#iam) section and:

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