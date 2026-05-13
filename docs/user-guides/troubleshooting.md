# Troubleshooting (FAQ)

## Overview

This document provides guidance for common problems that might be faced when using the Nova Forge SDK. 

## Permissions-Based Issues
### Unable to Deploy a Custom Model to Bedrock
If you are unable to use the SDK's built-in `deploy()` function due to permissioning issues, you can manually call the Bedrock APIs to import and deploy your models. 
This will still require some IAM permissions to be set up. 
The steps are outlined below.

#### Step 1: Locate Your Training Artifacts and extract your checkpoint_s3_path
* First, find where your training job is saved (`output_s3_path`) in S3. 
* For SMTJ jobs, follow the steps [here](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-iterative-training.html#nova-iterative-how-it-works) to get the s3 escrow location where your model is saved. 
* For SMHP, when you navigate to your `output_s3_path` S3 folder, open the `manifest.json` file which will only contain the `checkpoint_s3_path` value. 

#### Step 2: Import your custom model from S3 Escrow
* Follow the steps to [Create a Custom Model](https://docs.aws.amazon.com/bedrock/latest/userguide/create-custom-model-sdks.html) here. 
* Provide the `checkpoint_s3_path` value from Step 1 for the `s3Uri` value under `modelSourceConfig`. 

#### Step 3: Deploy your custom model in Bedrock
* After your custom model is imported from escrow, you can follow the steps [here](https://docs.aws.amazon.com/bedrock/latest/userguide/deploy-custom-model-on-demand.html#deploy-custom-model) to deploy the model to Bedrock using the Console, AWS CLI, or Bedrock APIs. 

#### Notes:
* If you're running into permission issues with importing and deploying your custom Nova model, please review the AWS documentation: [Create a service role for importing pre-trained models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-import-iam-role.html).