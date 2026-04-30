# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constants for RFT Multiturn infrastructure."""

# Shared role and policy names
RFT_EXECUTION_ROLE_NAME = "RFTExecutionRoleNovaSDK"
RFT_POLICY_NAME = "RFTPolicyNovaSDK"  # Used for both task role name and inline policy name

# Stack name suffix
STACK_NAME_SUFFIX = "NovaForgeSDK"

# Folder names for organizing RFT files
SDK_RFT_LOGS_DIR = "sdk-rft-logs"
SDK_RFT_SCRIPTS_DIR = "sdk-rft-scripts"

# Job status values
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_KILLED = "killed"
JOB_STATUS_FAILED = "failed"

# IAM propagation wait time in seconds
IAM_PROPAGATION_WAIT_TIME = 15
SAM_WAIT_TIME = 600

# Starter kit S3 location
STARTER_KIT_S3 = "s3://nova-rft-starter-kit-c7363-206080352451-us-east-1/v1"

# ECS task container image and starter kit path
ECS_IMAGE_URI = "public.ecr.aws/amazonlinux/amazonlinux:2023"
ECS_STARTER_KIT_PATH = "/root/v1"

# EC2 paths
EC2_BASE_PATH = "/home/ec2-user"
EC2_STARTER_KIT_PATH = f"{EC2_BASE_PATH}/v1"
EC2_LOGS_PATH = f"{EC2_BASE_PATH}/{SDK_RFT_LOGS_DIR}"
EC2_SCRIPTS_PATH = f"{EC2_BASE_PATH}/{SDK_RFT_SCRIPTS_DIR}"

# Python command used on EC2/ECS platforms
BASE_PYTHON_COMMAND = "python3.12"

# CloudFormation stack states that are unrecoverable — treat as non-existent
CFN_UNUSABLE_STACK_STATES = frozenset({"DELETE_FAILED", "ROLLBACK_COMPLETE", "ROLLBACK_FAILED"})

# SSM command polling interval in seconds
SSM_COMMAND_POLL_INTERVAL = 0.5

# SSM command max polling attempts
SSM_COMMAND_MAX_POLL_ATTEMPTS = 10
