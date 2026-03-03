# Copyright 2025 Amazon Inc

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
RFT_POLICY_NAME = (
    "RFTPolicyNovaSDK"  # Used for both task role name and inline policy name
)

# Stack name suffix
STACK_NAME_SUFFIX = "NovaForgeSDK"

# ECR repository name
ECR_REPO_NAME = "nova-rft-base"

# Folder names for organizing RFT files
SDK_RFT_LOGS_DIR = "sdk-rft-logs"
SDK_RFT_SCRIPTS_DIR = "sdk-rft-scripts"

# Log file names (legacy - kept for reference)
RFT_TRAIN_LOG = "rft_train.log"
RFT_EVAL_LOG = "rft_eval.log"
RFT_SAM_LOG = "rft_sam.log"

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

# SSM command polling interval in seconds
SSM_COMMAND_POLL_INTERVAL = 0.5

# SSM command max polling attempts
SSM_COMMAND_MAX_POLL_ATTEMPTS = 10
