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
from typing import Annotated, Optional

import boto3
from fastmcp import FastMCP

from amzn_nova_forge.dataset import JSONLDatasetLoader
from amzn_nova_forge.manager import (
    BedrockRuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge.model import (
    DeployPlatform,
    Model,
    NovaModelCustomizer,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge.monitor import CloudWatchLogMonitor
from amzn_nova_forge.recipe import EvaluationTask

mcp = FastMCP(
    "nova-forge",
    instructions=(
        "Nova Forge MCP server for fine-tuning and customizing Amazon Nova models. "
        "Use these tools to train, evaluate, deploy, and monitor Nova model customization jobs."
    ),
)

# --- Enum lookups ---

MODELS = {m.name: m for m in Model}
TRAINING_METHODS = {m.name: m for m in TrainingMethod}
PLATFORMS = {p.name: p for p in Platform}
DEPLOY_PLATFORMS = {p.name: p for p in DeployPlatform}
EVAL_TASKS = {t.name: t for t in EvaluationTask}


def _resolve(mapping: dict, key: str, label: str):
    """Look up a key in an enum mapping, raising ValueError with valid options on miss."""
    if key not in mapping:
        valid = ", ".join(mapping)
        raise ValueError(f"Unknown {label}: {key!r}. Valid options: {valid}")
    return mapping[key]


def _build_infra(
    platform: str,
    instance_type: Optional[str] = None,
    instance_count: Optional[int] = None,
    execution_role: Optional[str] = None,
    kms_key_id: Optional[str] = None,
    cluster_name: Optional[str] = None,
    namespace: Optional[str] = None,
):
    """Construct the appropriate RuntimeManager from flat parameters."""
    if platform == "SMTJ":
        if not instance_type or not instance_count:
            raise ValueError("SMTJ platform requires instance_type and instance_count")
        return SMTJRuntimeManager(
            instance_type=instance_type,
            instance_count=instance_count,
            execution_role=execution_role,
            kms_key_id=kms_key_id,
        )
    elif platform == "SMHP":
        if not instance_type or not instance_count or not cluster_name:
            raise ValueError(
                "SMHP platform requires instance_type, instance_count, and cluster_name"
            )
        return SMHPRuntimeManager(
            instance_type=instance_type,
            instance_count=instance_count,
            cluster_name=cluster_name,
            namespace=namespace or "kubeflow",
            kms_key_id=kms_key_id,
        )
    elif platform == "BEDROCK":
        if not execution_role:
            raise ValueError("BEDROCK platform requires execution_role")
        return BedrockRuntimeManager(
            execution_role=execution_role,
            kms_key_id=kms_key_id,
        )
    else:
        raise ValueError(
            f"Unknown platform: {platform}. Must be one of: SMTJ, SMHP, BEDROCK"
        )


def _format_job_result(result) -> str:
    """Format a job result object into a readable string."""
    if result is None:
        return "Dry run completed successfully. No job was launched."
    lines = [
        f"Job ID: {result.job_id}",
        f"Started: {result.started_time.isoformat()}",
    ]
    status, raw = result.get_job_status()
    lines.append(f"Status: {raw}")
    if hasattr(result, "method"):
        lines.append(f"Training Method: {result.method.value}")
    if hasattr(result, "model_artifacts") and result.model_artifacts:
        lines.append(f"Model Artifacts: {result.model_artifacts}")
    if hasattr(result, "eval_task"):
        lines.append(f"Eval Task: {result.eval_task.value}")
    if hasattr(result, "eval_output_path"):
        lines.append(f"Eval Output: {result.eval_output_path}")
    return "\n".join(lines)


# --- Tools ---


@mcp.tool()
def list_options() -> str:
    """List all available Nova models, training methods, platforms, deploy platforms, and evaluation tasks."""
    sections = [
        ("Models", [f"  {name}" for name in MODELS]),
        ("Training Methods", [f"  {name}" for name in TRAINING_METHODS]),
        ("Platforms", [f"  {name}" for name in PLATFORMS]),
        ("Deploy Platforms", [f"  {name}" for name in DEPLOY_PLATFORMS]),
        ("Evaluation Tasks", [f"  {name}" for name in EVAL_TASKS]),
    ]
    parts = []
    for title, items in sections:
        parts.append(f"{title}:\n" + "\n".join(items))
    return "\n\n".join(parts)


@mcp.tool()
def train(
    model: Annotated[str, "Model name (e.g. NOVA_PRO, NOVA_LITE, NOVA_MICRO, NOVA_LITE_2)"],
    training_method: Annotated[str, "Training method (e.g. SFT_LORA, SFT_FULL, DPO_LORA, DPO_FULL, CPT, RFT_LORA, RFT_FULL)"],
    platform: Annotated[str, "Infrastructure platform: SMTJ, SMHP, or BEDROCK"],
    job_name: Annotated[str, "Unique name for this training job"],
    data_s3_path: Annotated[str, "S3 path to training data"],
    output_s3_path: Annotated[str, "S3 path for output artifacts"],
    instance_type: Annotated[Optional[str], "Instance type (required for SMTJ/SMHP, e.g. ml.p5.48xlarge)"] = None,
    instance_count: Annotated[Optional[int], "Number of instances (required for SMTJ/SMHP)"] = None,
    execution_role: Annotated[Optional[str], "IAM execution role ARN (required for BEDROCK)"] = None,
    cluster_name: Annotated[Optional[str], "HyperPod cluster name (required for SMHP)"] = None,
    namespace: Annotated[Optional[str], "HyperPod namespace (default: kubeflow)"] = None,
    kms_key_id: Annotated[Optional[str], "KMS key ID for encryption"] = None,
    model_path: Annotated[Optional[str], "S3 path to a base model or checkpoint for iterative training"] = None,
    validation_data_s3_path: Annotated[Optional[str], "S3 path to validation data"] = None,
    overrides: Annotated[Optional[dict], "Recipe override parameters (e.g. hyperparameters)"] = None,
    dry_run: Annotated[bool, "If true, validate config without launching a job"] = False,
) -> str:
    """Launch a Nova model fine-tuning job. Returns the job ID and status."""
    infra = _build_infra(
        platform=platform,
        instance_type=instance_type,
        instance_count=instance_count,
        execution_role=execution_role,
        kms_key_id=kms_key_id,
        cluster_name=cluster_name,
        namespace=namespace,
    )
    customizer = NovaModelCustomizer(
        model=_resolve(MODELS, model, "model"),
        method=_resolve(TRAINING_METHODS, training_method, "training_method"),
        infra=infra,
        data_s3_path=data_s3_path,
        output_s3_path=output_s3_path,
        model_path=model_path,
    )
    result = customizer.train(
        job_name=job_name,
        overrides=overrides,
        validation_data_s3_path=validation_data_s3_path,
        dry_run=dry_run,
    )
    return _format_job_result(result)


@mcp.tool()
def evaluate(
    model: Annotated[str, "Model name (e.g. NOVA_PRO, NOVA_LITE)"],
    platform: Annotated[str, "Infrastructure platform: SMTJ, SMHP, or BEDROCK"],
    job_name: Annotated[str, "Unique name for this evaluation job"],
    eval_task: Annotated[str, "Evaluation task (e.g. MMLU, MMLU_PRO, BBH, GPQA, MATH, GEN_QA, IFEVAL, LLM_JUDGE)"],
    data_s3_path: Annotated[Optional[str], "S3 path to evaluation data"] = None,
    output_s3_path: Annotated[Optional[str], "S3 path for output"] = None,
    model_path: Annotated[Optional[str], "S3 path to the fine-tuned model to evaluate"] = None,
    instance_type: Annotated[Optional[str], "Instance type (required for SMTJ/SMHP)"] = None,
    instance_count: Annotated[Optional[int], "Number of instances (required for SMTJ/SMHP)"] = None,
    execution_role: Annotated[Optional[str], "IAM execution role ARN (required for BEDROCK)"] = None,
    cluster_name: Annotated[Optional[str], "HyperPod cluster name (required for SMHP)"] = None,
    namespace: Annotated[Optional[str], "HyperPod namespace"] = None,
    kms_key_id: Annotated[Optional[str], "KMS key ID for encryption"] = None,
    overrides: Annotated[Optional[dict], "Recipe override parameters"] = None,
    dry_run: Annotated[bool, "If true, validate config without launching"] = False,
) -> str:
    """Launch a Nova model evaluation job. Returns the job ID and status."""
    infra = _build_infra(
        platform=platform,
        instance_type=instance_type,
        instance_count=instance_count,
        execution_role=execution_role,
        kms_key_id=kms_key_id,
        cluster_name=cluster_name,
        namespace=namespace,
    )
    customizer = NovaModelCustomizer(
        model=_resolve(MODELS, model, "model"),
        method=TrainingMethod.EVALUATION,
        infra=infra,
        data_s3_path=data_s3_path,
        output_s3_path=output_s3_path,
        model_path=model_path,
    )
    result = customizer.evaluate(
        job_name=job_name,
        eval_task=_resolve(EVAL_TASKS, eval_task, "eval_task"),
        overrides=overrides,
        dry_run=dry_run,
    )
    return _format_job_result(result)


@mcp.tool()
def deploy(
    model: Annotated[str, "Model name (e.g. NOVA_PRO, NOVA_LITE)"],
    training_method: Annotated[str, "Training method used to produce the model"],
    platform: Annotated[str, "Infrastructure platform the model was trained on: SMTJ, SMHP, or BEDROCK"],
    deploy_platform: Annotated[str, "Where to deploy: SAGEMAKER, BEDROCK_OD, or BEDROCK_PT"] = "BEDROCK_OD",
    model_artifact_path: Annotated[Optional[str], "S3 path to the model artifact"] = None,
    endpoint_name: Annotated[Optional[str], "Name for the deployment endpoint"] = None,
    execution_role_name: Annotated[Optional[str], "IAM role name for deployment"] = None,
    instance_type: Annotated[Optional[str], "Instance type (for training infra setup)"] = None,
    instance_count: Annotated[Optional[int], "Instance count (for training infra setup)"] = None,
    execution_role: Annotated[Optional[str], "IAM execution role ARN (for training infra setup)"] = None,
    cluster_name: Annotated[Optional[str], "HyperPod cluster name"] = None,
    namespace: Annotated[Optional[str], "HyperPod namespace"] = None,
    kms_key_id: Annotated[Optional[str], "KMS key ID"] = None,
    sagemaker_instance_type: Annotated[Optional[str], "SageMaker endpoint instance type"] = "ml.p5.48xlarge",
) -> str:
    """Deploy a fine-tuned Nova model to SageMaker or Bedrock."""
    infra = _build_infra(
        platform=platform,
        instance_type=instance_type,
        instance_count=instance_count,
        execution_role=execution_role,
        kms_key_id=kms_key_id,
        cluster_name=cluster_name,
        namespace=namespace,
    )
    customizer = NovaModelCustomizer(
        model=_resolve(MODELS, model, "model"),
        method=_resolve(TRAINING_METHODS, training_method, "training_method"),
        infra=infra,
    )
    result = customizer.deploy(
        model_artifact_path=model_artifact_path,
        deploy_platform=_resolve(DEPLOY_PLATFORMS, deploy_platform, "deploy_platform"),
        endpoint_name=endpoint_name,
        execution_role_name=execution_role_name,
        sagemaker_instance_type=sagemaker_instance_type,
    )
    lines = [f"Deployment initiated."]
    if hasattr(result, "endpoint_name"):
        lines.append(f"Endpoint: {result.endpoint_name}")
    if hasattr(result, "job_id"):
        lines.append(f"Job ID: {result.job_id}")
    return "\n".join(lines)


@mcp.tool()
def get_job_status(
    job_id: Annotated[str, "The job ID to check status for"],
    platform: Annotated[str, "Platform the job ran on: SMTJ, SMHP, or BEDROCK"],
) -> str:
    """Get the current status of a training or evaluation job."""
    _resolve(PLATFORMS, platform, "platform")
    if platform == "SMTJ":
        client = boto3.client("sagemaker")
        response = client.describe_training_job(TrainingJobName=job_id)
        status = response.get("TrainingJobStatus", "Unknown")
        failure = response.get("FailureReason", "")
        lines = [
            f"Job: {job_id}",
            f"Status: {status}",
        ]
        if failure:
            lines.append(f"Failure Reason: {failure}")
        if "TrainingStartTime" in response:
            lines.append(f"Started: {response['TrainingStartTime'].isoformat()}")
        if "TrainingEndTime" in response:
            lines.append(f"Ended: {response['TrainingEndTime'].isoformat()}")
        return "\n".join(lines)
    elif platform == "BEDROCK":
        client = boto3.client("bedrock")
        response = client.get_model_customization_job(jobIdentifier=job_id)
        status = response.get("status", "Unknown")
        lines = [
            f"Job: {job_id}",
            f"Status: {status}",
        ]
        if "failureMessage" in response:
            lines.append(f"Failure: {response['failureMessage']}")
        return "\n".join(lines)
    else:
        return f"Job: {job_id}\nPlatform: {platform}\nUse get_logs to check progress."


@mcp.tool()
def get_logs(
    job_id: Annotated[str, "The job ID to retrieve logs for"],
    platform: Annotated[str, "Platform: SMTJ, SMHP, or BEDROCK"],
    limit: Annotated[Optional[int], "Maximum number of log events to return"] = 50,
) -> str:
    """Retrieve CloudWatch logs for a training or evaluation job."""
    monitor = CloudWatchLogMonitor.from_job_id(
        job_id=job_id,
        platform=_resolve(PLATFORMS, platform, "platform"),
    )
    logs = monitor.get_logs(limit=limit)
    if not logs:
        return f"No logs found for job {job_id}."
    lines = []
    for entry in logs:
        ts = entry.get("timestamp", "")
        msg = entry.get("message", "").strip()
        lines.append(f"[{ts}] {msg}")
    return "\n".join(lines)


@mcp.tool()
def validate_dataset(
    data_path: Annotated[str, "Path to dataset file (local or s3://)"],
    model: Annotated[Optional[str], "Model name to validate against"] = None,
) -> str:
    """Load and validate a JSONL dataset for Nova fine-tuning. Reports any schema errors."""
    loader = JSONLDatasetLoader()
    try:
        loader.load(data_path)
        model_enum = _resolve(MODELS, model, "model") if model else None
        loader.validate(model=model_enum)
        record_count = len(loader.dataset) if hasattr(loader, "dataset") else "unknown"
        return f"Dataset is valid.\nRecords: {record_count}\nPath: {data_path}"
    except Exception as e:
        return f"Dataset validation failed:\n{e}"


def main():
    """Entry point for running the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
