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
"""ForgeDeployer — owns the deployment lifecycle."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import boto3

from amzn_nova_forge.core.constants import (
    ESCROW_URI_TAG_KEY,
    ModelStatus,
    _escrow_tag_value,
)
from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    TrainingMethod,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.types import (
    DeploymentResult,
    EndpointInfo,
    ForgeConfig,
    validate_region,
)
from amzn_nova_forge.iam.iam_role_creator import (
    create_bedrock_execution_role,
    create_sagemaker_execution_role,
)
from amzn_nova_forge.model.model_config import ModelDeployResult
from amzn_nova_forge.telemetry import UNKNOWN, Feature, _telemetry_emitter
from amzn_nova_forge.util.bedrock import (
    BEDROCK_EXECUTION_ROLE_NAME,
    DEPLOYMENT_ARN_NAME,
    check_deployment_status,
    check_existing_deployment,
    find_bedrock_model_by_tag,
    get_required_bedrock_update_permissions,
    monitor_model_create,
    update_provisioned_throughput_model,
    wait_for_model_ready,
)
from amzn_nova_forge.util.logging import logger
from amzn_nova_forge.util.sagemaker import (
    SAGEMAKER_EXECUTION_ROLE_NAME,
    _validate_sagemaker_instance_type_for_model_deployment,
    create_sagemaker_endpoint,
    create_sagemaker_model,
    find_sagemaker_model_by_tag,
    setup_environment_variables,
)
from amzn_nova_forge.validation.endpoint_validator import (
    SAGEMAKER_ENDPOINT_ARN_REGEX,
    validate_endpoint_arn,
    validate_sagemaker_environment_variables,
    validate_unit_count,
)
from amzn_nova_forge.validation.validator import Validator


class ForgeDeployer:
    """Encapsulates the deployment lifecycle for Nova models.

    Does NOT use RuntimeManager — deployment is a direct platform API call.
    """

    def __init__(
        self,
        region: str,
        model: Model,
        deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS,
        config: Optional[ForgeConfig] = None,
        method: Optional[TrainingMethod] = None,
    ) -> None:
        # Region is required (no fallback) because deployment targets a specific region.
        self.region = region
        validate_region(self.region)
        self.model = model
        # TODO: Remove method out of the deployer
        self.method = method
        self.deployment_mode = deployment_mode
        self._config = config or ForgeConfig()

        # Model reuse: session cache of (platform, arn, escrow_path) tuples
        self._published_models: Set[Tuple[str, str, str]] = set()
        self.last_model_publish: Optional[ModelDeployResult] = None

    @_telemetry_emitter(
        Feature.DEPLOY,
        "deploy",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": kwargs.get("deploy_platform", DeployPlatform.BEDROCK_OD),
        },
    )
    def deploy(
        self,
        model_artifact_path: str,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        endpoint_name: Optional[str] = None,
        unit_count: int = 1,
        execution_role_name: Optional[str] = None,
        sagemaker_instance_type: Optional[str] = "ml.p5.48xlarge",
        sagemaker_environment_variables: Optional[Dict[str, Any]] = None,
        skip_model_reuse: bool = False,
    ) -> DeploymentResult:
        """Deploy a model to Bedrock or SageMaker.

        Args:
            model_artifact_path: S3 path to the trained model checkpoint,
                or an existing Bedrock custom model ARN.
            deploy_platform: Target platform.
            endpoint_name: Custom endpoint name (auto-generated if omitted).
            unit_count: Instance/PT unit count.
            execution_role_name: Optional IAM role name.
            sagemaker_instance_type: EC2 instance type for SageMaker.
            sagemaker_environment_variables: Optional env vars for SageMaker.
            skip_model_reuse: If True, always create a new model (skip tag-based discovery).

        Returns:
            DeploymentResult with endpoint information.
        """
        validate_unit_count(unit_count)

        if deploy_platform in (DeployPlatform.BEDROCK_OD, DeployPlatform.BEDROCK_PT):
            return self._deploy_to_bedrock(
                model_artifact_path=model_artifact_path,
                deploy_platform=deploy_platform,
                pt_units=unit_count,
                endpoint_name=endpoint_name,
                execution_role_name=execution_role_name,
                skip_model_reuse=skip_model_reuse,
            )
        elif deploy_platform == DeployPlatform.SAGEMAKER:
            if sagemaker_instance_type is None:
                raise ValueError("sagemaker_instance_type cannot be None for SageMaker deployment")

            context_length = None
            max_concurrency = None
            if sagemaker_environment_variables:
                context_length = sagemaker_environment_variables.get("CONTEXT_LENGTH")
                max_concurrency = sagemaker_environment_variables.get("MAX_CONCURRENCY")

            _validate_sagemaker_instance_type_for_model_deployment(
                sagemaker_instance_type, self.model, context_length, max_concurrency
            )

            artifact_path = (
                model_artifact_path
                if model_artifact_path.endswith("/")
                else model_artifact_path + "/"
            )

            if artifact_path.startswith("arn:aws:bedrock:"):
                raise ValueError(
                    "Cannot deploy Bedrock-customized models to SageMaker. "
                    "Train on SageMaker first."
                )

            return self._deploy_to_sagemaker(
                model_artifact_path=artifact_path,
                endpoint_name=endpoint_name,
                instance_type=sagemaker_instance_type,
                unit_count=unit_count,
                environment_variables=sagemaker_environment_variables,
                execution_role_name=execution_role_name,
            )
        else:
            raise ValueError(f"Unsupported deployment platform: {deploy_platform}")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @_telemetry_emitter(
        Feature.DEPLOY,
        "get_status",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
        },
    )
    def get_status(self, result: DeploymentResult) -> JobStatus:
        """Check deployment status from a DeploymentResult."""
        return result.status

    @_telemetry_emitter(
        Feature.DEPLOY,
        "get_status_by_arn",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
            "platform": kwargs.get("platform", UNKNOWN),
        },
    )
    def get_status_by_arn(self, endpoint_arn: str, platform: DeployPlatform) -> Optional[JobStatus]:
        """Check deployment status by ARN."""
        status_str = check_deployment_status(endpoint_arn, platform)
        if status_str is None:
            return None
        try:
            return JobStatus(status_str)
        except ValueError:
            logger.warning(f"Unknown deployment status: {status_str}")
            return None

    @_telemetry_emitter(
        Feature.DEPLOY,
        "get_logs",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
        },
    )
    def get_logs(
        self,
        job_result: Optional[DeploymentResult] = None,
        endpoint_arn: Optional[str] = None,
        platform: Optional[DeployPlatform] = None,
    ) -> None:
        """Log deployment status information."""
        arn = endpoint_arn or (job_result.endpoint.uri if job_result else None)

        if not arn:
            logger.info("No endpoint ARN available. Call deploy() first.")
            return

        if job_result is not None:
            platform = job_result.endpoint.platform
        elif platform is None:
            if arn.startswith("arn:aws:sagemaker:"):
                platform = DeployPlatform.SAGEMAKER
            else:
                platform = DeployPlatform.BEDROCK_OD

        status = check_deployment_status(arn, platform)
        logger.info(f"Deployment status for {arn}: {status}")

    # ------------------------------------------------------------------
    # Model reuse
    # ------------------------------------------------------------------

    @_telemetry_emitter(
        Feature.DEPLOY,
        "find_published_model",
        extra_info_fn=lambda self, *args, **kwargs: {
            "model": self.model.value,
        },
    )
    def find_published_model(
        self, platform: str, escrow_path: str, skip_model_reuse: bool = False
    ) -> Optional[str]:
        """Find a previously published model ARN for the given platform and escrow path.

        First checks session cache, then queries AWS APIs by escrow tag.

        Args:
            platform: "bedrock" or "sagemaker"
            escrow_path: S3 path used to create the model
            skip_model_reuse: If True, skip discovery and return None

        Returns:
            Model ARN if found, None otherwise
        """
        if skip_model_reuse:
            return None

        # Fast path: session cache
        for cached_platform, cached_arn, cached_path in self._published_models:
            if cached_platform == platform and cached_path == escrow_path:
                return cached_arn

        # Slow path: API-based tag discovery
        found_arn: Optional[str] = None
        try:
            if platform == "sagemaker":
                sagemaker_client = boto3.client("sagemaker", region_name=self.region)
                found_arn = find_sagemaker_model_by_tag(escrow_path, sagemaker_client)
            elif platform == "bedrock":
                bedrock_client = boto3.client("bedrock", region_name=self.region)
                found_arn = find_bedrock_model_by_tag(escrow_path, bedrock_client)
        except Exception as e:
            logger.warning(
                f"Could not search for existing {platform} models: {e}. Will create a new model."
            )
            return None

        if found_arn:
            self._published_models.add((platform, found_arn, escrow_path))
        return found_arn

    # ------------------------------------------------------------------
    # Two-stage Bedrock deployment: create model, then deploy endpoint
    # ------------------------------------------------------------------

    def create_custom_model(
        self,
        model_artifact_path: str,
        endpoint_name: Optional[str] = None,
        execution_role_name: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        skip_model_reuse: bool = False,
    ) -> ModelDeployResult:
        """Create a Bedrock custom model from S3 artifacts.

        Extracts the model-creation step from the deploy flow so users can
        create a model independently of endpoint deployment.

        Args:
            model_artifact_path: S3 path to trained model checkpoint.
            endpoint_name: Optional name prefix for the model name.
            execution_role_name: Optional IAM role name. If None, the SDK creates
                and manages the default role.
            tags: Optional list of {"key": str, "value": str} dicts for source tracking.
            skip_model_reuse: If True, always create a new model (skip tag-based discovery).

        Returns:
            ModelDeployResult with model_arn, model_name, escrow_uri, etc.
        """
        # Check for existing model by escrow tag
        existing = self.find_published_model("bedrock", model_artifact_path, skip_model_reuse)
        if existing:
            logger.info(
                f"Found existing Bedrock model {existing} for escrow URI "
                f"'{model_artifact_path}'. Reusing instead of creating a duplicate."
            )
            bedrock_client = boto3.client("bedrock", region_name=self.region)
            result = ModelDeployResult.from_arn(existing, bedrock_client)
            model_status = result.status

            if model_status == ModelStatus.ACTIVE:
                self.last_model_publish = result
                return result
            elif model_status == ModelStatus.CREATING:
                logger.info("Model %s still Creating. Waiting for completion...", existing)
                wait_for_model_ready(bedrock_client, existing)
                self.last_model_publish = result
                return result
            elif model_status == ModelStatus.FAILED:
                logger.warning("Model %s is Failed. Creating new model instead.", existing)
            else:
                logger.warning("Model %s has unknown status. Creating new model.", existing)

        bedrock_client = boto3.client("bedrock", region_name=self.region)
        iam_client = boto3.client("iam", region_name=self.region)

        # Resolve execution role
        if execution_role_name is None:
            try:
                bedrock_execution_role_arn = create_bedrock_execution_role(
                    iam_client=iam_client, role_name=BEDROCK_EXECUTION_ROLE_NAME
                )["Role"]["Arn"]
            except Exception as e:
                raise RuntimeError(f"Failed to create the default Bedrock IAM Execution Role: {e}")
        else:
            try:
                bedrock_execution_role_arn = iam_client.get_role(RoleName=execution_role_name)[
                    "Role"
                ]["Arn"]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to retrieve user-specified IAM role '{execution_role_name}': {e}"
                )

        if endpoint_name is None:
            if self.method is not None:
                name_format = f"{self.model}-{self.method.value}-{self.region}".lower()
            else:
                name_format = f"{self.model}-{self.region}".lower()
            endpoint_name = name_format.replace(".", "-").replace("_", "-")
        model_name = f"{endpoint_name}-{uuid.uuid4()}"[:63]

        create_kwargs: Dict[str, Any] = {
            "modelName": model_name,
            "modelSourceConfig": {"s3DataSource": {"s3Uri": model_artifact_path}},
            "roleArn": bedrock_execution_role_arn,
        }

        if self._config.kms_key_id:
            if self._config.kms_key_id.startswith("arn:aws:kms:"):
                kms_arn = self._config.kms_key_id
            else:
                sts_client = boto3.client("sts", region_name=self.region)
                account_id = sts_client.get_caller_identity()["Account"]
                kms_arn = f"arn:aws:kms:{self.region}:{account_id}:key/{self._config.kms_key_id}"
            create_kwargs["modelKmsKeyArn"] = kms_arn

        if tags:
            create_kwargs["modelTags"] = tags

        # Inject escrow URI tag (Bedrock format: lowercase key/value)
        escrow_tag = {
            "key": ESCROW_URI_TAG_KEY,
            "value": _escrow_tag_value(model_artifact_path),
        }
        all_tags = list(create_kwargs.get("modelTags", [])) + [escrow_tag]
        create_kwargs["modelTags"] = all_tags

        try:
            logger.info(f"Creating custom model '{model_name}'...")
            model = bedrock_client.create_custom_model(**create_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create custom model {model_name}: {e}")

        try:
            monitor_model_create(bedrock_client, model, model_name)
        except Exception as e:
            raise RuntimeError(
                f"Custom model '{model_name}' creation failed "
                f"(ARN: {model.get('modelArn', 'unknown')}): {e}"
            )

        result = ModelDeployResult(
            model_arn=model["modelArn"],
            model_name=model_name,
            escrow_uri=model_artifact_path,
            created_at=datetime.now(timezone.utc),
        )

        self.last_model_publish = result
        self._published_models.add(("bedrock", model["modelArn"], model_artifact_path))
        logger.info(f"Custom model created: {model['modelArn']}")

        return result

    def deploy_to_bedrock(
        self,
        model_deploy_result: Optional[ModelDeployResult] = None,
        model_arn: Optional[str] = None,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        pt_units: Optional[int] = None,
        endpoint_name: Optional[str] = None,
    ) -> DeploymentResult:
        """Deploy a published Bedrock custom model to an endpoint.

        Resolution order for model ARN:
        1. model_deploy_result provided -> use its model_arn
        2. model_arn provided -> use directly
        3. self.last_model_publish -> auto-use
        4. None -> raise ValueError

        Args:
            model_deploy_result: Result from create_custom_model().
            model_arn: Direct model ARN (alternative to model_deploy_result).
            deploy_platform: BEDROCK_OD or BEDROCK_PT.
            pt_units: Number of PT units (required for BEDROCK_PT).
            endpoint_name: Endpoint name (auto-generated if None).

        Returns:
            DeploymentResult with endpoint info and model_publish field.
        """
        if model_deploy_result is not None and model_arn is not None:
            raise ValueError(
                "Cannot provide both model_deploy_result and model_arn. "
                "Use model_deploy_result to deploy from a create_custom_model() result, "
                "or model_arn to deploy from a known ARN."
            )

        # Resolve model ARN
        resolved_publish = model_deploy_result
        if model_arn is None and resolved_publish is None:
            resolved_publish = self.last_model_publish
        if resolved_publish is not None:
            model_arn = resolved_publish.model_arn
        if model_arn is None or model_arn == "":
            raise ValueError(
                "No model ARN available. Call create_custom_model() first, "
                "or provide model_deploy_result or model_arn."
            )

        bedrock_client = boto3.client("bedrock", region_name=self.region)

        # Validate model is ready before attempting deployment
        if resolved_publish is not None:
            resolved_publish._bedrock_client = bedrock_client
            model_status = resolved_publish.status
            if model_status == ModelStatus.CREATING:
                logger.info(
                    "Model %s still Creating. Waiting for completion before deploying...",
                    model_arn,
                )
                wait_for_model_ready(bedrock_client, model_arn)
            elif model_status == ModelStatus.FAILED:
                raise ValueError(f"Cannot deploy model '{model_arn}': model status is Failed.")

        if endpoint_name is None:
            if self.method is not None:
                name_format = f"{self.model.value}-{self.method.value}-{self.region}".lower()
            else:
                name_format = f"{self.model.value}-{self.region}".lower()
            endpoint_name = name_format.replace(".", "-").replace("_", "-")

        # Check for existing deployment with same name
        existing_deployment_arn = check_existing_deployment(endpoint_name, deploy_platform)
        attempt_pt_update = False

        if existing_deployment_arn:
            if self.deployment_mode == DeploymentMode.FAIL_IF_EXISTS:
                raise Exception(
                    f"Deployment '{endpoint_name}' already exists on platform {deploy_platform}.\n"
                    f"ARN: {existing_deployment_arn}\n"
                    f"Change deployment_mode to override."
                )
            elif self.deployment_mode == DeploymentMode.UPDATE_IF_EXISTS:
                if deploy_platform != DeployPlatform.BEDROCK_PT:
                    raise Exception(
                        f"UPDATE_IF_EXISTS mode is only supported for Provisioned Throughput deployments.\n"
                        f"Current platform: {deploy_platform}"
                    )
                logger.info(
                    f"UPDATE_IF_EXISTS mode: Will update existing PT deployment '{endpoint_name}' in-place"
                )
                attempt_pt_update = True

        # IAM permission validation for PT update
        if (
            (
                self._config.validation_config is None
                or self._config.validation_config.get("iam", True)
            )
            and existing_deployment_arn
            and attempt_pt_update
        ):
            required_perms = get_required_bedrock_update_permissions(
                deploy_platform, existing_deployment_arn
            )
            errors: List[str] = []
            Validator._validate_calling_role_permissions(
                errors, required_perms, infra=None, region_name=self.region
            )
            if errors:
                raise Exception(
                    f"Cannot update existing PT deployment '{endpoint_name}': Missing permissions.\n"
                    f"{'; '.join(errors)}"
                )

        deployment_arn = None

        # Try PT update if applicable
        if attempt_pt_update:
            assert existing_deployment_arn is not None
            try:
                update_provisioned_throughput_model(
                    existing_deployment_arn, model_arn, endpoint_name
                )
                deployment_arn = existing_deployment_arn
                logger.info(f"Successfully updated existing PT deployment '{endpoint_name}'")
            except Exception as e:
                raise RuntimeError(f"Failed to deploy {endpoint_name} (model {model_arn}): {e}")

        # Create new deployment if needed
        if deployment_arn is None:
            try:
                logger.info(f"Creating deployment for endpoint '{endpoint_name}'...")
                if deploy_platform == DeployPlatform.BEDROCK_PT:
                    deployment = bedrock_client.create_provisioned_model_throughput(
                        modelUnits=pt_units,
                        provisionedModelName=endpoint_name,
                        modelId=model_arn,
                    )
                    deployment_arn = deployment[DEPLOYMENT_ARN_NAME.get(deploy_platform)]
                elif deploy_platform == DeployPlatform.BEDROCK_OD:
                    deployment = bedrock_client.create_custom_model_deployment(
                        modelDeploymentName=endpoint_name,
                        modelArn=model_arn,
                    )
                    deployment_arn = deployment[DEPLOYMENT_ARN_NAME.get(deploy_platform)]
                else:
                    raise ValueError(f"Unsupported Bedrock platform: {deploy_platform}")
            except Exception as e:
                hint = ""
                if resolved_publish:
                    hint = (
                        f"\nThe custom model was already created: ARN={resolved_publish.model_arn}. "
                        f"Retry with: deploy_to_bedrock(model_arn='{resolved_publish.model_arn}')"
                    )
                raise RuntimeError(
                    f"Failed to deploy {endpoint_name} (model {model_arn}): {e}{hint}"
                )

        create_time = datetime.now(timezone.utc)
        endpoint = EndpointInfo(
            platform=deploy_platform,
            endpoint_name=endpoint_name,
            uri=deployment_arn,
            model_artifact_path=(resolved_publish.escrow_uri if resolved_publish else model_arn),
        )

        result = DeploymentResult(
            endpoint=endpoint,
            created_at=create_time,
            model_publish=resolved_publish,
        )

        logger.info(
            f"\nSuccessfully started deploying {endpoint.endpoint_name}: \n"
            f"- Platform: {endpoint.platform}:\n"
            f"- ARN: {endpoint.uri}\n"
            f"- Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"- ETA: Deployment should be completed in about 30-45 minutes"
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Bedrock combined flow (create model + deploy endpoint)
    # ------------------------------------------------------------------

    def _deploy_to_bedrock(
        self,
        model_artifact_path: str,
        deploy_platform: DeployPlatform = DeployPlatform.BEDROCK_OD,
        pt_units: Optional[int] = None,
        endpoint_name: Optional[str] = None,
        execution_role_name: Optional[str] = None,
        skip_model_reuse: bool = False,
    ) -> DeploymentResult:
        # If the path is already a Bedrock model ARN, skip model creation
        is_bedrock_model_arn = (
            model_artifact_path.startswith("arn:aws:bedrock:")
            and ":custom-model/" in model_artifact_path
        )

        if is_bedrock_model_arn:
            logger.info(f"Using existing Bedrock custom model: {model_artifact_path}")
            bedrock_client = boto3.client("bedrock", region_name=self.region)
            resp = bedrock_client.get_custom_model(modelIdentifier=model_artifact_path)
            raw_status = resp.get("modelStatus", "")
            if raw_status == "Creating":
                logger.info("Model %s still Creating. Waiting...", model_artifact_path)
                wait_for_model_ready(bedrock_client, model_artifact_path)
            elif raw_status == "Failed":
                raise ValueError(
                    f"Cannot deploy model '{model_artifact_path}': model status is Failed."
                )
            return self.deploy_to_bedrock(
                model_arn=model_artifact_path,
                deploy_platform=deploy_platform,
                pt_units=pt_units,
                endpoint_name=endpoint_name,
            )

        # Pre-flight: check for existing endpoint before creating the model
        if endpoint_name is None:
            if self.method is not None:
                name_format = f"{self.model}-{self.method.value}-{self.region}".lower()
            else:
                name_format = f"{self.model}-{self.region}".lower()
            resolved_endpoint_name = name_format.replace(".", "-").replace("_", "-")
        else:
            resolved_endpoint_name = endpoint_name

        existing = check_existing_deployment(resolved_endpoint_name, deploy_platform)
        if existing:
            if self.deployment_mode == DeploymentMode.FAIL_IF_EXISTS:
                raise RuntimeError(
                    f"Deployment '{resolved_endpoint_name}' already exists on platform {deploy_platform}.\n"
                    f"ARN: {existing}\n"
                    f"Change deployment_mode to override."
                )
            if (
                self.deployment_mode == DeploymentMode.UPDATE_IF_EXISTS
                and deploy_platform != DeployPlatform.BEDROCK_PT
            ):
                raise Exception(
                    f"UPDATE_IF_EXISTS mode is only supported for Provisioned Throughput deployments.\n"
                    f"Current platform: {deploy_platform}"
                )

        publish_result = self.create_custom_model(
            model_artifact_path=model_artifact_path,
            endpoint_name=endpoint_name,
            execution_role_name=execution_role_name,
            skip_model_reuse=skip_model_reuse,
        )

        return self.deploy_to_bedrock(
            model_deploy_result=publish_result,
            deploy_platform=deploy_platform,
            pt_units=pt_units,
            endpoint_name=endpoint_name,
        )

    # ------------------------------------------------------------------
    # Internal: SageMaker deployment
    # ------------------------------------------------------------------

    def _deploy_to_sagemaker(
        self,
        model_artifact_path: str,
        instance_type: str,
        unit_count: int,
        endpoint_name: Optional[str] = None,
        environment_variables: Optional[Dict[str, Any]] = None,
        execution_role_name: Optional[str] = None,
        skip_model_reuse: bool = False,
    ) -> DeploymentResult:
        if environment_variables:
            validate_sagemaker_environment_variables(
                environment_variables, model=self.model, instance_type=instance_type
            )
            env_vars = environment_variables
        else:
            env_vars = setup_environment_variables()

        if endpoint_name is None:
            if self.method is not None:
                endpoint_name = f"{self.model.value}-{self.method.value}-sagemaker".replace(
                    "_", "-"
                ).lower()
            else:
                endpoint_name = f"{self.model.value}-sagemaker".replace("_", "-").lower()

        iam_client = boto3.client("iam", region_name=self.region)

        if execution_role_name is None:
            try:
                sagemaker_role_arn = create_sagemaker_execution_role(
                    iam_client=iam_client, role_name=SAGEMAKER_EXECUTION_ROLE_NAME
                )["Role"]["Arn"]
            except Exception as e:
                raise Exception(f"Failed to create the default SageMaker IAM Execution Role: {e}")
        else:
            try:
                sagemaker_role_arn = iam_client.get_role(RoleName=execution_role_name)["Role"][
                    "Arn"
                ]
            except Exception as e:
                raise Exception(
                    f"Failed to retrieve user-specified IAM role '{execution_role_name}': {e}"
                )

        sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        model_name = f"{endpoint_name}-model"
        endpoint_config_name = f"{endpoint_name}-config"

        # Check for existing model by escrow tag before creating a new one
        existing = self.find_published_model("sagemaker", model_artifact_path, skip_model_reuse)
        if existing:
            logger.info(
                f"Found existing SageMaker model {existing} for escrow URI "
                f"'{model_artifact_path}'. Reusing instead of creating a duplicate."
            )
            model_arn = existing
            model_name = existing.split("/")[-1]
        else:
            escrow_tag = {
                "Key": ESCROW_URI_TAG_KEY,
                "Value": _escrow_tag_value(model_artifact_path),
            }
            model_arn = create_sagemaker_model(
                region=self.region,
                model_name=model_name,
                model_s3_location=model_artifact_path,
                sagemaker_execution_role_arn=sagemaker_role_arn,
                sagemaker_client=sagemaker_client,
                environment=env_vars,
                deployment_mode=self.deployment_mode,
                tags=[escrow_tag],
            )
            self._published_models.add(("sagemaker", model_arn, model_artifact_path))

        model_deploy = ModelDeployResult(
            model_arn=model_arn,
            model_name=model_name,
            escrow_uri=model_artifact_path,
            created_at=datetime.now(timezone.utc),
        )
        self.last_model_publish = model_deploy

        try:
            endpoint_arn = create_sagemaker_endpoint(
                model_name=model_name,
                endpoint_config_name=endpoint_config_name,
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                sagemaker_client=sagemaker_client,
                initial_instance_count=unit_count,
                deployment_mode=self.deployment_mode,
            )
        except Exception as e:
            raise RuntimeError(
                f"SageMaker endpoint creation failed: {e}\n\n"
                f"The SageMaker model was already created: ARN={model_deploy.model_arn}\n"
                f"Retrieve it with: deployer.last_model_publish"
            ) from e

        create_time = datetime.now(timezone.utc)
        endpoint = EndpointInfo(
            platform=DeployPlatform.SAGEMAKER,
            endpoint_name=endpoint_name,
            uri=endpoint_arn,
            model_artifact_path=model_artifact_path,
        )

        return DeploymentResult(
            endpoint=endpoint, created_at=create_time, model_publish=model_deploy
        )
