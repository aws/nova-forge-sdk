import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    TrainingMethod,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.types import DeploymentResult, EndpointInfo, ForgeConfig
from amzn_nova_forge.deployer.forge_deployer import ForgeDeployer

PATCH_PREFIX = "amzn_nova_forge.deployer.forge_deployer"


@patch(f"{PATCH_PREFIX}.validate_region")
class TestForgeDeployerInit(unittest.TestCase):
    """Constructor tests."""

    def test_happy_path_defaults(self, mock_validate_region):
        deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_MICRO)
        self.assertEqual(deployer.region, "us-east-1")
        self.assertEqual(deployer.model, Model.NOVA_MICRO)
        self.assertEqual(deployer.deployment_mode, DeploymentMode.FAIL_IF_EXISTS)
        self.assertIsNone(deployer.method)
        self.assertIsInstance(deployer._config, ForgeConfig)
        mock_validate_region.assert_called_once_with("us-east-1")

    def test_unsupported_region_raises(self, mock_validate_region):
        mock_validate_region.side_effect = ValueError("not supported")
        with self.assertRaises(ValueError):
            ForgeDeployer(region="invalid-region", model=Model.NOVA_MICRO)

    def test_deployment_mode_default(self, mock_validate_region):
        deployer = ForgeDeployer(region="us-east-1", model=Model.NOVA_LITE)
        self.assertEqual(deployer.deployment_mode, DeploymentMode.FAIL_IF_EXISTS)

    def test_method_stored(self, mock_validate_region):
        deployer = ForgeDeployer(
            region="us-east-1",
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
        )
        self.assertEqual(deployer.method, TrainingMethod.SFT_LORA)


@patch(f"{PATCH_PREFIX}.find_bedrock_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeployBedrock(unittest.TestCase):
    """Tests for deploy() targeting Bedrock platforms."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    # ---- Bedrock OD happy path ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_successful_bedrock_od_deployment(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/my-deploy"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)
        self.assertEqual(
            result.endpoint.uri,
            "arn:aws:bedrock:us-east-1:123456789012:deployment/my-deploy",
        )
        mock_bedrock.create_custom_model.assert_called_once()
        mock_monitor.assert_called_once()

    # ---- Bedrock PT uses create_provisioned_model_throughput ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_bedrock_pt_uses_provisioned_throughput(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }
        mock_bedrock.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:aws:bedrock:us-east-1:123456789012:pt/my-pt"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
            unit_count=2,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_PT)
        mock_bedrock.create_provisioned_model_throughput.assert_called_once()
        call_kwargs = mock_bedrock.create_provisioned_model_throughput.call_args[1]
        self.assertEqual(call_kwargs["modelUnits"], 2)

    # ---- FAIL_IF_EXISTS raises when deployment exists ----

    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:deployment/existing",
    )
    @patch("boto3.client")
    def test_fail_if_exists_raises(
        self,
        mock_boto_client,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_boto_client.return_value = MagicMock()

        deployer = self._make_deployer(deployment_mode=DeploymentMode.FAIL_IF_EXISTS)
        with self.assertRaises(Exception) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )
        self.assertIn("already exists", str(ctx.exception))

    # ---- UPDATE_IF_EXISTS attempts PT update ----

    @patch(f"{PATCH_PREFIX}.update_provisioned_throughput_model")
    @patch(f"{PATCH_PREFIX}.Validator._validate_calling_role_permissions")
    @patch(f"{PATCH_PREFIX}.get_required_bedrock_update_permissions", return_value=[])
    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:pt/existing-pt",
    )
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_update_if_exists_attempts_pt_update(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_get_perms,
        mock_validate_perms,
        mock_update_pt,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
        }

        deployer = self._make_deployer(deployment_mode=DeploymentMode.UPDATE_IF_EXISTS)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
        )

        mock_update_pt.assert_called_once()
        self.assertEqual(
            result.endpoint.uri,
            "arn:aws:bedrock:us-east-1:123456789012:pt/existing-pt",
        )

    # ---- UPDATE_IF_EXISTS fails for non-PT platform ----

    @patch(
        f"{PATCH_PREFIX}.check_existing_deployment",
        return_value="arn:aws:bedrock:us-east-1:123456789012:deployment/existing",
    )
    @patch("boto3.client")
    def test_update_if_exists_fails_for_non_pt(
        self,
        mock_boto_client,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_boto_client.return_value = MagicMock()

        deployer = self._make_deployer(deployment_mode=DeploymentMode.UPDATE_IF_EXISTS)
        with self.assertRaises(Exception) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )
        self.assertIn("UPDATE_IF_EXISTS", str(ctx.exception))
        self.assertIn("Provisioned Throughput", str(ctx.exception))

    # ---- Existing Bedrock model ARN skips model creation ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_existing_bedrock_model_arn_skips_creation(
        self,
        mock_boto_client,
        mock_create_role,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/existing"
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/deploy"
        }

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path=model_arn,
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        mock_bedrock.create_custom_model.assert_not_called()
        self.assertIsInstance(result, DeploymentResult)

    # ---- Endpoint name auto-generation ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_endpoint_name_includes_method(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(method=TrainingMethod.SFT_LORA)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertIn("sft-lora", result.endpoint.endpoint_name)
        self.assertIn("us-east-1", result.endpoint.endpoint_name)

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_endpoint_name_without_method(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer()  # method=None by default
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        # Should NOT contain any method substring
        self.assertNotIn("sft", result.endpoint.endpoint_name)
        self.assertIn("us-east-1", result.endpoint.endpoint_name)

    # ---- KMS key handling ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_kms_key_full_arn_used_directly(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        kms_arn = "arn:aws:kms:us-east-1:123456789012:key/my-key-id"
        config = ForgeConfig(kms_key_id=kms_arn)
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(config=config)
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        create_kwargs = mock_bedrock.create_custom_model.call_args[1]
        self.assertEqual(create_kwargs["modelKmsKeyArn"], kms_arn)

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch(f"{PATCH_PREFIX}.create_bedrock_execution_role")
    @patch("boto3.client")
    def test_kms_key_id_gets_constructed_to_arn(
        self,
        mock_boto_client,
        mock_create_role,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            if service == "sts":
                return mock_sts
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        config = ForgeConfig(kms_key_id="my-key-id")
        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/BedrockRole"}
        }
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer(config=config)
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        create_kwargs = mock_bedrock.create_custom_model.call_args[1]
        expected_arn = "arn:aws:kms:us-east-1:123456789012:key/my-key-id"
        self.assertEqual(create_kwargs["modelKmsKeyArn"], expected_arn)

    # ---- execution_role_name uses get_role ----

    @patch(f"{PATCH_PREFIX}.check_existing_deployment", return_value=None)
    @patch(f"{PATCH_PREFIX}.monitor_model_create")
    @patch("boto3.client")
    def test_execution_role_name_uses_get_role(
        self,
        mock_boto_client,
        mock_monitor,
        mock_check_existing,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_bedrock = MagicMock()
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CustomRole"}
        }

        def client_side_effect(service, **kwargs):
            if service == "bedrock":
                return mock_bedrock
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "arn:aws:bedrock:us-east-1:123456789012:custom-model/m"
        }
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:aws:bedrock:us-east-1:123456789012:deployment/d"
        }

        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
            execution_role_name="CustomRole",
        )

        mock_iam.get_role.assert_called_once_with(RoleName="CustomRole")


@patch(f"{PATCH_PREFIX}.find_sagemaker_model_by_tag", return_value=None)
@patch(f"{PATCH_PREFIX}.validate_region")
class TestDeploySageMaker(unittest.TestCase):
    """Tests for deploy() targeting SageMaker."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch(f"{PATCH_PREFIX}.setup_environment_variables", return_value={"KEY": "VAL"})
    @patch("boto3.client")
    def test_successful_sagemaker_deployment(
        self,
        mock_boto_client,
        mock_setup_env,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-ep-model"
        )
        mock_create_endpoint.return_value = (
            "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-ep"
        )

        deployer = self._make_deployer()
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        self.assertIsInstance(result, DeploymentResult)
        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        mock_create_endpoint.assert_called_once()

    @patch(f"{PATCH_PREFIX}.validate_unit_count")
    def test_sagemaker_instance_type_none_raises(
        self, mock_validate_unit, mock_validate_region, mock_find_by_tag
    ):
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.deploy(
                model_artifact_path="s3://bucket/model",
                deploy_platform=DeployPlatform.SAGEMAKER,
                sagemaker_instance_type=None,
            )
        self.assertIn("sagemaker_instance_type cannot be None", str(ctx.exception))

    @patch(f"{PATCH_PREFIX}.validate_unit_count")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    def test_bedrock_model_arn_raises_for_sagemaker(
        self,
        mock_validate_instance,
        mock_validate_unit,
        mock_validate_region,
        mock_find_by_tag,
    ):
        deployer = self._make_deployer()
        with self.assertRaises(ValueError) as ctx:
            deployer.deploy(
                model_artifact_path="arn:aws:bedrock:us-east-1:123:custom-model/foo",
                deploy_platform=DeployPlatform.SAGEMAKER,
            )
        self.assertIn("Cannot deploy Bedrock-customized models", str(ctx.exception))

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.validate_sagemaker_environment_variables")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch("boto3.client")
    def test_environment_variables_validated(
        self,
        mock_boto_client,
        mock_create_role,
        mock_validate_env,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/ep-model"
        mock_create_endpoint.return_value = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep"

        env_vars = {"MY_VAR": "value"}
        deployer = self._make_deployer()
        deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_environment_variables=env_vars,
        )

        mock_validate_env.assert_called_once_with(
            env_vars, model=Model.NOVA_MICRO, instance_type="ml.p5.48xlarge"
        )

    @patch(f"{PATCH_PREFIX}.create_sagemaker_endpoint")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_model")
    @patch(f"{PATCH_PREFIX}._validate_sagemaker_instance_type_for_model_deployment")
    @patch(f"{PATCH_PREFIX}.create_sagemaker_execution_role")
    @patch(f"{PATCH_PREFIX}.setup_environment_variables", return_value={})
    @patch("boto3.client")
    def test_sagemaker_endpoint_name_includes_method(
        self,
        mock_boto_client,
        mock_setup_env,
        mock_create_role,
        mock_validate_instance,
        mock_create_model,
        mock_create_endpoint,
        mock_validate_region,
        mock_find_by_tag,
    ):
        mock_iam = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        mock_create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMakerRole"}
        }
        mock_create_model.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model/ep-model"
        mock_create_endpoint.return_value = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/ep"

        deployer = self._make_deployer(method=TrainingMethod.SFT_LORA)
        result = deployer.deploy(
            model_artifact_path="s3://bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
        )

        self.assertIn("sft-lora", result.endpoint.endpoint_name)
        self.assertIn("sagemaker", result.endpoint.endpoint_name)


@patch(f"{PATCH_PREFIX}.validate_region")
class TestGetStatus(unittest.TestCase):
    """Tests for get_status and get_status_by_arn."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    def test_get_status_delegates_to_result_status(self, mock_validate_region):
        deployer = self._make_deployer()
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="ep",
            uri="arn:aws:bedrock:us-east-1:123:deployment/d",
            model_artifact_path="s3://bucket/model",
        )
        result = DeploymentResult(endpoint=endpoint, created_at=datetime.now(timezone.utc))

        with patch.object(
            DeploymentResult,
            "status",
            new_callable=lambda: property(lambda self: JobStatus.COMPLETED),
        ):
            status = deployer.get_status(result)
            self.assertEqual(status, JobStatus.COMPLETED)

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value="InProgress")
    def test_get_status_by_arn_returns_job_status(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        status = deployer.get_status_by_arn(
            "arn:aws:bedrock:us-east-1:123:deployment/d", DeployPlatform.BEDROCK_OD
        )
        self.assertEqual(status, JobStatus.IN_PROGRESS)
        mock_check_status.assert_called_once_with(
            "arn:aws:bedrock:us-east-1:123:deployment/d", DeployPlatform.BEDROCK_OD
        )

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value=None)
    def test_get_status_by_arn_returns_none_for_unknown(
        self, mock_check_status, mock_validate_region
    ):
        deployer = self._make_deployer()
        status = deployer.get_status_by_arn(
            "arn:aws:bedrock:us-east-1:123:deployment/d", DeployPlatform.BEDROCK_OD
        )
        self.assertIsNone(status)


@patch(f"{PATCH_PREFIX}.validate_region")
class TestGetLogs(unittest.TestCase):
    """Tests for get_logs."""

    def _make_deployer(self, **kwargs):
        defaults = dict(region="us-east-1", model=Model.NOVA_MICRO)
        defaults.update(kwargs)
        return ForgeDeployer(**defaults)

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value="Completed")
    def test_get_logs_with_job_result(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        endpoint = EndpointInfo(
            platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="ep",
            uri="arn:aws:bedrock:us-east-1:123:deployment/d",
            model_artifact_path="s3://bucket/model",
        )
        result = DeploymentResult(endpoint=endpoint, created_at=datetime.now(timezone.utc))

        deployer.get_logs(job_result=result)
        mock_check_status.assert_called_once_with(
            "arn:aws:bedrock:us-east-1:123:deployment/d", DeployPlatform.BEDROCK_OD
        )

    @patch(f"{PATCH_PREFIX}.check_deployment_status", return_value="InProgress")
    def test_get_logs_with_endpoint_arn_only(self, mock_check_status, mock_validate_region):
        deployer = self._make_deployer()
        deployer.get_logs(endpoint_arn="arn:aws:sagemaker:us-east-1:123:endpoint/ep")
        mock_check_status.assert_called_once_with(
            "arn:aws:sagemaker:us-east-1:123:endpoint/ep", DeployPlatform.SAGEMAKER
        )

    @patch(f"{PATCH_PREFIX}.logger")
    def test_get_logs_no_arn_logs_info(self, mock_logger, mock_validate_region):
        deployer = self._make_deployer()
        deployer.get_logs()
        mock_logger.info.assert_called_once_with("No endpoint ARN available. Call deploy() first.")


if __name__ == "__main__":
    unittest.main()
