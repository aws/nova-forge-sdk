import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from amzn_nova_forge.manager.runtime_manager import (
    _METHOD_TO_SERVERLESS_CONFIG,
    DEFAULT_SMTJ_JOB_MAX_RUNTIME,
    JobConfig,
    SMTJServerlessRuntimeManager,
    _get_hub_content,
)
from amzn_nova_forge.model.model_enums import Model, TrainingMethod


class TestSMTJServerlessRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.model_package_group_name = "test-model-package-group"
        self.mock_group_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "model-package-group/test-model-package-group"
        )
        self.mock_hub_content_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "hub-content/SageMakerPublicHub/Model/nova-textgeneration-lite-v2/1.0.0"
        )

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def _create_manager(self, mock_setup, **kwargs):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name, **kwargs
        )
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        manager.model_package_group_arn = self.mock_group_arn
        return manager

    # --- Initialization tests ---

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_initialization_defaults(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )
        self.assertEqual(
            manager.model_package_group_name, self.model_package_group_name
        )
        self.assertIsNone(manager.instance_type)
        self.assertIsNone(manager.instance_count)
        self.assertFalse(manager.encrypt_inter_container_traffic)
        self.assertIsNone(manager.subnets)
        self.assertIsNone(manager.security_group_ids)
        self.assertEqual(manager.max_job_runtime, DEFAULT_SMTJ_JOB_MAX_RUNTIME)

    @patch.object(SMTJServerlessRuntimeManager, "setup", return_value=None)
    def test_initialization_custom_params(self, mock_setup):
        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name,
            execution_role=self.mock_role,
            kms_key_id="my-kms-key",
            encrypt_inter_container_traffic=True,
            subnets=["subnet-1"],
            security_group_ids=["sg-1"],
            max_job_runtime=3600,
        )
        self.assertEqual(manager.subnets, ["subnet-1"])
        self.assertEqual(manager.security_group_ids, ["sg-1"])
        self.assertTrue(manager.encrypt_inter_container_traffic)
        self.assertEqual(manager.max_job_runtime, 3600)
        self.assertEqual(manager.kms_key_id, "my-kms-key")

    # --- Setup tests ---

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.get_execution_role")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_creates_model_package_group_if_not_exists(
        self,
        mock_boto_session_cls,
        mock_boto_client,
        mock_get_execution_role,
        mock_session_cls,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.exceptions.ClientError = type("ClientError", (Exception,), {})
        mock_client.describe_model_package_group.side_effect = (
            mock_client.exceptions.ClientError()
        )
        mock_client.create_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        mock_get_execution_role.return_value = self.mock_role

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )

        self.assertEqual(manager.region, "us-west-2")
        self.assertEqual(manager.execution_role, self.mock_role)
        self.assertEqual(manager.model_package_group_arn, self.mock_group_arn)
        mock_client.create_model_package_group.assert_called_once_with(
            ModelPackageGroupName=self.model_package_group_name
        )

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.get_execution_role")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_uses_existing_model_package_group(
        self,
        mock_boto_session_cls,
        mock_boto_client,
        mock_get_execution_role,
        mock_session_cls,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = None
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.describe_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        mock_get_execution_role.return_value = self.mock_role

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name
        )

        self.assertEqual(
            manager.region, "us-east-1"
        )  # fallback when region_name is None
        self.assertEqual(manager.model_package_group_arn, self.mock_group_arn)
        mock_client.create_model_package_group.assert_not_called()

    @patch("amzn_nova_forge.manager.runtime_manager.Session")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.session.Session")
    def test_setup_uses_explicit_execution_role(
        self, mock_boto_session_cls, mock_boto_client, mock_session_cls
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = "us-east-1"
        mock_boto_session_cls.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_client.describe_model_package_group.return_value = {
            "ModelPackageGroupArn": self.mock_group_arn
        }
        mock_boto_client.return_value = mock_client

        manager = SMTJServerlessRuntimeManager(
            model_package_group_name=self.model_package_group_name,
            execution_role=self.mock_role,
        )

        self.assertEqual(manager.execution_role, self.mock_role)

    # --- _extract_hyperparameters tests ---

    def test_extract_hyperparameters_flat(self):
        manager = self._create_manager()
        recipe = {"lr": 0.001, "max_steps": 100, "global_batch_size": 8}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(
            result,
            {
                "learning_rate": "0.001",
                "max_steps": "100",
                "global_batch_size": "8",
            },
        )

    def test_extract_hyperparameters_nested(self):
        manager = self._create_manager()
        recipe = {
            "run": {"name": "my-run", "max_steps": 50},
            "trainer": {"lr": 0.01},
        }
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(
            result,
            {
                "name": "my-run",
                "max_steps": "50",
                "learning_rate": "0.01",
            },
        )

    def test_extract_hyperparameters_three_levels(self):
        manager = self._create_manager()
        recipe = {"model": {"peft": {"alpha": 16}}}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(result, {"lora_alpha": "16"})

    def test_extract_hyperparameters_skips_none(self):
        manager = self._create_manager()
        recipe = {"lr": None, "max_steps": 10}
        result = manager._extract_hyperparameters(recipe)
        self.assertEqual(result, {"max_steps": "10"})

    def test_extract_hyperparameters_empty_recipe(self):
        manager = self._create_manager()
        self.assertEqual(manager._extract_hyperparameters({}), {})

    # --- _build_serverless_job_config tests ---

    def test_build_serverless_job_config_sft_lora(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(
            TrainingMethod.SFT_LORA, "arn:model"
        )
        self.assertEqual(
            config,
            {
                "BaseModelArn": "arn:model",
                "AcceptEula": True,
                "JobType": "FineTuning",
                "CustomizationTechnique": "SFT",
                "Peft": "LORA",
            },
        )

    def test_build_serverless_job_config_sft_full(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(
            TrainingMethod.SFT_FULL, "arn:model"
        )
        self.assertNotIn("Peft", config)
        self.assertEqual(config["CustomizationTechnique"], "SFT")

    def test_build_serverless_job_config_dpo_lora(self):
        manager = self._create_manager()
        config = manager._build_serverless_job_config(
            TrainingMethod.DPO_LORA, "arn:model"
        )
        self.assertEqual(config["CustomizationTechnique"], "DPO")
        self.assertEqual(config["Peft"], "LORA")

    def test_build_serverless_job_config_unsupported_method(self):
        manager = self._create_manager()
        with self.assertRaises(KeyError):
            manager._build_serverless_job_config(TrainingMethod.CPT, "arn:model")

    # --- _resolve_base_model_arn tests ---

    @patch(
        "amzn_nova_forge.manager.runtime_manager._get_hub_content",
        return_value={"HubContentArn": "arn:hub:content"},
    )
    def test_resolve_base_model_arn(self, mock_hub):
        manager = self._create_manager()
        arn = manager._resolve_base_model_arn(Model.NOVA_LITE_2)
        self.assertEqual(arn, "arn:hub:content")
        mock_hub.assert_called_once_with(
            hub_name="SageMakerPublicHub",
            hub_content_name="nova-textgeneration-lite-v2",
            hub_content_type="Model",
            region="us-east-1",
        )

    # --- execute tests ---

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    @patch("amzn_nova_forge.manager.runtime_manager.DataSet")
    def test_execute_success(self, mock_dataset_cls, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}

        mock_dataset = MagicMock()
        mock_dataset.arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:dataset/train-input"
        )
        mock_dataset_cls.create.return_value = mock_dataset

        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {"model_type": "amazon.nova-2-lite-v1:0:256k"},
            "trainer": {"lr": 0.001},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-serverless-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                data_s3_path="s3://input-bucket/data.jsonl",
                method=TrainingMethod.SFT_LORA,
            )
            job_id = manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        self.assertEqual(job_id, "test-job")
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["TrainingJobName"], "test-serverless-job")
        self.assertEqual(call_kwargs["RoleArn"], self.mock_role)
        self.assertIn("ServerlessJobConfig", call_kwargs)
        self.assertEqual(
            call_kwargs["ServerlessJobConfig"]["CustomizationTechnique"], "SFT"
        )
        self.assertEqual(call_kwargs["ServerlessJobConfig"]["Peft"], "LORA")
        self.assertIn("InputDataConfig", call_kwargs)
        self.assertEqual(
            call_kwargs["ModelPackageConfig"]["ModelPackageGroupArn"],
            self.mock_group_arn,
        )

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    def test_execute_without_data_s3_path(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("InputDataConfig", call_kwargs)

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    def test_execute_with_kms_key(self, mock_hub_content):
        manager = self._create_manager(kms_key_id="my-kms-key")
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["OutputDataConfig"]["KmsKeyId"], "my-kms-key")

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    def test_execute_with_vpc_config(self, mock_hub_content):
        manager = self._create_manager(
            subnets=["subnet-abc"], security_group_ids=["sg-123"]
        )
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["VpcConfig"]["Subnets"], ["subnet-abc"])
        self.assertEqual(call_kwargs["VpcConfig"]["SecurityGroupIds"], ["sg-123"])

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    def test_execute_with_mlflow_config(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        }

        recipe = {
            "run": {
                "model_type": "amazon.nova-2-lite-v1:0:256k",
                "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456789012:mlflow/my-server",
                "mlflow_experiment_name": "my-exp",
                "mlflow_run_name": "my-run",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://output-bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            manager.execute(job_config)
        finally:
            os.unlink(recipe_path)

        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(
            call_kwargs["MlflowConfig"]["MlflowResourceArn"],
            "arn:aws:sagemaker:us-east-1:123456789012:mlflow/my-server",
        )
        self.assertEqual(call_kwargs["MlflowConfig"]["MlflowExperimentName"], "my-exp")
        self.assertEqual(call_kwargs["MlflowConfig"]["MlflowRunName"], "my-run")

    def test_execute_fails_without_output_s3_path(self):
        manager = self._create_manager()
        job_config = JobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="/tmp/recipe.yaml",
            output_s3_path=None,
            method=TrainingMethod.SFT_LORA,
        )
        with self.assertRaises(AssertionError):
            manager.execute(job_config)

    def test_execute_fails_without_method(self):
        manager = self._create_manager()
        job_config = JobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="/tmp/recipe.yaml",
            output_s3_path="s3://bucket/output",
            method=None,
        )
        with self.assertRaises(AssertionError):
            manager.execute(job_config)

    @patch("amzn_nova_forge.manager.runtime_manager._get_hub_content")
    def test_execute_raises_on_api_error(self, mock_hub_content):
        manager = self._create_manager()
        mock_hub_content.return_value = {"HubContentArn": self.mock_hub_content_arn}
        manager.sagemaker_client.create_training_job.side_effect = Exception(
            "API error"
        )

        recipe = {"run": {"model_type": "amazon.nova-2-lite-v1:0:256k"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(recipe, f)
            recipe_path = f.name

        try:
            job_config = JobConfig(
                job_name="test-job",
                image_uri="",
                recipe_path=recipe_path,
                output_s3_path="s3://bucket/output",
                method=TrainingMethod.SFT_LORA,
            )
            with self.assertRaises(Exception) as ctx:
                manager.execute(job_config)
            self.assertEqual(str(ctx.exception), "API error")
        finally:
            os.unlink(recipe_path)

    # --- cleanup tests ---

    def test_cleanup_success(self):
        manager = self._create_manager()
        manager.cleanup("test-job")
        manager.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        manager.sagemaker_client.close.assert_called_once()

    def test_cleanup_raises_on_error(self):
        manager = self._create_manager()
        manager.sagemaker_client.stop_training_job.side_effect = Exception(
            "Cleanup failed"
        )
        with self.assertRaises(Exception) as ctx:
            manager.cleanup("test-job")
        self.assertEqual(str(ctx.exception), "Cleanup failed")

    # --- required_calling_role_permissions tests ---

    def test_required_calling_role_permissions(self):
        perms = SMTJServerlessRuntimeManager.required_calling_role_permissions(
            data_s3_path="s3://input/data.jsonl",
            output_s3_path="s3://output/results",
        )
        perm_actions = [p[0] if isinstance(p, tuple) else p for p in perms]
        self.assertIn("sagemaker:CreateTrainingJob", perm_actions)
        self.assertIn("sagemaker:DescribeTrainingJob", perm_actions)
        self.assertIn("iam:PassRole", perm_actions)
        self.assertIn("s3:GetObject", perm_actions)


class TestGetHubContent(unittest.TestCase):
    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_get_hub_content_parses_json_document(self, mock_boto_client):
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            "HubContentArn": "arn:content",
            "HubContentDocument": json.dumps({"key": "value"}),
        }
        mock_boto_client.return_value = mock_client

        result = _get_hub_content("hub", "content", "Model", "us-east-1")
        self.assertEqual(result["HubContentDocument"], {"key": "value"})

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_get_hub_content_leaves_non_json_string(self, mock_boto_client):
        mock_client = MagicMock()
        mock_client.describe_hub_content.return_value = {
            "HubContentArn": "arn:content",
            "HubContentDocument": "not-json",
        }
        mock_boto_client.return_value = mock_client

        result = _get_hub_content("hub", "content", "Model", "us-east-1")
        self.assertEqual(result["HubContentDocument"], "not-json")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3.client")
    def test_get_hub_content_raises_on_error(self, mock_boto_client):
        mock_client = MagicMock()
        mock_client.describe_hub_content.side_effect = Exception("Not found")
        mock_boto_client.return_value = mock_client

        with self.assertRaises(RuntimeError) as ctx:
            _get_hub_content("hub", "content", "Model", "us-east-1")
        self.assertIn("Not found", str(ctx.exception))


class TestMethodToServerlessConfig(unittest.TestCase):
    def test_all_expected_methods_present(self):
        expected = {
            TrainingMethod.SFT_LORA,
            TrainingMethod.SFT_FULL,
            TrainingMethod.DPO_LORA,
            TrainingMethod.DPO_FULL,
            # TrainingMethod.RFT_LORA,
            # TrainingMethod.RFT_FULL,
        }
        self.assertEqual(set(_METHOD_TO_SERVERLESS_CONFIG.keys()), expected)

    def test_lora_methods_have_peft(self):
        for method in (
            TrainingMethod.SFT_LORA,
            TrainingMethod.DPO_LORA,
            # TrainingMethod.RFT_LORA,
        ):
            _, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            self.assertEqual(peft, "LORA")

    def test_full_methods_have_no_peft(self):
        for method in (
            TrainingMethod.SFT_FULL,
            TrainingMethod.DPO_FULL,
            # TrainingMethod.RFT_FULL,
        ):
            _, peft = _METHOD_TO_SERVERLESS_CONFIG[method]
            self.assertIsNone(peft)


if __name__ == "__main__":
    unittest.main()
