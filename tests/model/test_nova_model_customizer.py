import unittest
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import boto3
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.model.nova_model_customizer import NovaModelCustomizer
from amzn_nova_customization_sdk.model.result import (
    EvaluationResult,
    SMTJBatchInferenceResult,
    SMTJEvaluationResult,
)
from amzn_nova_customization_sdk.model.result.job_result import JobStatus
from amzn_nova_customization_sdk.model.result.training_result import TrainingResult
from amzn_nova_customization_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_customization_sdk.util.recipe import RecipePath


class MockRecipePath(RecipePath):
    """Mock RecipePath that acts as a context manager for tests"""

    def __init__(self, path: str):
        super().__init__(path, temp=False)

    def close(self):
        # Override to prevent actual cleanup in tests
        pass


class TestNovaModelCustomizer(unittest.TestCase):
    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.SFT_LORA
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.instance_count = 2

        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            # Mock recipe metadata and templates
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            # Add IAM and SageMaker mocking for validation
            mock_iam = MagicMock()
            mock_iam.get_role.return_value = {
                "Role": {
                    "AssumeRolePolicyDocument": {
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "sagemaker.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            }
                        ]
                    }
                }
            }
            mock_sagemaker = MagicMock()

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                elif service == "iam":
                    return mock_iam
                elif service == "sagemaker":
                    return mock_sagemaker
                return MagicMock()

            mock_client.side_effect = client_side_effect

            self.customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=self.mock_runtime_manager,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

    def test_init_raises_value_error_on_unsupported_region(self):
        with patch("boto3.session.Session") as mock_session:
            type(mock_session.return_value).region_name = PropertyMock(
                return_value="unsupported-region"
            )
            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=self.model,
                    method=self.method,
                    infra=self.mock_runtime_manager,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )
            self.assertIn("unsupported-region", str(context.exception))
            self.assertIn("not supported", str(context.exception))

    def test_set_model_config_variants(self):
        for model in Model:
            mock_infra = create_autospec(SMTJRuntimeManager)

            with (
                patch("boto3.client") as mock_client,
                patch(
                    "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
                ) as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    elif service == "s3":
                        return mock_s3
                    return MagicMock()

                mock_client.side_effect = client_side_effect

                customizer = NovaModelCustomizer(
                    model=model,
                    method=self.method,
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

                self.assertEqual(customizer.model.model_type, model.model_type)
                self.assertEqual(customizer.model.model_path, model.model_path)

    def test_invalid_model_raises_error(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with (
            patch("boto3.client") as mock_client,
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=self.method,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

            self.assertIsNotNone(customizer)

    def test_customizer_attributes(self):
        self.assertEqual(self.customizer.model, Model.NOVA_MICRO)
        self.assertEqual(self.customizer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(self.customizer.data_s3_path, self.data_s3_path)
        self.assertEqual(self.customizer.output_s3_path, self.output_s3_path)
        self.assertEqual(self.customizer.region, "us-east-1")

    def test_model_setter_getter(self):
        """Test that model setter and getter work correctly."""
        # Test initial value
        self.assertEqual(self.customizer.model, Model.NOVA_MICRO)

        # Test setter - change to NOVA_LITE
        self.customizer.model = Model.NOVA_LITE
        self.assertEqual(self.customizer.model, Model.NOVA_LITE)

        # Test setter - change to NOVA_PRO
        self.customizer.model = Model.NOVA_PRO
        self.assertEqual(self.customizer.model, Model.NOVA_PRO)

    def test_multiple_training_methods(self):
        for method in TrainingMethod:
            mock_infra = create_autospec(SMTJRuntimeManager)

            with (
                patch("boto3.client") as mock_client,
                patch(
                    "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
                ) as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    elif service == "s3":
                        return mock_s3
                    return MagicMock()

                mock_client.side_effect = client_side_effect

                customizer = NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=method,
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

                self.assertEqual(customizer.method, method)

    @patch("amzn_nova_customization_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_exists(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
        )

        expected_output = "s3://sagemaker-nova-123456789012-us-east-1/output"
        self.assertEqual(customizer.output_s3_path, expected_output)
        mock_s3.head_bucket.assert_called_once_with(
            Bucket="sagemaker-nova-123456789012-us-east-1"
        )
        mock_s3.create_bucket.assert_not_called()

    @patch("amzn_nova_customization_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_creation_fails(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("Bucket does not exist")
        mock_s3.create_bucket.side_effect = Exception("Access Denied")

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)

        with self.assertRaises(Exception) as context:
            NovaModelCustomizer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
            )

        self.assertIn("Failed to create output bucket", str(context.exception))
        self.assertIn("Access Denied", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_explicit_output_path_bucket_exists(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        custom_output = "s3://my-custom-bucket/my-output/"

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=custom_output,
        )

        self.assertEqual(customizer.output_s3_path, custom_output)
        mock_s3.head_bucket.assert_called_once()
        mock_s3.create_bucket.assert_not_called()

    @patch("amzn_nova_customization_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata")
    @patch("boto3.client")
    def test_explicit_output_path_bucket_does_not_exist(
        self, mock_client, mock_get_hub_metadata, mock_download_s3
    ):
        mock_get_hub_metadata.return_value = {
            "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
            "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
        }
        mock_download_s3.return_value = ({}, {})

        s3_exceptions = boto3.client("s3").exceptions

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_s3 = MagicMock()
        mock_s3.exceptions = s3_exceptions
        mock_s3.head_bucket.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "NoSuchBucket",
                    "Message": "The specified bucket does not exist",
                }
            },
            operation_name="HeadBucket",
        )

        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_client.side_effect = client_side_effect

        mock_infra = create_autospec(SMTJRuntimeManager)
        custom_output = "s3://new-bucket/my-output/"

        customizer = NovaModelCustomizer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=mock_infra,
            data_s3_path=self.data_s3_path,
            output_s3_path=custom_output,
        )

        self.assertEqual(customizer.output_s3_path, custom_output)

        mock_s3.head_bucket.assert_called_once_with(
            Bucket="new-bucket", ExpectedBucketOwner="123456789012"
        )
        mock_s3.create_bucket.assert_called_once_with(Bucket="new-bucket")

    def test_data_mixing_raises_error_on_unsupported_platform(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with (
            patch("boto3.client") as mock_client,
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=TrainingMethod.CPT,
                    infra=mock_infra,  # SMTJ, not SMHP
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                    data_mixing_enabled=True,
                )

            self.assertIn("Data mixing is only supported for", str(context.exception))

    def test_data_mixing_raises_error_on_unsupported_training_method(self):
        mock_infra = create_autospec(SMHPRuntimeManager)

        with (
            patch("boto3.client") as mock_client,
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with self.assertRaises(ValueError) as context:
                NovaModelCustomizer(
                    model=Model.NOVA_MICRO,
                    method=TrainingMethod.RFT_FULL,  # Not supported for datamixing
                    infra=mock_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                    data_mixing_enabled=True,
                )

            self.assertIn("Data mixing is only supported for", str(context.exception))


class TestTrain(TestNovaModelCustomizer):
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_job_name_truncation(
        self, mock_boto_client, mock_uuid, mock_build_and_validate
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "a" * 100  # Very long UUID
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        long_job_name = "b" * 100  # Very long name

        self.customizer.train(job_name=long_job_name)

        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertLessEqual(len(job_config.job_name), 63)

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_sft_basic_success(
        self, mock_boto_client, mock_uuid, mock_build_and_validate
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        expected_job_id = "job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.train(job_name="test-job")

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(
            result.model_artifacts.checkpoint_s3_path,
            "s3://test-bucket/checkpoints/model",
        )
        self.assertEqual(
            result.model_artifacts.output_s3_path, "s3://output-bucket/output"
        )
        self.assertIsInstance(result.started_time, datetime)

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_recipe.yaml")

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_rft_basic_success(
        self, mock_boto_client, mock_uuid, mock_build_and_validate
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        expected_job_id = "job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        self.customizer.method = TrainingMethod.RFT_LORA
        result = self.customizer.train(
            job_name="test-job",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:my-function-name",
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.RFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(
            result.model_artifacts.checkpoint_s3_path,
            "s3://test-bucket/checkpoints/model",
        )
        self.assertEqual(
            result.model_artifacts.output_s3_path, "s3://output-bucket/output"
        )
        self.assertIsInstance(result.started_time, datetime)

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_recipe.yaml")

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_train_runtime_manager_failure(self, mock_build_and_validate):
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.side_effect = Exception("Runtime failure")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertEqual(str(context.exception), "Runtime failure")

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_train_recipe_build_failure(self, mock_build_and_validate):
        mock_build_and_validate.side_effect = Exception("Recipe build failed")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertIn("Recipe build failed", str(context.exception))

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_train_dry_run(self, mock_build_and_validate):
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        result = self.customizer.train(job_name="test-job", dry_run=True)

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_smhp_basic_success(
        self, mock_boto_client, mock_uuid, mock_build_and_validate
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-smhp"
        mock_build_and_validate.return_value = (
            "mock_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-namespace"

        expected_job_id = "smhp-job-123"
        mock_smhp_infra.execute.return_value = expected_job_id

        with (
            patch("boto3.client") as mock_client,
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            smhp_customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=mock_smhp_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

        result = smhp_customizer.train(job_name="test-smhp-job")

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)

        from amzn_nova_customization_sdk.model.result import SMHPTrainingResult

        self.assertIsInstance(result, SMHPTrainingResult)
        self.assertEqual(result.cluster_name, "test-cluster")
        self.assertEqual(result.namespace, "test-namespace")

        mock_smhp_infra.execute.assert_called_once()
        call_args = mock_smhp_infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-smhp-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)


class TestEvaluate(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_evaluate_basic_success(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)
        self.assertTrue(result.eval_output_path.endswith("/output/output.tar.gz"))

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-eval-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_eval_recipe.yaml")
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_evaluate_runtime_manager_failure(
        self, mock_build_and_validate, mock_boto_client
    ):
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()
        self.mock_runtime_manager.execute.side_effect = Exception(
            "Eval runtime failure"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.evaluate(
                job_name="test-eval-job",
                eval_task=EvaluationTask.MMLU,
                model_path="s3://test/model",
            )

        self.assertEqual(str(context.exception), "Eval runtime failure")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_evaluate_dry_run_returns_none(
        self, mock_build_and_validate, mock_boto_client
    ):
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            dry_run=True,
            model_path="s3://test/model",
        )

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_evaluate_with_processor_config(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test-func",
            "preprocessing": {"enabled": True},
            "postprocessing": {"enabled": True},
            "aggregation": "average",
        }

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            processor=processor_config,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_evaluate_does_not_use_customizer_data_for_builtin_tasks(
        self, mock_uuid, mock_build_and_validate, mock_recipe_init, mock_boto_client
    ):
        """Test that built-in eval tasks (MMLU) don't use customizer's data_s3_path"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            None,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        # Set customizer's data_s3_path (from training)
        self.customizer.data_s3_path = "s3://training-data/train.jsonl"

        expected_job_id = "eval-job-builtin"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,  # Built-in task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with None for data_s3_path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertIsNone(init_call_kwargs["data_s3_path"])

    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_evaluate_uses_customizer_data_for_byod_tasks(
        self, mock_uuid, mock_build_and_validate, mock_boto_client, mock_recipe_init
    ):
        """Test that BYOD eval tasks (GEN_QA) use customizer's data_s3_path as fallback"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            "s3://training-data/train.jsonl",
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        # Set customizer's data_s3_path (from training)
        training_data_path = "s3://training-data/train.jsonl"
        self.customizer.data_s3_path = training_data_path

        expected_job_id = "eval-job-byod"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.GEN_QA,  # BYOD task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with training data path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertEqual(init_call_kwargs["data_s3_path"], training_data_path)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_evaluate_smhp_basic_success(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-smhp-uuid"
        mock_build_and_validate.return_value = (
            "mock_eval_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-eval-cluster"
        mock_smhp_infra.namespace = "test-eval-namespace"

        expected_job_id = "smhp-eval-job-456"
        mock_smhp_infra.execute.return_value = expected_job_id

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            with (
                patch(
                    "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
                ) as mock_download_s3,
            ):
                mock_get_hub_metadata.return_value = {
                    "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                    "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
                }
                mock_download_s3.return_value = ({}, {})

                smhp_customizer = NovaModelCustomizer(
                    model=self.model,
                    method=self.method,
                    infra=mock_smhp_infra,
                    data_s3_path=self.data_s3_path,
                    output_s3_path=self.output_s3_path,
                )

        result = smhp_customizer.evaluate(
            job_name="test-smhp-eval-job",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://test/model",
        )

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(result.eval_task, EvaluationTask.MMLU)
        self.assertIsInstance(result.started_time, datetime)

        from amzn_nova_customization_sdk.model.result import SMHPEvaluationResult

        self.assertIsInstance(result, SMHPEvaluationResult)
        self.assertEqual(result.cluster_name, "test-eval-cluster")
        self.assertEqual(result.namespace, "test-eval-namespace")
        self.assertTrue(result.eval_output_path.endswith("/eval-result/"))

        mock_smhp_infra.execute.assert_called_once()
        call_args = mock_smhp_infra.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-smhp-eval-job", job_config.job_name)
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")


class TestDeploy(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    def test_use_deployment_name_if_user_provided(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
            endpoint_name="test-endpoint-name",
        )

        self.assertEqual("test-endpoint-name", result.endpoint.endpoint_name)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    def test_bedrock_role_already_created(
        self, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        self.endpoint_name = "test-endpoint-name"

        mock_bedrock_role_creation.side_effect = Exception(
            "Failed to find or create the BedrockDeployModelExecutionRole:"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_PT,
            )
        self.assertIn(
            "Failed to find or create the BedrockDeployModelExecutionRole:",
            str(context.exception),
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    def test_deploy_bedrock_od_success(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)
        self.assertEqual(result.endpoint.uri, "test:on-demand-deployment:arn")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    def test_deploy_bedrock_pt_success(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "test:provisioned-throughput-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_PT,
        )

        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_PT)
        self.assertEqual(
            result.endpoint.uri, "test:provisioned-throughput-deployment:arn"
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    def test_create_custom_model_failure(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model.side_effect = Exception(
            "Failed to create custom model"
        )
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.BEDROCK_OD,
            )

        self.assertIn("Failed to create custom model", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    def test_auto_generate_deployment_name_if_not_provided(
        self, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_boto_client.return_value = mock_bedrock

        mock_bedrock.get_custom_model.return_value = {"modelStatus": "ACTIVE"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:on-demand-deployment:arn"
        }
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.BEDROCK_OD,
        )

        self.assertEqual(
            "model-nova-micro-trainingmethod-sft-lora-us-east-1",
            result.endpoint.endpoint_name,
        )

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.check_existing_deployment"
    )
    def test_deploy_fails_when_endpoint_exists_and_fail_if_exists_mode(
        self,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test that deploy fails when endpoint exists and deployment_mode=FAIL_IF_EXISTS (default)"""
        # Mock existing deployment found
        mock_check_existing.return_value = (
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/existing"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model.tar.gz",
                endpoint_name="test-endpoint",
            )

        error_msg = str(context.exception)
        self.assertIn("already exists", error_msg)
        self.assertIn("Change deployment_mode", error_msg)

        # Verify we checked for existing deployment
        mock_check_existing.assert_called_once_with(
            "test-endpoint", DeployPlatform.BEDROCK_OD
        )

        # Verify we didn't proceed with deployment
        mock_bedrock_role_creation.assert_not_called()
        mock_monitor.assert_not_called()

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.check_existing_deployment"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.update_provisioned_throughput_model"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.Validator._validate_calling_role_permissions"
    )
    def test_deploy_pt_update_success_uses_existing_arn(
        self,
        mock_validate_perms,
        mock_update_pt,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test PT update success: uses existing ARN, no new deployment created"""
        existing_arn = "test:existing-pt:arn"
        mock_check_existing.return_value = existing_arn
        mock_validate_perms.return_value = None
        mock_update_pt.return_value = None  # Success

        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {
            "modelArn": "test:new-model:arn"
        }
        mock_boto_client.return_value = mock_bedrock
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "test:role:arn"}}

        with (
            patch(
                "amzn_nova_customization_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata_2,
            patch(
                "amzn_nova_customization_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3_2,
        ):
            mock_get_hub_metadata_2.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3_2.return_value = ({}, {})

            pt_customizer = NovaModelCustomizer(
                model=Model.NOVA_LITE,
                method=TrainingMethod.SFT_LORA,
                infra=self.mock_runtime_manager,
                data_s3_path="s3://test-bucket/data.jsonl",
                deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
            )

        result = pt_customizer.deploy(
            model_artifact_path="s3://test-bucket/model.tar.gz",
            deploy_platform=DeployPlatform.BEDROCK_PT,
            pt_units=10,
            endpoint_name="test-pt-endpoint",
        )

        # Verify update was called
        mock_update_pt.assert_called_once_with(
            existing_arn, "test:new-model:arn", "test-pt-endpoint"
        )

        # Verify NO new deployment created
        mock_bedrock.create_provisioned_model_throughput.assert_not_called()

        # Verify result uses existing ARN
        self.assertEqual(result.endpoint.uri, existing_arn)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.check_existing_deployment"
    )
    def test_deploy_succeeds_when_no_existing_endpoint(
        self,
        mock_check_existing,
        mock_monitor,
        mock_bedrock_role_creation,
        mock_boto_client,
    ):
        """Test that deploy succeeds normally when no existing endpoint found"""
        # Mock no existing deployment found
        mock_check_existing.return_value = None

        # Mock bedrock client and responses
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:new-deployment:arn"
        }
        mock_boto_client.return_value = mock_bedrock

        # Mock role creation
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model.tar.gz",
            endpoint_name="new-endpoint",
        )

        # Verify we checked for existing deployment
        mock_check_existing.assert_called_once_with(
            "new-endpoint", DeployPlatform.BEDROCK_OD
        )

        # Verify deployment proceeded successfully
        mock_bedrock_role_creation.assert_called_once()
        mock_monitor.assert_called_once()
        self.assertEqual(result.endpoint.endpoint_name, "new-endpoint")
        self.assertEqual(result.endpoint.uri, "test:new-deployment:arn")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_with_job_result(
        self, mock_extract, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        """Test deploy with job_result parameter"""
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_boto_client.return_value = mock_bedrock
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None
        mock_extract.return_value = "s3://extracted/checkpoint/path"

        # Mock job result
        mock_job_result = MagicMock()
        mock_job_result.model_artifacts.output_s3_path = "s3://output/path"
        mock_job_result.job_id = "test-job-123"
        mock_job_result.get_job_status.return_value = (JobStatus.COMPLETED, "Completed")

        result = self.customizer.deploy(job_result=mock_job_result)

        mock_extract.assert_called_once_with(
            output_s3_path="s3://output/path",
            job_result=mock_job_result,
        )
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.monitor_model_create"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_with_last_job_id(
        self, mock_extract, mock_monitor, mock_bedrock_role_creation, mock_boto_client
    ):
        """Test deploy using last job ID when no parameters provided"""
        mock_bedrock = MagicMock()
        mock_bedrock.create_custom_model.return_value = {"modelArn": "test:model:arn"}
        mock_bedrock.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "test:deployment:arn"
        }
        mock_boto_client.return_value = mock_bedrock
        mock_bedrock_role_creation.return_value = {"Role": {"Arn": "bedrock:role:arn"}}
        mock_monitor.return_value = None
        mock_extract.return_value = "s3://extracted/checkpoint/path"

        # Set last job ID
        self.customizer.job_id = "last-job-456"

        result = self.customizer.deploy()

        mock_extract.assert_called_once_with(
            output_s3_path=self.customizer.output_s3_path, job_id="last-job-456"
        )
        self.assertEqual(result.endpoint.platform, DeployPlatform.BEDROCK_OD)

    @patch("boto3.client")
    def test_deploy_no_checkpoint_path_available(self, mock_boto_client):
        """Test deploy fails when no checkpoint path can be determined"""
        with self.assertRaises(Exception) as context:
            self.customizer.deploy()

        self.assertIn("No model path provided", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
    )
    def test_deploy_extraction_fails(self, mock_extract, mock_boto_client):
        """Test deploy fails when checkpoint extraction fails"""
        mock_extract.side_effect = Exception("Extraction failed")

        mock_job_result = MagicMock()
        mock_job_result.model_artifacts.output_s3_path = "s3://output/path"
        mock_job_result.job_id = "test-job-123"

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(job_result=mock_job_result)

        self.assertIn("Extraction failed", str(context.exception))


class TestBatchInference(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch("uuid.uuid4")
    def test_batch_inference_basic_success(
        self, mock_uuid, mock_build_and_validate, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-inference-uuid"
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "inference-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.batch_inference(
            job_name="test-inference-job",
            input_path="s3://test-bucket/model-s3-path",
            output_s3_path="s3://test-bucket/output",
        )

        self.assertIsInstance(result, SMTJBatchInferenceResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertIsInstance(result.started_time, datetime)
        self.assertTrue(result.inference_output_path.endswith("/output/output.tar.gz"))

        self.mock_runtime_manager.execute.assert_called_once()
        call_args = self.mock_runtime_manager.execute.call_args
        job_config = call_args.kwargs["job_config"]
        self.assertIn("test-inference-job", job_config.job_name)
        self.assertEqual(job_config.data_s3_path, self.data_s3_path)
        self.assertEqual(job_config.output_s3_path, self.output_s3_path)
        self.assertEqual(job_config.recipe_path, "mock_inference_recipe.yaml")
        self.assertEqual(job_config.input_s3_data_type, "S3Prefix")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_batch_inference_runtime_manager_failure(
        self, mock_build_and_validate, mock_boto_client
    ):
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()
        self.mock_runtime_manager.execute.side_effect = Exception(
            "Inference runtime failure"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.batch_inference(
                job_name="test-inference-job",
                input_path="s3://test-bucket/input",
                output_s3_path="s3://test-bucket/inference-data",
            )

        self.assertEqual(str(context.exception), "Inference runtime failure")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    def test_batch_inference_dry_run_returns_none(
        self, mock_build_and_validate, mock_boto_client
    ):
        mock_build_and_validate.return_value = (
            "mock_inference_recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        mock_boto_client.return_value = MagicMock()

        result = self.customizer.batch_inference(
            job_name="test-inference-job",
            input_path="s3://test-bucket/input",
            output_s3_path="s3://test-bucket/inference-data",
            dry_run=True,
        )

        self.assertIsNone(result)
        self.mock_runtime_manager.execute.assert_not_called()


class TestPlatformValidation(TestNovaModelCustomizer):
    """Tests for platform compatibility validation in evaluate() and batch_inference()"""

    def setUp(self):
        super().setUp()
        # Use NOVA_LITE_2 for platform tests (supports both evaluate and batch_inference)
        self.customizer.model = Model.NOVA_LITE_2

    def test_customizer_attributes(self):
        """Override parent test to check NOVA_LITE_2"""
        self.assertEqual(self.customizer.model, Model.NOVA_LITE_2)
        self.assertEqual(self.customizer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(self.customizer.data_s3_path, self.data_s3_path)
        self.assertEqual(self.customizer.output_s3_path, self.output_s3_path)
        self.assertEqual(self.customizer.region, "us-east-1")

    def test_model_setter_getter(self):
        """Override parent test - temporarily set to NOVA_MICRO to test parent logic"""
        # Temporarily set to NOVA_MICRO to test the parent's setter/getter logic
        self.customizer.model = Model.NOVA_MICRO

        # Run parent test logic
        super().test_model_setter_getter()

        # Restore to NOVA_LITE_2 for other tests in this class
        self.customizer.model = Model.NOVA_LITE_2

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    def test_evaluate_platform_mismatch_smhp_checkpoint_smtj_execution(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMTJ execution raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"

        with self.assertRaises(ValueError) as context:
            self.customizer.evaluate(
                job_name="test-eval", eval_task=EvaluationTask.MMLU
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMHP", error_msg)
        self.assertIn("SMTJ", error_msg)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    def test_evaluate_platform_mismatch_smtj_checkpoint_smhp_execution(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMHP execution raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"

        # Create SMHP customizer
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        with self.assertRaises(ValueError) as context:
            self.customizer.evaluate(
                job_name="test-eval", eval_task=EvaluationTask.MMLU
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMTJ", error_msg)
        self.assertIn("SMHP", error_msg)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_evaluate_platform_match_smtj(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMTJ execution succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        result = self.customizer.evaluate(
            job_name="test-eval", eval_task=EvaluationTask.MMLU
        )

        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_evaluate_platform_match_smhp(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMHP execution succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        mock_smhp_infra.execute.return_value = "job-456"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        result = self.customizer.evaluate(
            job_name="test-eval", eval_task=EvaluationTask.MMLU
        )

        self.assertIsNotNone(result)
        mock_smhp_infra.execute.assert_called_once()

    @patch("amzn_nova_customization_sdk.util.logging.logger")
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_evaluate_unknown_checkpoint_platform_logs_warning(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client, mock_logger
    ):
        """Test that unknown checkpoint platform logs warning but continues"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://my-custom-bucket/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        result = self.customizer.evaluate(
            job_name="test-eval", eval_task=EvaluationTask.MMLU
        )

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Cannot determine platform", warning_msg)
        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    def test_batch_inference_platform_mismatch_smhp_to_smtj(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMTJ batch_inference raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"

        with self.assertRaises(ValueError) as context:
            self.customizer.batch_inference(
                job_name="test-inference",
                input_path="s3://test/input",
                output_s3_path="s3://test/output",
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMHP", error_msg)
        self.assertIn("SMTJ", error_msg)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    def test_batch_inference_platform_mismatch_smtj_to_smhp(
        self, mock_resolve, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMHP batch_inference raises ValueError"""
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        with self.assertRaises(ValueError) as context:
            self.customizer.batch_inference(
                job_name="test-inference",
                input_path="s3://test/input",
                output_s3_path="s3://test/output",
            )

        error_msg = str(context.exception)
        self.assertIn("Platform mismatch", error_msg)
        self.assertIn("SMTJ", error_msg)
        self.assertIn("SMHP", error_msg)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_batch_inference_platform_match_smtj(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMTJ checkpoint on SMTJ batch_inference succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-smtj-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-123"

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_batch_inference_platform_match_smhp(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client
    ):
        """Test that SMHP checkpoint on SMHP batch_inference succeeds"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://customer-escrow-123-hp-abc/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )

        # Configure as SMHP
        mock_smhp_infra = create_autospec(SMHPRuntimeManager)
        mock_smhp_infra.cluster_name = "test-cluster"
        mock_smhp_infra.namespace = "test-ns"
        mock_smhp_infra.execute.return_value = "job-456"
        self.customizer.infra = mock_smhp_infra
        self.customizer.platform = Platform.SMHP

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        self.assertIsNotNone(result)
        mock_smhp_infra.execute.assert_called_once()

    @patch("amzn_nova_customization_sdk.util.logging.logger")
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate"
    )
    @patch(
        "amzn_nova_customization_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    @patch("uuid.uuid4")
    def test_batch_inference_unknown_checkpoint_platform_logs_warning(
        self, mock_uuid, mock_resolve, mock_build, mock_boto_client, mock_logger
    ):
        """Test that unknown checkpoint platform logs warning but continues"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid"
        mock_boto_client.return_value = MagicMock()
        mock_resolve.return_value = "s3://my-custom-bucket/checkpoint"
        mock_build.return_value = (
            "recipe.yaml",
            self.output_s3_path,
            self.data_s3_path,
            "image",
        )
        self.mock_runtime_manager.execute.return_value = "job-789"

        result = self.customizer.batch_inference(
            job_name="test-inference",
            input_path="s3://test/input",
            output_s3_path="s3://test/output",
        )

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Cannot determine platform", warning_msg)
        self.assertIsNotNone(result)
        self.mock_runtime_manager.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
