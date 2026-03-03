import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import boto3
from botocore.exceptions import ClientError

from amzn_nova_forge_sdk.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge_sdk.model.model_config import ModelArtifacts
from amzn_nova_forge_sdk.model.model_enums import (
    DeploymentMode,
    DeployPlatform,
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_forge_sdk.model.nova_model_customizer import (
    NovaModelCustomizer,
)
from amzn_nova_forge_sdk.model.nova_model_customizer_util import (
    collect_all_parameters,
    generate_job_hash,
    get_recipe_directory,
    get_result_file_path,
    load_existing_result,
    matches_job_cache_criteria,
    persist_result,
    should_persist_results,
)
from amzn_nova_forge_sdk.model.result import (
    EvaluationResult,
    SMTJBatchInferenceResult,
    SMTJEvaluationResult,
)
from amzn_nova_forge_sdk.model.result.job_result import JobStatus
from amzn_nova_forge_sdk.model.result.training_result import (
    SMTJTrainingResult,
    TrainingResult,
)
from amzn_nova_forge_sdk.recipe.recipe_config import EvaluationTask
from amzn_nova_forge_sdk.util.recipe import RecipePath
from amzn_nova_forge_sdk.util.sagemaker import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_CONCURRENCY,
)
from amzn_nova_forge_sdk.validation.validator import (
    get_rft_verification_samples,
    should_verify_rft_lambda,
    verify_rft_lambda,
)


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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
                    "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
                    "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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

    @patch("amzn_nova_forge_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata")
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

    @patch("amzn_nova_forge_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata")
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

    @patch("amzn_nova_forge_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata")
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

    @patch("amzn_nova_forge_sdk.util.recipe.download_templates_from_s3")
    @patch("amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata")
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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    def test_train_recipe_build_failure(self, mock_build_and_validate):
        mock_build_and_validate.side_effect = Exception("Recipe build failed")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertIn("Recipe build failed", str(context.exception))

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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

        from amzn_nova_forge_sdk.model.result import SMHPTrainingResult

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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
        "amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,  # Built-in task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with None for data_s3_path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertIsNone(init_call_kwargs["data_s3_path"])

    @patch(
        "amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.__init__",
        return_value=None,
    )
    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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

        self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.GEN_QA,  # BYOD task
            model_path="s3://test/model",
        )

        # Verify RecipeBuilder was initialized with training data path
        init_call_kwargs = mock_recipe_init.call_args[1]
        self.assertEqual(init_call_kwargs["data_s3_path"], training_data_path)

    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
                    "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
                ) as mock_get_hub_metadata,
                patch(
                    "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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

        from amzn_nova_forge_sdk.model.result import SMHPEvaluationResult

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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_sagemaker_execution_role"
    )
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_model_and_endpoint_config"
    )
    def test_deploy_sagemaker_success(
        self, mock_create, mock_sagemaker_role_creation, mock_boto_client
    ):
        mock_sagemaker_role_creation.return_value = {
            "Role": {"Arn": "sagemaker:role:arn"}
        }
        mock_create.return_value = "sagemaker:endpoint:arn"

        result = self.customizer.deploy(
            model_artifact_path="s3://test-bucket/model",
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_instance_type="ml.p5.48xlarge",
        )

        mock_sagemaker_role_creation.assert_called_once()
        mock_create.assert_called_once()
        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_sagemaker_execution_role"
    )
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_model_and_endpoint_config"
    )
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
    )
    def test_deploy_sagemaker_with_job_success(
        self,
        mock_checkpoint_resolution,
        mock_create,
        mock_sagemaker_role_creation,
        mock_boto_client,
    ):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker
        mock_sagemaker_role_creation.return_value = {
            "Role": {"Arn": "sagemaker:role:arn"}
        }
        mock_create.return_value = "sagemaker:endpoint:arn"
        mock_checkpoint_resolution.return_value = "s3://xn---checkpointbucket/ckpt"

        mock_job_result = MagicMock()
        mock_job_result.model_type = Model.NOVA_MICRO

        result = self.customizer.deploy(
            job_result=mock_job_result,
            deploy_platform=DeployPlatform.SAGEMAKER,
            sagemaker_instance_type="ml.p5.48xlarge",
        )

        mock_create.assert_called_with(
            region="us-east-1",
            model_name="nova-micro-sft-lora-sagemaker-model",
            model_s3_location="s3://xn---checkpointbucket/ckpt/",
            sagemaker_execution_role_arn="sagemaker:role:arn",
            endpoint_config_name="nova-micro-sft-lora-sagemaker-config",
            endpoint_name="nova-micro-sft-lora-sagemaker",
            instance_type="ml.p5.48xlarge",
            environment={
                "CONTEXT_LENGTH": DEFAULT_CONTEXT_LENGTH,
                "MAX_CONCURRENCY": DEFAULT_MAX_CONCURRENCY,
            },
            sagemaker_client=mock_sagemaker,
            initial_instance_count=1,
            deployment_mode=DeploymentMode.FAIL_IF_EXISTS,
        )

        self.assertEqual(result.endpoint.platform, DeployPlatform.SAGEMAKER)
        self.assertEqual(result.endpoint.uri, "sagemaker:endpoint:arn")

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_sagemaker_execution_role"
    )
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_model_and_endpoint_config"
    )
    def test_deploy_sagemaker_failure(
        self, mock_create, mock_sagemaker_role_creation, mock_boto_client
    ):
        mock_sagemaker_role_creation.return_value = {
            "Role": {"Arn": "sagemaker:role:arn"}
        }
        mock_create.side_effect = Exception("Failed to create deployment")

        with self.assertRaises(Exception) as context:
            self.customizer.deploy(
                model_artifact_path="s3://test-bucket/model",
                deploy_platform=DeployPlatform.SAGEMAKER,
                sagemaker_instance_type="ml.p5.48xlarge",
            )
        self.assertIn("Failed to create deployment", str(context.exception))

    @patch("boto3.client")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.check_existing_deployment")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.check_existing_deployment")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.update_provisioned_throughput_model"
    )
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.Validator._validate_calling_role_permissions"
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
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata_2,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
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
            unit_count=10,
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.check_existing_deployment")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.create_bedrock_execution_role"
    )
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.monitor_model_create")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer_util.extract_checkpoint_path_from_job_output"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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

    @patch("amzn_nova_forge_sdk.util.logging.logger")
    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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

    @patch("amzn_nova_forge_sdk.util.logging.logger")
    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch(
        "amzn_nova_forge_sdk.model.nova_model_customizer.resolve_model_checkpoint_path"
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

    def test_invoke_inference_no_endpoint_raises_error(self):
        """Test that an error is raised when no endpoint arn or endpoint info is provided"""
        with self.assertRaises(ValueError):
            self.customizer.invoke_inference({"messages": []})

    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.invoke_sagemaker_inference")
    def test_invoke_inference_sagemaker_endpoint(
        self, mock_invoke_sagemaker, mock_boto3_client
    ):
        # Setup
        endpoint_arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"
        request_body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": False,
        }
        mock_runtime_client = MagicMock()
        mock_boto3_client.return_value = mock_runtime_client
        mock_invoke_sagemaker.return_value = "Inference Result"

        # Execute
        result = self.customizer.invoke_inference(request_body, endpoint_arn)

        # Assert
        mock_boto3_client.assert_called_once_with(
            "sagemaker-runtime", region_name="us-east-1"
        )
        mock_invoke_sagemaker.assert_called_once_with(
            request_body, "test-endpoint", mock_runtime_client
        )
        assert result == "Inference Result"

    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.model.nova_model_customizer.invoke_model")
    def test_invoke_inference_bedrock_endpoint(
        self, mock_invoke_model, mock_boto3_client
    ):
        endpoint_arn = (
            "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/test-model"
        )
        request_body = {
            "system": [{"text": "You are an assistant"}],
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        mock_runtime_client = MagicMock()
        mock_boto3_client.return_value = mock_runtime_client
        mock_invoke_model.return_value = "Bedrock Inference Result"

        # Execute
        result = self.customizer.invoke_inference(request_body, endpoint_arn)

        # Assert
        mock_boto3_client.assert_called_once_with(
            "bedrock-runtime", region_name="us-east-1"
        )
        mock_invoke_model.assert_called_once_with(
            model_id=endpoint_arn,
            request_body=request_body,
            bedrock_runtime=mock_runtime_client,
        )
        assert result == "Bedrock Inference Result"


class TestLambdaVerification(unittest.TestCase):
    """Test suite for RFT lambda verification functionality"""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.RFT_LORA
        self.data_s3_path = "s3://test-bucket/data.jsonl"
        self.output_s3_path = "s3://test-bucket/output"

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.instance_count = 2

        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
            patch(
                "amzn_nova_forge_sdk.util.recipe.get_hub_recipe_metadata"
            ) as mock_get_hub_metadata,
            patch(
                "amzn_nova_forge_sdk.util.recipe.download_templates_from_s3"
            ) as mock_download_s3,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            mock_get_hub_metadata.return_value = {
                "SmtjRecipeTemplateS3Uri": "s3://test-bucket/recipe.yaml",
                "SmtjOverrideParamsS3Uri": "s3://test-bucket/overrides.json",
            }
            mock_download_s3.return_value = ({}, {})

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

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

    def test_should_verify_rft_lambda_no_validation_config(self):
        """Test should_verify_rft_lambda returns False when validation_config is None"""
        result = should_verify_rft_lambda(None)
        self.assertFalse(result)

    def test_should_verify_rft_lambda_empty_validation_config(self):
        """Test should_verify_rft_lambda returns False when validation_config is empty"""
        result = should_verify_rft_lambda({})
        self.assertFalse(result)

    def test_should_verify_rft_lambda_boolean_true(self):
        """Test should_verify_rft_lambda returns True when rft_lambda is True"""
        result = should_verify_rft_lambda({"rft_lambda": True})
        self.assertTrue(result)

    def test_should_verify_rft_lambda_boolean_false(self):
        """Test should_verify_rft_lambda returns False when rft_lambda is False"""
        result = should_verify_rft_lambda({"rft_lambda": False})
        self.assertFalse(result)

    def test_should_verify_rft_lambda_dict_enabled_true(self):
        """Test should_verify_rft_lambda returns True when dict config has enabled=True"""
        result = should_verify_rft_lambda(
            {"rft_lambda": {"enabled": True, "samples": 5}}
        )
        self.assertTrue(result)

    def test_should_verify_rft_lambda_dict_enabled_false(self):
        """Test should_verify_rft_lambda returns False when dict config has enabled=False"""
        result = should_verify_rft_lambda(
            {"rft_lambda": {"enabled": False, "samples": 5}}
        )
        self.assertFalse(result)

    def test_should_verify_rft_lambda_dict_no_enabled_key(self):
        """Test should_verify_rft_lambda returns False when dict config has no enabled key"""
        result = should_verify_rft_lambda({"rft_lambda": {"samples": 5}})
        self.assertFalse(result)

    def test_should_verify_rft_lambda_with_other_validation_keys(self):
        """Test should_verify_rft_lambda works correctly with other validation keys present"""
        result = should_verify_rft_lambda(
            {
                "iam": True,
                "infra": True,
                "rft_lambda": True,
            }
        )
        self.assertTrue(result)

    def test_get_rft_verification_samples_no_validation_config(self):
        """Test get_rft_verification_samples returns default when validation_config is None"""
        result = get_rft_verification_samples(None)
        self.assertEqual(result, 10)

    def test_get_rft_verification_samples_empty_validation_config(self):
        """Test get_rft_verification_samples returns default when validation_config is empty"""
        result = get_rft_verification_samples({})
        self.assertEqual(result, 10)

    def test_get_rft_verification_samples_boolean_config(self):
        """Test get_rft_verification_samples returns default when rft_lambda is boolean"""
        result = get_rft_verification_samples({"rft_lambda": True})
        self.assertEqual(result, 10)

    def test_get_rft_verification_samples_dict_with_samples(self):
        """Test get_rft_verification_samples returns specified samples from dict config"""
        result = get_rft_verification_samples(
            {"rft_lambda": {"enabled": True, "samples": 5}}
        )
        self.assertEqual(result, 5)

    def test_get_rft_verification_samples_dict_without_samples(self):
        """Test get_rft_verification_samples returns default when dict has no samples key"""
        result = get_rft_verification_samples({"rft_lambda": {"enabled": True}})
        self.assertEqual(result, 10)

    def test_get_rft_verification_samples_dict_with_custom_samples(self):
        """Test get_rft_verification_samples returns custom sample count"""
        result = get_rft_verification_samples(
            {"rft_lambda": {"enabled": True, "samples": 20}}
        )
        self.assertEqual(result, 20)

    def test_verify_rft_lambda_no_data_s3_path(self):
        """Test verify_rft_lambda raises error when data_s3_path is not set"""
        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=None,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Cannot verify RFT lambda", str(context.exception))
        self.assertIn("data_s3_path is not set", str(context.exception))

    def test_verify_rft_lambda_invalid_s3_path(self):
        """Test verify_rft_lambda raises error for invalid S3 path"""
        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path="invalid-path",
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Invalid S3 path", str(context.exception))
        self.assertIn("s3://bucket-name/path/to/data.jsonl", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_s3_read_failure(self, mock_boto_client):
        """Test verify_rft_lambda handles S3 read failures"""
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("Access Denied")
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Failed to read samples", str(context.exception))
        self.assertIn("Access Denied", str(context.exception))
        self.assertIn("verify the S3 path is correct", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_invalid_json(self, mock_boto_client):
        """Test verify_rft_lambda handles invalid JSON in data file"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"valid": "json"}',
            b"invalid json here",
        ]
        mock_s3.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("Failed to parse JSON", str(context.exception))
        self.assertIn("line 2", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_empty_data_file(self, mock_boto_client):
        """Test verify_rft_lambda handles empty data file"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = []
        mock_s3.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=10,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("No samples found", str(context.exception))
        self.assertIn(
            "ensure the data file contains valid JSONL data", str(context.exception)
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_lambda_invocation_failure(self, mock_boto_client):
        """Test verify_rft_lambda handles lambda invocation failures"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_lambda.invoke.return_value = {"StatusCode": 500}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_missing_reward_field(self, mock_boto_client):
        """Test verify_rft_lambda detects missing reward field in response"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'{"result": "no reward"}'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_invalid_reward_type(self, mock_boto_client):
        """Test verify_rft_lambda detects non-numeric reward values"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"prompt": "test", "completion": "test"}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'{"reward": "not a number"}'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        self.assertIn("RFT lambda verification failed", str(context.exception))

    @patch("boto3.client")
    def test_verify_rft_lambda_success_with_int_reward(self, mock_boto_client):
        """Test verify_rft_lambda succeeds with integer reward"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = (
            b'[{"id": "sample_1", "aggregate_reward_score": 5}]'
        )
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=1,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_success_with_float_reward(self, mock_boto_client):
        """Test verify_rft_lambda succeeds with float reward"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = (
            b'[{"id": "sample_1", "aggregate_reward_score": 3.14}]'
        )
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=1,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_multiple_samples(self, mock_boto_client):
        """Test verify_rft_lambda verifies multiple samples"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test1"}]}',
            b'{"id": "sample_2", "messages": [{"role": "user", "content": "test2"}]}',
            b'{"id": "sample_3", "messages": [{"role": "user", "content": "test3"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 1.0}, {"id": "sample_2", "aggregate_reward_score": 1.0}, {"id": "sample_3", "aggregate_reward_score": 1.0}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Should not raise any exception
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=3,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_limits_samples_read(self, mock_boto_client):
        """Test verify_rft_lambda only reads requested number of samples"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        # Provide more samples than requested
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test1"}]}',
            b'{"id": "sample_2", "messages": [{"role": "user", "content": "test2"}]}',
            b'{"id": "sample_3", "messages": [{"role": "user", "content": "test3"}]}',
            b'{"id": "sample_4", "messages": [{"role": "user", "content": "test4"}]}',
            b'{"id": "sample_5", "messages": [{"role": "user", "content": "test5"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_payload = MagicMock()
        mock_payload.read.return_value = b'[{"id": "sample_1", "aggregate_reward_score": 1.0}, {"id": "sample_2", "aggregate_reward_score": 1.0}]'
        mock_lambda.invoke.return_value = {"StatusCode": 200, "Payload": mock_payload}

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Request only 2 samples
        verify_rft_lambda(
            lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
            sample_count=2,
            data_s3_path=self.data_s3_path,
            region="us-east-1",
            platform=Platform.SMTJ,
        )

    @patch("boto3.client")
    def test_verify_rft_lambda_error_includes_troubleshooting(self, mock_boto_client):
        """Test verify_rft_lambda error messages include troubleshooting guidance"""
        mock_s3 = MagicMock()
        mock_response = MagicMock()
        mock_response["Body"].iter_lines.return_value = [
            b'{"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}',
        ]
        mock_s3.get_object.return_value = mock_response

        mock_lambda = MagicMock()
        mock_lambda.invoke.side_effect = Exception("Lambda timeout")

        def client_side_effect(service, **kwargs):
            if service == "s3":
                return mock_s3
            elif service == "lambda":
                return mock_lambda
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            verify_rft_lambda(
                lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
                sample_count=1,
                data_s3_path=self.data_s3_path,
                region="us-east-1",
                platform=Platform.SMTJ,
            )

        error_msg = str(context.exception)
        self.assertIn("RFT lambda verification failed", error_msg)


if __name__ == "__main__":
    unittest.main()


class TestJobCachingAndPersistence(unittest.TestCase):
    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.method = TrainingMethod.SFT_LORA
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"
        self.temp_dir = tempfile.mkdtemp()

        self.mock_runtime_manager = create_autospec(SMTJRuntimeManager)
        self.mock_runtime_manager.instance_count = 2

        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

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
                generated_recipe_dir=self.temp_dir,
                enable_job_caching=True,
            )
            self.customizer.job_cache_dir = self.temp_dir

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_aws_mocked_customizer(
        self,
        temp_dir,
        runtime_manager_params=None,
        customizer_params=None,
        training_job_status="Completed",
    ):
        """
        Helper method to create a NovaModelCustomizer with AWS mocking for tests that need custom instances.

        Args:
            temp_dir: Temporary directory for generated recipes
            runtime_manager_params: Dict of parameters for SMTJRuntimeManager constructor
            customizer_params: Dict of parameters for NovaModelCustomizer constructor
            training_job_status: Status to return from mock SageMaker describe_training_job

        Returns:
            Tuple of (customizer, mock_sagemaker, client_side_effect)
        """
        import inspect

        # Default parameters
        default_runtime_params = {
            "instance_type": "ml.g5.12xlarge",
            "instance_count": 1,
        }
        default_customizer_params = {
            "model": Model.NOVA_LITE_2,
            "method": TrainingMethod.SFT_LORA,
            "data_s3_path": "s3://test-bucket/data.jsonl",
            "generated_recipe_dir": temp_dir,
        }

        # Merge with provided parameters
        runtime_params = {**default_runtime_params, **(runtime_manager_params or {})}
        customizer_params = {**default_customizer_params, **(customizer_params or {})}

        # Validate parameters using introspection
        runtime_sig = inspect.signature(SMTJRuntimeManager.__init__)
        customizer_sig = inspect.signature(NovaModelCustomizer.__init__)

        # Filter out 'self' parameter and validate
        valid_runtime_params = {
            k: v
            for k, v in runtime_params.items()
            if k in runtime_sig.parameters and k != "self"
        }
        valid_customizer_params = {
            k: v
            for k, v in customizer_params.items()
            if k in customizer_sig.parameters and k != "self"
        }

        # Set up AWS mocks
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}
        mock_s3.create_bucket.return_value = {}
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
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": training_job_status,
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }

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

        # Create runtime manager and customizer with validated parameters
        runtime = create_autospec(SMTJRuntimeManager)
        runtime.instance_count = valid_runtime_params.get("instance_count", 1)
        runtime.instance_type = valid_runtime_params.get(
            "instance_type", "ml.g5.12xlarge"
        )
        valid_customizer_params["infra"] = runtime
        customizer = NovaModelCustomizer(**valid_customizer_params)
        if temp_dir:
            customizer.job_cache_dir = temp_dir

        return customizer, mock_sagemaker, client_side_effect

    def test_init_with_job_caching_enabled(self):
        """Test initialization with job caching enabled"""
        self.assertTrue(self.customizer.enable_job_caching)
        self.assertIsNotNone(self.customizer._job_caching_config)
        self.assertEqual(self.customizer.generated_recipe_dir, self.temp_dir)

    def test_should_persist_results_true_when_job_caching_enabled(self):
        """Test _should_persist_results returns True when enable_job_caching is True"""
        self.assertTrue(should_persist_results(self.customizer))

    def test_should_persist_results_false_when_job_cache_dir_empty(self):
        """Test _should_persist_results returns False when job_cache_dir is empty"""
        self.customizer.enable_job_caching = True
        self.customizer.job_cache_dir = ""
        self.assertFalse(should_persist_results(self.customizer))

    def test_should_persist_results_false_when_job_caching_disabled(self):
        """Test _should_persist_results returns False when enable_job_caching is False"""
        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = (
                self._create_aws_mocked_customizer(
                    temp_dir=self.temp_dir,
                    customizer_params={
                        "enable_job_caching": False,
                    },
                )
            )
            mock_client.side_effect = client_side_effect

        self.assertFalse(should_persist_results(customizer))

    def test_get_recipe_directory_input_types(self):
        """Test _get_recipe_directory with different input types"""
        # Test with directory path
        directory_path = get_recipe_directory(self.customizer.generated_recipe_dir)
        self.assertEqual(directory_path, self.temp_dir)

        # Test with file path
        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            file_path = "/path/to/recipe.yaml"
            customizer, mock_sagemaker, client_side_effect = (
                self._create_aws_mocked_customizer(
                    temp_dir=None, customizer_params={"generated_recipe_dir": file_path}
                )
            )
            mock_client.side_effect = client_side_effect

        directory_path = get_recipe_directory(customizer.generated_recipe_dir)
        self.assertEqual(directory_path, "/path/to")

        # Test with None
        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = (
                self._create_aws_mocked_customizer(
                    temp_dir=None, customizer_params={"generated_recipe_dir": None}
                )
            )
            mock_client.side_effect = client_side_effect

        directory_path = get_recipe_directory(customizer.generated_recipe_dir)
        self.assertIsNone(directory_path)

    def test_generate_job_hash(self):
        """Test _generate_job_hash produces consistent hashes for same parameters and different hashes for different parameters"""
        # Test consistency - same parameters produce same hash
        params = {"epochs": 3, "lr": 0.001, "batch_size": 32}
        hash1 = generate_job_hash(self.customizer, "test-job", "train", **params)
        hash2 = generate_job_hash(self.customizer, "test-job", "train", **params)
        self.assertEqual(hash1, hash2)

        # Test differentiation - different parameters produce different hashes
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        hash4 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 5, "lr": 0.001},
        )
        self.assertNotEqual(hash3, hash4)

        # Test that train-specific parameters like rft_multiturn_infra and
        # validation_data_s3_path produce different hashes when they differ
        hash5 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            validation_data_s3_path="s3://bucket/val1.jsonl",
        )
        hash6 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            validation_data_s3_path="s3://bucket/val2.jsonl",
        )
        self.assertNotEqual(hash5, hash6)

    def test_generate_job_hash_includes_overrides(self):
        """Test _generate_job_hash includes override params in hash by default"""
        # With default config (include_recipe=True), override params are included in hash
        hash1 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        hash2 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 5, "lr": 0.01},
        )
        # Hashes differ because overrides are included
        self.assertNotEqual(hash1, hash2)

        # Same overrides should produce same hash
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "train",
            overrides={"max_epochs": 3, "lr": 0.001},
        )
        self.assertEqual(hash1, hash3)

    def test_get_result_file_path_with_job_caching(self):
        """Test _get_result_file_path with job caching enabled"""
        file_path = get_result_file_path(
            self.customizer,
            "test-job",
            "training",
            overrides={"max_epochs": 3, "lr": 0.001},
        )

        # Should generate a path with short hash of the full segmented hash
        filename = file_path.name
        self.assertTrue(filename.startswith("test-job_training_"))
        self.assertTrue(filename.endswith(".json"))
        # Extract timestamp from last part of filename (before .json)
        filename_parts = filename.replace(".json", "").split("_")
        timestamp_part = filename_parts[-1]  # Last part is the timestamp
        self.assertEqual(
            len(timestamp_part), 17
        )  # YYYYMMDDHHMMSSMMM should be 17 chars
        # Verify it's all digits (valid timestamp format)
        self.assertTrue(timestamp_part.isdigit())

    def test_get_result_file_path_raises_error_when_persistence_disabled(self):
        """Test _get_result_file_path raises error when persistence is disabled"""
        with (
            patch("boto3.client") as mock_client,
            patch("sagemaker.get_execution_role") as mock_get_execution_role,
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

            customizer, mock_sagemaker, client_side_effect = (
                self._create_aws_mocked_customizer(
                    temp_dir=self.temp_dir,
                    customizer_params={"enable_job_caching": False},
                )
            )
            mock_client.side_effect = client_side_effect

        with self.assertRaises(ValueError) as context:
            get_result_file_path(customizer, "test-job", "training")

        self.assertIn(
            "Cannot get result file path when persistence is disabled",
            str(context.exception),
        )

    @patch("boto3.client")
    def test_load_existing_result_success(self, mock_boto_client):
        """Test _load_existing_result successfully loads existing result when hash matches"""

        # Configure the mock to return completed status
        def mock_describe_training_job(*args, **kwargs):
            print(f"describe_training_job called with: {args}, {kwargs}")
            response = {
                "TrainingJobStatus": "Completed",
                "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
                "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
            }
            print(f"Returning response: {response}")
            return response

        mock_boto_client.return_value.describe_training_job.side_effect = (
            mock_describe_training_job
        )

        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        # Create a real job result that can be properly serialized
        job_result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
            sagemaker_client=mock_boto_client.return_value,
        )

        # Persist the result which should add the job cache hash
        persist_result(
            self.customizer,
            job_result,
            "test-job",
            "training",
            overrides={"max_epochs": 3},
        )

        # Try to load the result with same parameters
        print(
            f"Mock configured: {mock_boto_client.return_value.describe_training_job.return_value}"
        )
        loaded_result = load_existing_result(
            self.customizer, "test-job", "training", overrides={"max_epochs": 3}
        )
        print(f"Loaded result: {loaded_result}")

        # Should find and return the result
        self.assertIsNotNone(loaded_result)
        self.assertEqual(loaded_result.job_id, "test-job-123")

        # Try to load with different parameters - should not find it
        loaded_result_different = load_existing_result(
            self.customizer, "test-job", "training", overrides={"max_epochs": 5}
        )
        self.assertIsNone(loaded_result_different)

    def test_load_existing_result_file_not_found(self):
        """Test _load_existing_result returns None when file doesn't exist"""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")
        result = load_existing_result(self.customizer, non_existent_file, "train")
        self.assertIsNone(result)

    def test_persist_result_success(self):
        """Test _persist_result successfully saves result"""
        # Create a real job result object
        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        result = SMTJTrainingResult(
            job_id="test-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
            sagemaker_client=MagicMock(),
        )

        persist_result(
            self.customizer, result, "test-job", "training", overrides={"max_epochs": 3}
        )

        # Verify file was created
        files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(files), 1)

        # Verify file contains job cache hash
        with open(files[0], "r") as f:
            loaded_data = json.load(f)
        self.assertIn("_job_cache_hash", loaded_data)

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_with_job_caching_existing_result(
        self, mock_boto_client, mock_uuid, mock_build
    ):
        """Test train method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = (
            "mock_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
            "mock-image-uri",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        mock_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://test/checkpoint", output_s3_path="s3://test/output"
        )

        existing_result = SMTJTrainingResult(
            job_id="existing-job-123",
            started_time=datetime.now(),
            method=TrainingMethod.SFT_LORA,
            model_type=Model.NOVA_LITE_2,
            model_artifacts=mock_artifacts,
        )

        # Persist the result with same parameters that will be used in train call
        persist_result(
            self.customizer,
            existing_result,
            "test-job",
            "training",
            recipe_path=None,
            overrides={},
            rft_lambda_arn=None,
        )

        # Mock the result to appear completed by configuring the existing mock
        mock_boto_client.return_value.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }

        result = self.customizer.train(job_name="test-job")

        # Should return existing result without calling runtime manager
        self.mock_runtime_manager.execute.assert_not_called()
        self.assertEqual(result.job_id, "existing-job-123")
        self.assertEqual(result.job_id, "existing-job-123")

    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_with_job_caching_no_existing_result(
        self, mock_boto_client, mock_uuid, mock_build
    ):
        """Test train method executes normally when no existing result found"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = (
            "mock_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
            "mock-image-uri",
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

        # Should execute normally and persist result
        self.mock_runtime_manager.execute.assert_called_once()
        self.assertEqual(result.job_id, expected_job_id)

        # Verify result was persisted - check that a file was created
        files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(files), 1)

        # Verify the file contains the expected job_id
        with open(files[0], "r") as f:
            import json

            data = json.load(f)
        self.assertEqual(data["job_id"], expected_job_id)

    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_evaluate_with_job_caching_existing_result(
        self, mock_uuid, mock_build, mock_boto_client
    ):
        """Test evaluate method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build.return_value = (
            "mock_eval_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        existing_result = SMTJEvaluationResult(
            job_id="existing-eval-123",
            started_time=datetime.now(),
            eval_task=EvaluationTask.MMLU,
            eval_output_path="s3://test/eval/output",
        )

        # Persist the result with same parameters that will be used in evaluate call
        persist_result(
            self.customizer,
            existing_result,
            "test-eval-job",
            "evaluation",
            eval_task=EvaluationTask.MMLU,
            model_path=None,
            subtask=None,
            data_s3_path="s3://test-bucket/eval-data",
            recipe_path=None,
            overrides={},
            processor=None,
            rl_env=None,
        )

        # Mock the result to appear completed by mocking the SageMaker API call
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }
        with patch("boto3.client", return_value=mock_sagemaker):
            result = self.customizer.evaluate(
                job_name="test-eval-job",
                eval_task=EvaluationTask.MMLU,
                data_s3_path="s3://test-bucket/eval-data",
            )

            # Should return existing result without calling runtime manager
            self.mock_runtime_manager.execute.assert_not_called()
            self.assertEqual(result.job_id, "existing-eval-123")

    @patch("boto3.client")
    @patch("amzn_nova_forge_sdk.recipe.recipe_builder.RecipeBuilder.build_and_validate")
    @patch("uuid.uuid4")
    def test_batch_inference_with_job_caching_existing_result(
        self, mock_uuid, mock_build, mock_boto_client
    ):
        """Test batch_inference method returns existing result when job cache finds match"""
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-inference-uuid"
        mock_build.return_value = (
            "mock_inference_recipe.yaml",
            "s3://test/output",
            "s3://test/data",
        )
        mock_boto_client.return_value = MagicMock()

        # Create and persist an existing completed result
        existing_result = SMTJBatchInferenceResult(
            job_id="existing-inference-123",
            started_time=datetime.now(),
            inference_output_path="s3://test/inference/output",
        )

        # Persist the result with same parameters that will be used in batch_inference call
        persist_result(
            self.customizer,
            existing_result,
            "test-inference-job",
            "inference",
            input_path="s3://test-bucket/input",
            output_s3_path="s3://test-bucket/output",
            model_path=None,
            endpoint=None,
            recipe_path=None,
            overrides={},
        )

        # Mock the result to appear completed by mocking the SageMaker API call
        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "CheckpointConfig": {"S3Uri": "s3://test/checkpoint"},
            "OutputDataConfig": {"S3OutputPath": "s3://test/output"},
        }
        with patch("boto3.client", return_value=mock_sagemaker):
            result = self.customizer.batch_inference(
                job_name="test-inference-job",
                input_path="s3://test-bucket/input",
                output_s3_path="s3://test-bucket/output",
            )

            # Should return existing result without calling runtime manager
            self.mock_runtime_manager.execute.assert_not_called()
            self.assertEqual(result.job_id, "existing-inference-123")

    def test_job_caching_disabled_by_default(self):
        """Test that job caching is disabled by default"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("boto3.client") as mock_boto,
                patch("sagemaker.get_execution_role") as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                # Test default - job caching disabled by default
                customizer, _, client_side_effect = self._create_aws_mocked_customizer(
                    temp_dir=None,
                    customizer_params={
                        "generated_recipe_dir": None,
                    },
                )
                mock_boto.side_effect = client_side_effect

                # Job caching is disabled by default
                self.assertFalse(customizer.enable_job_caching)
                self.assertFalse(should_persist_results(customizer))

                # Test with enable_job_caching=True - uses temp_dir for job_cache_dir
                customizer_enabled, _, client_side_effect2 = (
                    self._create_aws_mocked_customizer(
                        temp_dir=temp_dir,
                        customizer_params={
                            "generated_recipe_dir": None,
                            "enable_job_caching": True,
                        },
                    )
                )
                mock_boto.side_effect = client_side_effect2

                # With enable_job_caching=True, persistence is enabled
                self.assertTrue(customizer_enabled.enable_job_caching)
                self.assertTrue(should_persist_results(customizer_enabled))

    def test_job_caching_with_failed_jobs(self):
        """Test that failed jobs are not returned by job caching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use helper method to create mocked customizer with failed job status
            with (
                patch("boto3.client") as mock_boto,
                patch("sagemaker.get_execution_role") as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = (
                    self._create_aws_mocked_customizer(
                        temp_dir,
                        training_job_status="Failed",
                        customizer_params={"enable_job_caching": True},
                    )
                )
                mock_boto.side_effect = client_side_effect

                # Create and persist a failed result
                failed_result = SMTJTrainingResult(
                    job_id="failed-job-456",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )

                job_params = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 10},
                }

                # Persist the failed result
                persist_result(
                    customizer, failed_result, "failed-job", "training", **job_params
                )

                # Try to load it - should return None because job failed
                loaded_failed = load_existing_result(
                    customizer, "failed-job", "training", **job_params
                )
                self.assertIsNone(loaded_failed, "Should not return failed job results")

    def test_job_cache_hash_collision_handling(self):
        """Test that hash collisions are handled correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("boto3.client") as mock_boto,
                patch("sagemaker.get_execution_role") as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = (
                    self._create_aws_mocked_customizer(
                        temp_dir,
                        customizer_params={
                            "model": Model.NOVA_LITE_2,
                            "method": TrainingMethod.SFT_LORA,
                            "data_s3_path": "s3://test-bucket/data.jsonl",
                            "generated_recipe_dir": temp_dir,
                            "enable_job_caching": True,
                        },
                    )
                )
                mock_boto.side_effect = client_side_effect

                job_params1 = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 3},
                }

                job_params2 = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 5},  # Different parameter
                }

                # Create and persist first result with job_params1
                result1 = SMTJTrainingResult(
                    job_id="job-1",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )
                persist_result(
                    customizer, result1, "same-job", "training", **job_params1
                )

                # Small delay to ensure different timestamps
                import time

                time.sleep(0.001)

                # Create and persist second result with job_params2 (same job name, different params)
                result2 = SMTJTrainingResult(
                    job_id="job-2",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )
                persist_result(
                    customizer, result2, "same-job", "training", **job_params2
                )

                # Should have 2 files with same job name but different timestamps
                files = list(Path(temp_dir).glob("*.json"))
                self.assertEqual(
                    len(files),
                    2,
                    f"Expected 2 files with same job name but different params, found {len(files)}",
                )

                # Loading with job_params1 should return result1
                loaded_result1 = load_existing_result(
                    customizer, "same-job", "training", **job_params1
                )
                self.assertIsNotNone(
                    loaded_result1, "Should find matching result for job_params1"
                )
                self.assertEqual(loaded_result1.job_id, "job-1")

                # Loading with job_params2 should return result2
                loaded_result2 = load_existing_result(
                    customizer, "same-job", "training", **job_params2
                )
                self.assertIsNotNone(
                    loaded_result2, "Should find matching result for job_params2"
                )
                self.assertEqual(loaded_result2.job_id, "job-2")

    def test_complete_job_caching_workflow(self):
        """Test complete job caching workflow with real JobResult objects"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use helper method to create mocked customizer
            with (
                patch("boto3.client") as mock_boto,
                patch("sagemaker.get_execution_role") as mock_get_execution_role,
            ):
                mock_get_execution_role.return_value = (
                    "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
                )

                customizer, mock_sagemaker, client_side_effect = (
                    self._create_aws_mocked_customizer(
                        temp_dir, customizer_params={"enable_job_caching": True}
                    )
                )
                mock_boto.side_effect = client_side_effect

                # Step 1: Create and persist a training result
                original_result = SMTJTrainingResult(
                    job_id="original-job-123",
                    started_time=datetime.now(),
                    method=TrainingMethod.SFT_LORA,
                    model_type=Model.NOVA_LITE_2,
                    model_artifacts=ModelArtifacts(
                        checkpoint_s3_path="s3://test/checkpoint",
                        output_s3_path="s3://test/output",
                    ),
                    sagemaker_client=mock_sagemaker,
                )

                # Persist the result with specific parameters
                job_params = {
                    "data_s3_path": "s3://test-bucket/data.jsonl",
                    "instance_type": "ml.g5.12xlarge",
                    "instance_count": 1,
                    "method": TrainingMethod.SFT_LORA,
                    "model": Model.NOVA_LITE_2,
                    "overrides": {"max_epochs": 3},
                }

                persist_result(
                    customizer, original_result, "test-job", "training", **job_params
                )

                # Verify file was created
                files = list(Path(temp_dir).glob("*.json"))
                self.assertEqual(
                    len(files),
                    1,
                    f"Expected 1 file after persistence, found {len(files)}",
                )

                # Step 2: Try to load existing result with same parameters
                loaded_result = load_existing_result(
                    customizer, "test-job", "training", **job_params
                )

                self.assertIsNotNone(
                    loaded_result,
                    "Should have found existing result with matching parameters",
                )
                self.assertEqual(
                    loaded_result.job_id,
                    "original-job-123",
                    f"Expected job_id 'original-job-123', got '{loaded_result.job_id}'",
                )
                self.assertTrue(
                    hasattr(loaded_result, "_job_cache_hash"),
                    "Loaded result should have job cache hash",
                )

                # Step 3: Try to load with different parameters (should not find)
                different_params = job_params.copy()
                different_params["overrides"] = {"max_epochs": 5}  # Different parameter

                loaded_different = load_existing_result(
                    customizer, "test-job", "training", **different_params
                )
                self.assertIsNone(
                    loaded_different, "Should not find result with different parameters"
                )

    def test_job_caching_checks_all_override_parameters(self):
        """Test that job caching checks all override parameters, not just defaults"""
        # Create hashes with custom override parameter not in DEFAULT_OVERRIDE_PARAMS
        hash1 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )
        hash2 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value2", "lr": 0.001},
        )

        # Should not match because custom_param is different
        self.assertFalse(
            matches_job_cache_criteria(
                self.customizer._job_caching_config, hash1, hash2
            )
        )

        # Create hashes with same custom parameter
        hash3 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )
        hash4 = generate_job_hash(
            self.customizer,
            "test-job",
            "training",
            overrides={"custom_param": "value1", "lr": 0.001},
        )

        # Should match because all parameters are the same
        self.assertTrue(
            matches_job_cache_criteria(
                self.customizer._job_caching_config, hash3, hash4
            )
        )

    def test_matches_criteria_include_infra(self):
        """Test that include_infra=True causes infra differences to reject cache hits."""
        hash1 = "instance_type:aaaa,model:bbbb,method:cccc"
        hash2 = "instance_type:xxxx,model:bbbb,method:cccc"
        config = {**self.customizer._job_caching_config, "include_infra": True}
        self.assertFalse(matches_job_cache_criteria(config, hash1, hash2))
        # Same infra should match
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash1))

    def test_matches_criteria_exclude_params(self):
        """Test that exclude_params removes specific fields from comparison."""
        hash1 = "model:aaaa,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        hash2 = "model:xxxx,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        config = {**self.customizer._job_caching_config, "exclude_params": ["model"]}
        # model differs but is excluded, so should match
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))

    def test_matches_criteria_wildcard_exclude(self):
        """Test that exclude_params=['*'] skips all built-in groups, only checking include_params."""
        hash1 = "model:aaaa,method:bbbb,custom:1111"
        hash2 = "model:xxxx,method:yyyy,custom:1111"
        config = {"exclude_params": ["*"], "include_params": ["custom"]}
        # Everything differs except custom, but wildcard exclude skips built-in checks
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))
        # Different custom should fail even with wildcard exclude
        hash3 = "model:xxxx,method:yyyy,custom:9999"
        self.assertFalse(matches_job_cache_criteria(config, hash1, hash3))

    def test_matches_criteria_include_core_false(self):
        """Test that include_core=False skips core field comparison."""
        hash1 = "model:aaaa,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        hash2 = "model:xxxx,method:bbbb,data_s3_path:cccc,job_type:dddd,model_path:eeee"
        config = {**self.customizer._job_caching_config, "include_core": False}
        # model differs but core is not checked
        self.assertTrue(matches_job_cache_criteria(config, hash1, hash2))

    def test_collect_all_parameters_happy_path(self):
        """Test that collect_all_parameters merges infra, customizer, and job params with correct priority."""
        params = collect_all_parameters(
            self.customizer, "test-job", "training", custom_key="custom_value"
        )
        # Customizer fields present
        self.assertEqual(params["model"], self.customizer.model.value)
        self.assertEqual(params["method"], self.customizer.method.value)
        self.assertEqual(params["data_s3_path"], self.customizer.data_s3_path)
        # Job params present and highest priority
        self.assertEqual(params["custom_key"], "custom_value")
        # Infra fields prefixed with infra_
        self.assertIn("infra_instance_count", params)

    def test_collect_all_parameters_skips_private_and_callable(self):
        """Test that infra private attrs and callables are excluded."""
        self.customizer.infra._private = "secret"
        self.customizer.infra.some_method = lambda: None
        params = collect_all_parameters(self.customizer, "test-job", "training")
        self.assertNotIn("infra__private", params)
        self.assertNotIn("infra_some_method", params)
        # Cleanup
        del self.customizer.infra._private
        del self.customizer.infra.some_method
