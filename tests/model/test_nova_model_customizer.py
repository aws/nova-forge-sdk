import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import boto3
from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import (
    ModelArtifacts,
)
from amzn_nova_customization_sdk.model.model_enums import (
    DeployPlatform,
    Model,
    TrainingMethod,
)
from amzn_nova_customization_sdk.model.nova_model_customizer import NovaModelCustomizer
from amzn_nova_customization_sdk.model.result import (
    SMTJBatchInferenceResult,
    SMTJEvaluationResult,
)
from amzn_nova_customization_sdk.model.result.training_result import TrainingResult
from amzn_nova_customization_sdk.recipe_config.eval_config import EvaluationTask
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
        ):
            mock_get_execution_role.return_value = (
                "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            )

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

            with patch("boto3.client") as mock_client:
                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service):
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

    def test_set_image_uri_smtj_sft(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

        self.assertTrue(
            customizer.image_uri.startswith(
                "708977205387.dkr.ecr.us-east-1.amazonaws.com"
            )
        )
        self.assertIn("SM-TJ-", customizer.image_uri)
        self.assertIn("SFT-latest", customizer.image_uri)
        self.assertEqual(
            customizer.image_uri,
            "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-fine-tune-repo:SM-TJ-SFT-latest",
        )

    def test_set_image_uri_smtj_eval(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=self.model,
                method=TrainingMethod.EVALUATION,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

        self.assertTrue(
            customizer.image_uri.startswith(
                "708977205387.dkr.ecr.us-east-1.amazonaws.com"
            )
        )
        self.assertIn("SM-TJ-", customizer.image_uri)
        self.assertIn("Eval-latest", customizer.image_uri)
        self.assertEqual(
            customizer.image_uri,
            "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-evaluation-repo:SM-TJ-Eval-latest",
        )

    def test_set_image_uri_smhp(self):
        mock_infra = create_autospec(SMHPRuntimeManager)

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service):
                if service == "sts":
                    return mock_sts
                elif service == "s3":
                    return mock_s3
                return MagicMock()

            mock_client.side_effect = client_side_effect

            customizer = NovaModelCustomizer(
                model=self.model,
                method=self.method,
                infra=mock_infra,
                data_s3_path=self.data_s3_path,
                output_s3_path=self.output_s3_path,
            )

        self.assertIn("SM-HP-", customizer.image_uri)

    def test_invalid_model_raises_error(self):
        mock_infra = create_autospec(SMTJRuntimeManager)

        with patch("boto3.client") as mock_client:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_s3 = MagicMock()
            mock_s3.head_bucket.return_value = {}

            def client_side_effect(service):
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

    def test_multiple_training_methods(self):
        for method in TrainingMethod:
            mock_infra = create_autospec(SMTJRuntimeManager)

            with patch("boto3.client") as mock_client:
                mock_sts = MagicMock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_s3 = MagicMock()
                mock_s3.head_bucket.return_value = {}

                def client_side_effect(service):
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

    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_exists(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service):
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

    @patch("boto3.client")
    def test_auto_generate_output_path_bucket_creation_fails(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("Bucket does not exist")
        mock_s3.create_bucket.side_effect = Exception("Access Denied")

        def client_side_effect(service):
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

    @patch("boto3.client")
    def test_explicit_output_path_bucket_exists(self, mock_client):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()

        def client_side_effect(service):
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

    @patch("boto3.client")
    def test_explicit_output_path_bucket_does_not_exist(self, mock_client):
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

        def client_side_effect(service):
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


class TestValidate(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder._validate_user_input"
    )
    def test_validate_calls_recipe_builder_validation(
        self, mock_validate, mock_boto3_client
    ):
        # Mock AWS clients to avoid auth issues
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
            return {
                "sts": mock_sts,
                "s3": mock_s3,
                "iam": mock_iam,
                "sagemaker": mock_sagemaker,
            }[service]

        mock_boto3_client.side_effect = client_side_effect

        self.customizer.validate(job_name="test-job")
        mock_validate.assert_called_once_with(validation_config=None)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder._validate_user_input"
    )
    def test_validate_with_validation_config(self, mock_validate, mock_boto3_client):
        # Mock AWS clients
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            return {"sts": mock_sts, "s3": mock_s3}[service]

        mock_boto3_client.side_effect = client_side_effect

        validation_config = {"iam": False, "infra": True}
        self.customizer.validation_config = validation_config
        self.customizer.validate(job_name="test-job")
        mock_validate.assert_called_once_with(validation_config=validation_config)

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder._validate_user_input"
    )
    def test_validate_raises_validation_error(self, mock_validate, mock_boto3_client):
        # Mock AWS clients
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            return {"sts": mock_sts, "s3": mock_s3}[service]

        mock_boto3_client.side_effect = client_side_effect

        mock_validate.side_effect = ValueError("Invalid configuration")
        with self.assertRaises(ValueError) as context:
            self.customizer.validate(job_name="test-job")
        self.assertEqual(str(context.exception), "Invalid configuration")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.rft_recipe_builder.RFTRecipeBuilder._validate_user_input"
    )
    def test_validate_rft_method(self, mock_validate, mock_boto3_client):
        # Mock AWS clients
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            return {"sts": mock_sts, "s3": mock_s3}[service]

        mock_boto3_client.side_effect = client_side_effect

        self.customizer.method = TrainingMethod.RFT_LORA
        self.customizer.validate(
            job_name="test-job",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test",
        )
        mock_validate.assert_called_once_with(validation_config=None)

    @patch("boto3.client")
    def test_validate_unsupported_method(self, mock_boto3_client):
        # Mock AWS clients
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_side_effect(service, **kwargs):
            return {"sts": mock_sts, "s3": mock_s3}[service]

        mock_boto3_client.side_effect = client_side_effect

        self.customizer.method = TrainingMethod.EVALUATION
        with self.assertRaises(ValueError) as context:
            self.customizer.validate(job_name="test-job")
        self.assertIn("not yet supported", str(context.exception))


class TestTrain(TestNovaModelCustomizer):
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder.build"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_job_name_truncation(self, mock_boto_client, mock_uuid, mock_build):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "a" * 100  # Very long UUID
        mock_build.return_value = MockRecipePath("mock_recipe.yaml")
        self.mock_runtime_manager.execute.return_value = "job-789"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": "s3://test-bucket/checkpoints/model"},
            "OutputDataConfig": {"S3OutputPath": "s3://output-bucket/output"},
        }
        mock_boto_client.return_value = mock_sagemaker

        long_job_name = "b" * 100  # Very long name

        self.customizer.train(job_name=long_job_name)

        call_kwargs = self.mock_runtime_manager.execute.call_args[1]
        self.assertLessEqual(len(call_kwargs["job_name"]), 63)

    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder.build"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_sft_basic_success(self, mock_boto_client, mock_uuid, mock_build):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = MockRecipePath("mock_recipe.yaml")

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
        call_kwargs = self.mock_runtime_manager.execute.call_args[1]
        self.assertIn("test-job", call_kwargs["job_name"])
        self.assertEqual(call_kwargs["data_s3_path"], self.data_s3_path)
        self.assertEqual(call_kwargs["output_s3_path"], self.output_s3_path)
        self.assertEqual(call_kwargs["recipe"], "mock_recipe.yaml")

    @patch(
        "amzn_nova_customization_sdk.recipe_builder.rft_recipe_builder.RFTRecipeBuilder.build"
    )
    @patch("uuid.uuid4")
    @patch("boto3.client")
    def test_train_rft_basic_success(self, mock_boto_client, mock_uuid, mock_build):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-uuid-1234"
        mock_build.return_value = MockRecipePath("mock_recipe.yaml")

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
        call_kwargs = self.mock_runtime_manager.execute.call_args[1]
        self.assertIn("test-job", call_kwargs["job_name"])
        self.assertEqual(call_kwargs["data_s3_path"], self.data_s3_path)
        self.assertEqual(call_kwargs["output_s3_path"], self.output_s3_path)
        self.assertEqual(call_kwargs["recipe"], "mock_recipe.yaml")

    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder.build"
    )
    def test_train_runtime_manager_failure(self, mock_build):
        mock_build.return_value = MockRecipePath("mock_recipe.yaml")
        self.mock_runtime_manager.execute.side_effect = Exception("Runtime failure")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertEqual(str(context.exception), "Runtime failure")

    @patch(
        "amzn_nova_customization_sdk.recipe_builder.sft_recipe_builder.SFTRecipeBuilder.build"
    )
    def test_train_recipe_build_failure(self, mock_build):
        mock_build.side_effect = Exception("Recipe build failed")

        with self.assertRaises(Exception) as context:
            self.customizer.train(job_name="test-job")

        self.assertIn("Recipe build failed", str(context.exception))


class TestEvaluate(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.eval_recipe_builder.EvalRecipeBuilder.build"
    )
    @patch("uuid.uuid4")
    def test_evaluate_basic_success(self, mock_uuid, mock_build, mock_boto_client):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-eval-uuid"
        mock_build.return_value = MockRecipePath("mock_eval_recipe.yaml")
        mock_boto_client.return_value = MagicMock()

        expected_job_id = "eval-job-123"
        self.mock_runtime_manager.execute.return_value = expected_job_id

        result = self.customizer.evaluate(
            job_name="test-eval-job",
            eval_task=EvaluationTask.MMLU,
            data_s3_path="s3://test-bucket/eval-data",
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, expected_job_id)
        self.assertEqual(result.eval_task, EvaluationTask.MMLU)
        self.assertIsInstance(result.started_time, datetime)
        self.assertTrue(result.eval_output_path.endswith("/output/output.tar.gz"))

        self.mock_runtime_manager.execute.assert_called_once()
        call_kwargs = self.mock_runtime_manager.execute.call_args[1]
        self.assertIn("test-eval-job", call_kwargs["job_name"])
        self.assertEqual(call_kwargs["data_s3_path"], "s3://test-bucket/eval-data")
        self.assertEqual(call_kwargs["output_s3_path"], self.output_s3_path)
        self.assertEqual(call_kwargs["recipe"], "mock_eval_recipe.yaml")
        self.assertEqual(call_kwargs["input_s3_data_type"], "S3Prefix")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.eval_recipe_builder.EvalRecipeBuilder.build"
    )
    def test_evaluate_runtime_manager_failure(self, mock_build, mock_boto_client):
        mock_build.return_value = MockRecipePath("mock_eval_recipe.yaml")
        mock_boto_client.return_value = MagicMock()
        self.mock_runtime_manager.execute.side_effect = Exception(
            "Eval runtime failure"
        )

        with self.assertRaises(Exception) as context:
            self.customizer.evaluate(
                job_name="test-eval-job",
                eval_task=EvaluationTask.MMLU,
                data_s3_path="s3://test-bucket/eval-data",
            )

        self.assertEqual(str(context.exception), "Eval runtime failure")


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


class TestBatchInference(TestNovaModelCustomizer):
    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.batch_inference_recipe_builder.BatchInferenceRecipeBuilder.build"
    )
    @patch("uuid.uuid4")
    def test_batch_inference_basic_success(
        self, mock_uuid, mock_build, mock_boto_client
    ):
        mock_uuid.return_value = MagicMock()
        mock_uuid.return_value.__str__ = lambda x: "test-inference-uuid"
        mock_build.return_value = MockRecipePath("mock_inference_recipe.yaml")
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
        call_kwargs = self.mock_runtime_manager.execute.call_args[1]
        self.assertIn("test-inference-job", call_kwargs["job_name"])
        self.assertEqual(call_kwargs["data_s3_path"], "s3://test-bucket/model-s3-path")
        self.assertEqual(call_kwargs["output_s3_path"], self.output_s3_path)
        self.assertEqual(call_kwargs["recipe"], "mock_inference_recipe.yaml")
        self.assertEqual(call_kwargs["input_s3_data_type"], "S3Prefix")

    @patch("boto3.client")
    @patch(
        "amzn_nova_customization_sdk.recipe_builder.batch_inference_recipe_builder.BatchInferenceRecipeBuilder.build"
    )
    def test_batch_inference_runtime_manager_failure(
        self, mock_build, mock_boto_client
    ):
        mock_build.return_value = MockRecipePath("mock_inference_recipe.yaml")
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


if __name__ == "__main__":
    unittest.main()
