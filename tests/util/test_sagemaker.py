import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import DeploymentMode, Model, Platform
from amzn_nova_forge.core.result.inference_result import (
    SingleInferenceResult,
)
from amzn_nova_forge.core.types import ModelArtifacts
from amzn_nova_forge.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
    SMTJServerlessRuntimeManager,
)
from amzn_nova_forge.util.sagemaker import (
    _get_sagemaker_inference_image,
    _validate_sagemaker_instance_type_for_model_deployment,
    create_sagemaker_endpoint,
    create_sagemaker_model,
    get_model_artifacts,
    invoke_sagemaker_inference,
    setup_environment_variables,
)


class TestSagemaker(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.endpoint_name = "test-endpoint"
        self.request_body = {"inputs": "Test prompt", "stream": False}
        self.region = "us-east-1"
        self.model_name = "test-model"
        self.model_s3_location = "s3://test-bucket/model/"
        self.sagemaker_execution_role_arn = "arn:aws:iam::123456789012:role/test-role"
        self.endpoint_config_name = "test-endpoint-config"
        self.endpoint_name = "test-endpoint"

    def test_create_sagemaker_model_success(self):
        mock_client = MagicMock()
        mock_client.describe_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-east-1:123:model/test-model"
        }

        result = create_sagemaker_model(
            region=self.region,
            model_name=self.model_name,
            model_s3_location=self.model_s3_location,
            sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
            sagemaker_client=mock_client,
        )

        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123:model/test-model")
        mock_client.create_model.assert_called_once()

    def test_create_sagemaker_model_already_exists(self):
        mock_client = MagicMock()
        mock_client.describe_model.return_value = {"ModelName": "test-model"}

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                model_s3_location=self.model_s3_location,
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=mock_client,
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_create_sagemaker_model_invalid_s3(self):
        with self.assertRaises(ValueError):
            create_sagemaker_model(
                region=self.region,
                model_name=self.model_name,
                model_s3_location="invalid-s3-uri",
                sagemaker_execution_role_arn=self.sagemaker_execution_role_arn,
                sagemaker_client=MagicMock(),
            )

    @patch("amzn_nova_forge.util.sagemaker._monitor_endpoint_creation")
    def test_create_sagemaker_endpoint_success(self, mock_monitor):
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.create_endpoint_config.return_value = {"EndpointConfigArn": "test-config-arn"}
        mock_client.create_endpoint.return_value = {"EndpointArn": "test-endpoint-arn"}
        mock_monitor.return_value = "InService"

        result = create_sagemaker_endpoint(
            model_name=self.model_name,
            endpoint_config_name=self.endpoint_config_name,
            endpoint_name=self.endpoint_name,
            instance_type="ml.g5.4xlarge",
            sagemaker_client=mock_client,
        )

        self.assertEqual(result, "test-endpoint-arn")
        mock_client.create_endpoint_config.assert_called_once()
        mock_client.create_endpoint.assert_called_once()

    def test_create_sagemaker_endpoint_already_exists(self):
        mock_client = MagicMock()
        mock_client.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException"}}, ""
        )
        mock_client.describe_endpoint.return_value = {"EndpointName": "test-endpoint"}

        with self.assertRaises(Exception) as ctx:
            create_sagemaker_endpoint(
                model_name=self.model_name,
                endpoint_config_name=self.endpoint_config_name,
                endpoint_name=self.endpoint_name,
                instance_type="ml.g5.4xlarge",
                sagemaker_client=mock_client,
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_get_sagemaker_inference_image_unsupported_region(self):
        with self.assertRaises(ValueError):
            _get_sagemaker_inference_image(region="unsupported_region")

    def test_get_sagemaker_inference_image_supported_region(self):
        self.assertEqual(
            _get_sagemaker_inference_image(region="us-east-1"),
            "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-inference-repo:SM-Inference-latest",
        )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj(self, mock_boto_client):
        job_name = "test-training-job"
        checkpoint_s3_uri = "s3://my-bucket/checkpoints/"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "CheckpointConfig": {"S3Uri": checkpoint_s3_uri},
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJRuntimeManager)
        infra.platform = Platform.SMTJ

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_uri)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_training_job.assert_called_once_with(TrainingJobName=job_name)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_single_rig(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        checkpoint_s3_path = "s3://my-bucket/hyperpod-checkpoints/"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {
                    "EnvironmentConfig": {
                        "S3OutputPath": checkpoint_s3_path,
                    }
                }
            ]
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_cluster.assert_called_once_with(ClusterName=cluster_name)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_multiple_rigs(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {
            "RestrictedInstanceGroups": [
                {"EnvironmentConfig": {"S3OutputPath": "s3://bucket1/path1/"}},
                {"EnvironmentConfig": {"S3OutputPath": "s3://bucket2/path2/"}},
            ]
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_no_rigs(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {"RestrictedInstanceGroups": []}
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_unsupported_platform(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        job_name = "test-job"
        output_s3_path = "s3://my-bucket/output/"

        infra = MagicMock(spec=RuntimeManager)

        with self.assertRaises(ValueError) as context:
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

        self.assertIn("Unsupported platform", str(context.exception))

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj_client_error(self, mock_boto_client):
        job_name = "test-training-job"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound", "Message": "Job not found"}},
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJRuntimeManager)
        infra.platform = Platform.SMTJ

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_client_error(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFound", "Message": "Cluster not found"}},
            operation_name="DescribeCluster",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.platform = Platform.SMHP
        infra.cluster_name = cluster_name

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_serverless_with_model_package(self, mock_boto_client):
        """Serverless jobs return output_model_arn and checkpoint_s3_path from model package."""
        job_name = "test-serverless-job"
        model_package_arn = "arn:aws:sagemaker:us-east-1:123:model-package/group/1"
        output_s3_path = "s3://my-bucket/output/"
        checkpoint_s3_path = "s3://customer-escrow/job/384/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "OutputModelPackageArn": model_package_arn,
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_sagemaker.describe_model_package.return_value = {
            "InferenceSpecification": {
                "Containers": [{"ModelDataSource": {"S3DataSource": {"S3Uri": checkpoint_s3_path}}}]
            }
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        result = get_model_artifacts(job_name=job_name, infra=infra, output_s3_path=output_s3_path)

        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)
        self.assertEqual(result.output_model_arn, model_package_arn)
        mock_sagemaker.describe_model_package.assert_called_once_with(
            ModelPackageName=model_package_arn
        )

    @patch("amzn_nova_forge.util.sagemaker.boto3.client")
    def test_get_model_artifacts_serverless_no_model_package(self, mock_boto_client):
        """Serverless job with no OutputModelPackageArn returns None checkpoint and arn."""
        job_name = "test-serverless-job"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.return_value = {
            "OutputDataConfig": {"S3OutputPath": output_s3_path},
        }
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJServerlessRuntimeManager)
        infra.platform = Platform.SMTJServerless

        result = get_model_artifacts(job_name=job_name, infra=infra, output_s3_path=output_s3_path)

        self.assertIsNone(result.checkpoint_s3_path)
        self.assertIsNone(result.output_model_arn)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_model_package.assert_not_called()

    def test_setup_environment_variables_with_optional_params(self):
        """Test setup_environment_variables with optional parameters"""
        env_vars = setup_environment_variables(
            temperature="0.7",
            top_p="0.9",
            top_k="50",
            max_new_tokens="100",
            logprobs="5",
        )

        self.assertIn("CONTEXT_LENGTH", env_vars)
        self.assertIn("MAX_CONCURRENCY", env_vars)
        self.assertIn("DEFAULT_TEMPERATURE", env_vars)
        self.assertIn("DEFAULT_TOP_P", env_vars)
        self.assertIn("DEFAULT_TOP_K", env_vars)
        self.assertIn("DEFAULT_MAX_NEW_TOKENS", env_vars)
        self.assertIn("DEFAULT_LOGPROBS", env_vars)

    def test_non_streaming_inference(self):
        """Test non-streaming inference invocation"""
        mock_sagemaker_client = MagicMock()
        mock_response = {
            "Body": MagicMock(
                read=lambda: json.dumps(
                    {
                        "id": "test-id",
                        "created": datetime.now().timestamp(),
                        "choices": ["test_result"],
                    }
                ).encode("utf-8")
            )
        }
        mock_sagemaker_client.invoke_endpoint.return_value = mock_response

        result = invoke_sagemaker_inference(
            request_body=self.request_body,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        mock_sagemaker_client.invoke_endpoint.assert_called_once()

        self.assertIsInstance(result, SingleInferenceResult)

        self.assertEqual(
            result.get(),
            {
                "inference_results": {
                    "is_streaming": False,
                    "response": ["test_result"],
                }
            },
        )

    def test_streaming_inference(self):
        """Test streaming inference invocation"""
        mock_sagemaker_client = MagicMock()

        mock_response_json = {
            "ResponseMetadata": {
                "RequestId": "test-request-id",
            },
            "Body": [
                {"PayloadPart": {"Bytes": b"Hello "}},
                {"PayloadPart": {"Bytes": b"world"}},
                {"SomeOtherEvent": "ignored"},
            ],
        }

        mock_sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response_json

        streaming_request = self.request_body.copy()
        streaming_request["stream"] = True

        result = invoke_sagemaker_inference(
            request_body=streaming_request,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        self.assertIsInstance(result, SingleInferenceResult)

        self.assertIsNotNone(result._streaming_response)
        self.assertIsNone(result._nonstreaming_response)

        collected_results = list(result._streaming_response)

        mock_sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once()
        self.assertEqual(collected_results, ["Hello ", "world"])

    def test_streaming_inference_empty_response(self):
        """Test streaming inference with empty response"""
        mock_sagemaker_client = MagicMock()

        mock_response_json = {
            "ResponseMetadata": {
                "RequestId": "test-request-id",
            },
            "Body": [],
        }

        mock_sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response_json

        streaming_request = self.request_body.copy()
        streaming_request["stream"] = True

        result = invoke_sagemaker_inference(
            request_body=streaming_request,
            endpoint_name=self.endpoint_name,
            sagemaker_client=mock_sagemaker_client,
        )

        self.assertIsInstance(result, SingleInferenceResult)

        collected_results = list(result._streaming_response)

        self.assertEqual(collected_results, [])

    def test_invoke_sagemaker_inference_error_handling(self):
        """Test handling of errors"""
        mock_sagemaker_client = MagicMock()
        mock_sagemaker_client.invoke_endpoint.side_effect = RuntimeError("Unexpected error")

        with self.assertRaises(Exception) as context:
            invoke_sagemaker_inference(
                request_body=self.request_body,
                endpoint_name=self.endpoint_name,
                sagemaker_client=mock_sagemaker_client,
            )

    def test_valid_instance_types(self):
        # Test valid instance types for each model (based on SUPPORTED_SMI_CONFIGS)
        valid_test_cases = [
            (Model.NOVA_MICRO, "ml.g5.12xlarge"),
            (Model.NOVA_MICRO, "ml.g5.24xlarge"),
            (Model.NOVA_MICRO, "ml.g6.12xlarge"),
            (Model.NOVA_MICRO, "ml.g6.24xlarge"),
            (Model.NOVA_MICRO, "ml.g6.48xlarge"),
            (Model.NOVA_MICRO, "ml.p5.48xlarge"),
            (Model.NOVA_LITE, "ml.g6.12xlarge"),
            (Model.NOVA_LITE, "ml.g6.24xlarge"),
            (Model.NOVA_LITE, "ml.g6.48xlarge"),
            (Model.NOVA_LITE, "ml.p5.48xlarge"),
            (Model.NOVA_LITE_2, "ml.g6.48xlarge"),
            (Model.NOVA_LITE_2, "ml.p5.48xlarge"),
            (Model.NOVA_PRO, "ml.p5.48xlarge"),
        ]

        for model, instance_type in valid_test_cases:
            _validate_sagemaker_instance_type_for_model_deployment(instance_type, model)

    def test_invalid_instance_types(self):
        # Test invalid instance types for each model
        invalid_test_cases = [
            (Model.NOVA_MICRO, "ml.fake-instance"),
            (Model.NOVA_LITE, "ml.m5.2xlarge"),
            (Model.NOVA_LITE_2, "ml.m5.2xlarge"),
            (Model.NOVA_PRO, "ml.m5.2xlarge"),
        ]

        for model, instance_type in invalid_test_cases:
            with self.assertRaises(ValueError):
                _validate_sagemaker_instance_type_for_model_deployment(instance_type, model)


if __name__ == "__main__":
    unittest.main()
