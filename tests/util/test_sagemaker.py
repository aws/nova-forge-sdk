import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from amzn_nova_customization_sdk.manager.runtime_manager import (
    RuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.model.model_config import ModelArtifacts
from amzn_nova_customization_sdk.util.sagemaker import get_model_artifacts


class TestSagemaker(unittest.TestCase):
    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
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

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_uri)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_training_job.assert_called_once_with(
            TrainingJobName=job_name
        )

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
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
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertEqual(result.checkpoint_s3_path, checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)
        mock_sagemaker.describe_cluster.assert_called_once_with(
            ClusterName=cluster_name
        )

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
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
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_no_rigs(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/hyperpod-output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.return_value = {"RestrictedInstanceGroups": []}
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.cluster_name = cluster_name

        result = get_model_artifacts(
            job_name=job_name,
            infra=infra,
            output_s3_path=output_s3_path,
        )

        self.assertIsInstance(result, ModelArtifacts)
        self.assertIsNone(result.checkpoint_s3_path)
        self.assertEqual(result.output_s3_path, output_s3_path)

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
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

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smtj_client_error(self, mock_boto_client):
        job_name = "test-training-job"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_training_job.side_effect = ClientError(
            error_response={
                "Error": {"Code": "ResourceNotFound", "Message": "Job not found"}
            },
            operation_name="DescribeTrainingJob",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMTJRuntimeManager)

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )

    @patch("amzn_nova_customization_sdk.util.sagemaker.boto3.client")
    def test_get_model_artifacts_smhp_client_error(self, mock_boto_client):
        job_name = "test-hyperpod-job"
        cluster_name = "test-cluster"
        output_s3_path = "s3://my-bucket/output/"

        mock_sagemaker = MagicMock()
        mock_sagemaker.describe_cluster.side_effect = ClientError(
            error_response={
                "Error": {"Code": "ResourceNotFound", "Message": "Cluster not found"}
            },
            operation_name="DescribeCluster",
        )
        mock_boto_client.return_value = mock_sagemaker

        infra = MagicMock(spec=SMHPRuntimeManager)
        infra.cluster_name = cluster_name

        with self.assertRaises(ClientError):
            get_model_artifacts(
                job_name=job_name,
                infra=infra,
                output_s3_path=output_s3_path,
            )


if __name__ == "__main__":
    unittest.main()
