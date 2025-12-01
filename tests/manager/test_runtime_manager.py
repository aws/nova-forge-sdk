import json
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_customization_sdk.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    HYPERPOD_RECIPE_PATH,
)


class TestSMTJRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1

    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def _create_manager(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        manager.execution_role = self.mock_role
        manager.sagemaker_client = MagicMock()
        manager.sagemaker_session = MagicMock()
        manager.region = "us-east-1"
        return manager

    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def test_initialization(self, mock_setup):
        manager = SMTJRuntimeManager(self.instance_type, self.instance_count)
        manager.execution_role = self.mock_role
        manager.region = "us-east-1"

        self.assertEqual(manager.instance_type, self.instance_type)
        self.assertEqual(manager.instance_count, self.instance_count)
        self.assertEqual(manager.region, "us-east-1")

    @patch(
        "amzn_nova_customization_sdk.manager.runtime_manager.sagemaker.session.Session"
    )
    @patch(
        "amzn_nova_customization_sdk.manager.runtime_manager.sagemaker.get_execution_role"
    )
    @patch("amzn_nova_customization_sdk.manager.runtime_manager.boto3.client")
    @patch("amzn_nova_customization_sdk.manager.runtime_manager.boto3.session.Session")
    def test_setup_(
        self,
        mock_boto_session_class,
        mock_boto_client,
        mock_get_execution_role,
        mock_sagemaker_session_class,
    ):
        mock_boto_session = MagicMock()
        mock_boto_session.region_name = None
        mock_boto_session_class.return_value = mock_boto_session

        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        mock_role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        mock_get_execution_role.return_value = mock_role

        mock_sagemaker_session = MagicMock()
        mock_sagemaker_session_class.return_value = mock_sagemaker_session

        manager = SMTJRuntimeManager("ml.m5.xlarge", 1)

        self.assertEqual(manager.region, "us-east-1")
        self.assertEqual(manager.sagemaker_client, mock_client)
        self.assertEqual(manager.execution_role, mock_role)
        self.assertEqual(manager.sagemaker_session, mock_sagemaker_session)

        mock_boto_session_class.assert_called_once()
        mock_boto_client.assert_called_once_with("sagemaker", region_name="us-east-1")
        mock_get_execution_role.assert_called_once_with(use_default=True)
        mock_sagemaker_session_class.assert_called_once_with(
            boto_session=mock_boto_session, sagemaker_client=mock_client
        )

    @patch("amzn_nova_customization_sdk.manager.runtime_manager.PyTorch")
    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def test_execute_success(self, mock_setup, mock_pytorch):
        manager = self._create_manager()

        mock_estimator = MagicMock()
        mock_pytorch.return_value = mock_estimator
        mock_estimator.fit.return_value = None

        manager._get_training_job_arn = MagicMock(
            return_value="arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job"
        )

        job_id = manager.execute(
            job_name="test-job",
            data_s3_path="s3://input-bucket/data",
            output_s3_path="s3://output-bucket/output",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            recipe="/path/to/recipe",
            input_s3_data_type="data_type",
        )

        mock_pytorch.assert_called_once()
        mock_estimator.fit.assert_called_once()
        self.assertEqual(job_id, "test-job")

    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def test_cleanup_success(self, mock_setup):
        manager = self._create_manager()
        mock_client = manager.sagemaker_client

        manager.cleanup("test-job")

        mock_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )
        mock_client.close.assert_called_once()

    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def test_cleanup_handles_error(self, mock_setup):
        manager = self._create_manager()
        mock_client = manager.sagemaker_client
        mock_client.stop_training_job.side_effect = Exception("Cleanup failed")

        with self.assertRaises(Exception) as context:
            manager.cleanup("test-job")

        self.assertEqual(str(context.exception), "Cleanup failed")
        mock_client.stop_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )

    @patch("amzn_nova_customization_sdk.manager.runtime_manager.PyTorch")
    @patch.object(SMTJRuntimeManager, "_setup", return_value=None)
    def test_execute_handles_error(self, mock_setup, mock_pytorch):
        manager = self._create_manager()

        mock_estimator = MagicMock()
        mock_estimator.fit.side_effect = Exception("Training failed")
        mock_pytorch.return_value = mock_estimator

        with self.assertRaises(Exception) as context:
            manager.execute(
                job_name="test-job",
                data_s3_path="s3://input-bucket/data",
                output_s3_path="s3://output-bucket/output",
                image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
                recipe="/path/to/recipe",
                input_s3_data_type="data_type",
            )

        self.assertEqual(str(context.exception), "Training failed")
        mock_pytorch.assert_called_once()
        mock_estimator.fit.assert_called_once()


class TestSMHPRuntimeManager(unittest.TestCase):
    def setUp(self):
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1
        self.cluster_name = "test-cluster"
        self.namespace = "test-namespace"

    @patch("subprocess.run")
    def test_initialization(self, mock_run):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "connect-cluster",
                "--cluster-name",
                self.cluster_name,
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(manager.instance_type, self.instance_type)
        self.assertEqual(manager.instance_count, self.instance_count)
        self.assertEqual(manager.cluster_name, self.cluster_name)
        self.assertEqual(manager.namespace, self.namespace)

    @patch("subprocess.run")
    def test_initialization_fails(self, mock_run):
        mock_run.return_value.stderr = "Connection failed"

        with self.assertRaises(Exception):
            SMHPRuntimeManager(
                self.instance_type,
                self.instance_count,
                self.cluster_name,
                self.namespace,
            )

    @patch("subprocess.run")
    def test_execute_success(self, mock_run):
        mock_run.return_value.stdout = "NAME: test-job-123"
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        mock_run.reset_mock()  # Reset call count after initialization

        image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest"

        job_id = manager.execute(
            job_name="test-job",
            data_s3_path="s3://input-bucket/data",
            output_s3_path="s3://output-bucket/output",
            image_uri=image_uri,
            recipe=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
            input_s3_data_type=None,
        )

        override_parameters = json.dumps(
            {
                "instance_type": self.instance_type,
                "container": image_uri,
            }
        )

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "start-job",
                "--namespace",
                self.namespace,
                "--recipe",
                "path/to/recipe",  # HyperPod CLI prefix and .yaml should be removed
                "--override-parameters",
                override_parameters,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(job_id, "test-job-123")

    @patch("subprocess.run")
    def test_execute_missing_parameters(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="")

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        with self.assertRaises(ValueError):
            manager.execute(
                job_name="",
                data_s3_path="s3://input-bucket/data",
                output_s3_path="s3://output-bucket/output",
                image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
                recipe=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
                input_s3_data_type=None,
            )

    @patch("subprocess.run")
    def test_execute_handles_error(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            Exception("Failed to start job"),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        with self.assertRaises(Exception) as context:
            manager.execute(
                job_name="test-job",
                data_s3_path="s3://input-bucket/data",
                output_s3_path="s3://output-bucket/output",
                image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
                recipe=f"{HYPERPOD_RECIPE_PATH}/path/to/recipe.yaml",
                input_s3_data_type=None,
            )

        self.assertEqual(str(context.exception), "Failed to start job")

    @patch("subprocess.run")
    def test_cleanup_success(self, mock_run):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )
        mock_run.reset_mock()  # Reset call count after initialization

        manager.cleanup("test-job")

        mock_run.assert_called_once_with(
            [
                "hyperpod",
                "cancel-job",
                "--job-name",
                "test-job",
                "--namespace",
                self.namespace,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_cleanup_handles_error(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            Exception("Cleanup failed"),
        ]

        manager = SMHPRuntimeManager(
            self.instance_type, self.instance_count, self.cluster_name, self.namespace
        )

        with self.assertRaises(Exception) as context:
            manager.cleanup("test-job")

        self.assertEqual(str(context.exception), "Cleanup failed")


if __name__ == "__main__":
    unittest.main()
