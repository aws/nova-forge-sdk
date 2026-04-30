# Copyright Amazon.com, Inc. or its affiliates

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
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.manager.runtime_manager import (
    DataPrepJobConfig,
    JobConfig,
    SMTJDataPrepRuntimeManager,
)
from amzn_nova_forge.model.model_enums import Platform


class TestSMTJDataPrepRuntimeManager(unittest.TestCase):
    def _create_manager(self, **overrides):
        defaults = {
            "instance_type": "ml.g5.2xlarge",
            "instance_count": 2,
            "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/ray-dataprep:latest",
            "s3_artifact_bucket": "test-bucket",
            "region": "us-east-1",
        }
        defaults.update(overrides)

        with patch.object(SMTJDataPrepRuntimeManager, "setup", return_value=None):
            manager = SMTJDataPrepRuntimeManager(**defaults)

        # Set attributes that setup() would normally initialize
        manager.region = defaults.get("region", "us-east-1")
        manager.sagemaker_client = MagicMock()
        manager.s3_client = MagicMock()
        manager.s3_artifact_bucket = defaults.get("s3_artifact_bucket", "test-bucket")
        manager.execution_role = "arn:aws:iam::123456789012:role/TestRole"
        manager.script_s3_path = (
            "s3://test-bucket/nova-forge/smtj-dataprep-artifacts/scripts/invoke_smtj_pipeline.py"
        )
        manager.deps_s3_prefix = "s3://test-bucket/nova-forge/smtj-dataprep-artifacts/deps"
        return manager

    def test_platform(self):
        manager = self._create_manager()
        self.assertEqual(manager.platform, Platform.SMTJ)

    def test_instance_type_and_count(self):
        manager = self._create_manager()
        self.assertEqual(manager.instance_type, "ml.g5.2xlarge")
        self.assertEqual(manager.instance_count, 2)

    def test_default_instance_count(self):
        manager = self._create_manager(instance_count=1)
        self.assertEqual(manager.instance_count, 1)

    def test_execute_requires_dataprep_job_config(self):
        manager = self._create_manager()
        job_config = JobConfig(job_name="test", image_uri="img", recipe_path="/recipe")
        with self.assertRaises(TypeError) as ctx:
            manager.execute(job_config)
        self.assertIn("DataPrepJobConfig", str(ctx.exception))

    def test_execute_requires_pipeline_id(self):
        manager = self._create_manager()
        config = DataPrepJobConfig(
            job_name="j",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
        )
        with self.assertRaises(ValueError) as ctx:
            manager.execute(config)
        self.assertIn("pipeline_id", str(ctx.exception))

    def test_execute_requires_data_s3_path(self):
        manager = self._create_manager()
        config = DataPrepJobConfig(
            job_name="j",
            image_uri="",
            recipe_path="",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )
        with self.assertRaises(ValueError) as ctx:
            manager.execute(config)
        self.assertIn("data_s3_path", str(ctx.exception))

    def test_execute_requires_output_s3_path(self):
        manager = self._create_manager()
        config = DataPrepJobConfig(
            job_name="j",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            extra_args={"pipeline_id": "default_text_filter"},
        )
        with self.assertRaises(ValueError) as ctx:
            manager.execute(config)
        self.assertIn("output_s3_path", str(ctx.exception))

    def test_execute_success(self):
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-dataprep-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            text_field="doc_text",
            extra_args={
                "pipeline_id": "default_text_filter",
                "max_url_to_text_ratio": 0.3,
            },
        )

        job_name = manager.execute(config)

        self.assertEqual(job_name, "nova-forge-dataprep-test")
        manager.sagemaker_client.create_training_job.assert_called_once()

        # Verify create_training_job params
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(call_kwargs["TrainingJobName"], "nova-forge-dataprep-test")
        self.assertEqual(call_kwargs["RoleArn"], "arn:aws:iam::123456789012:role/TestRole")
        self.assertEqual(
            call_kwargs["AlgorithmSpecification"]["TrainingImage"],
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/ray-dataprep:latest",
        )
        self.assertEqual(call_kwargs["ResourceConfig"]["InstanceType"], "ml.g5.2xlarge")
        self.assertEqual(call_kwargs["ResourceConfig"]["InstanceCount"], 2)

        # Verify hyperparameters
        hp = call_kwargs["HyperParameters"]
        self.assertEqual(hp["pipeline_id"], "default_text_filter")
        self.assertEqual(hp["input_path"], "s3://bucket/input")
        self.assertEqual(hp["output_path"], "s3://bucket/output")
        self.assertEqual(hp["text_field"], "doc_text")
        self.assertIn("extra_args", hp)

    def test_execute_defaults_job_name_from_pipeline_id(self):
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "exact_dedup_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        job_name = call_kwargs["TrainingJobName"]
        prefix, _, suffix = job_name.rpartition("-")
        self.assertEqual(prefix, "nova-forge-dataprep-exact-dedup-filter")
        self.assertTrue(suffix.isdigit(), f"Expected numeric timestamp suffix, got {suffix!r}")
        self.assertLessEqual(len(job_name), 63)

    def test_execute_raises_on_failure(self):
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "OutOfMemory",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        with self.assertRaises(RuntimeError) as ctx:
            manager.execute(config)
        self.assertIn("Failed", str(ctx.exception))
        self.assertIn("OutOfMemory", str(ctx.exception))

    def test_execute_raises_on_timeout(self):
        from botocore.exceptions import WaiterError

        manager = self._create_manager()
        manager.max_wait_time = 60
        manager.poll_interval = 30

        # Make the waiter raise WaiterError to simulate timeout
        mock_waiter = MagicMock()
        mock_waiter.wait.side_effect = WaiterError(
            "training_job_completed_or_stopped", "Max attempts exceeded", {}
        )
        manager.sagemaker_client.get_waiter.return_value = mock_waiter

        # After waiter fails, describe_training_job shows job is still InProgress
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        with self.assertRaises(TimeoutError) as ctx:
            manager.execute(config)
        self.assertIn("did not reach a terminal state", str(ctx.exception))

    def test_execute_waiter_used_with_correct_config(self):
        manager = self._create_manager()
        manager.poll_interval = 15
        manager.max_wait_time = 300

        mock_waiter = MagicMock()
        manager.sagemaker_client.get_waiter.return_value = mock_waiter
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        job_name = manager.execute(config)
        self.assertEqual(job_name, "nova-forge-test")

        manager.sagemaker_client.get_waiter.assert_called_once_with(
            "training_job_completed_or_stopped"
        )
        mock_waiter.wait.assert_called_once_with(
            TrainingJobName="nova-forge-test",
            WaiterConfig={"Delay": 15, "MaxAttempts": 20},
        )

    def test_execute_vpc_config(self):
        manager = self._create_manager(
            subnets=["subnet-abc123"],
            security_group_ids=["sg-abc123"],
        )
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertIn("VpcConfig", call_kwargs)
        self.assertEqual(call_kwargs["VpcConfig"]["Subnets"], ["subnet-abc123"])
        self.assertEqual(call_kwargs["VpcConfig"]["SecurityGroupIds"], ["sg-abc123"])

    def test_execute_kms_key(self):
        manager = self._create_manager(kms_key_id="arn:aws:kms:us-east-1:123:key/abc")
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertEqual(
            call_kwargs["OutputDataConfig"]["KmsKeyId"],
            "arn:aws:kms:us-east-1:123:key/abc",
        )
        self.assertEqual(
            call_kwargs["ResourceConfig"]["VolumeKmsKeyId"],
            "arn:aws:kms:us-east-1:123:key/abc",
        )

    def test_execute_no_vpc_when_not_configured(self):
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        self.assertNotIn("VpcConfig", call_kwargs)

    def test_cleanup_stops_job(self):
        manager = self._create_manager()
        manager.cleanup("nova-forge-test-123")
        manager.sagemaker_client.stop_training_job.assert_called_once_with(
            TrainingJobName="nova-forge-test-123"
        )

    def test_cleanup_propagates_error(self):
        manager = self._create_manager()
        manager.sagemaker_client.stop_training_job.side_effect = Exception("API error")

        with self.assertRaises(Exception) as ctx:
            manager.cleanup("nova-forge-test-123")
        self.assertEqual(str(ctx.exception), "API error")

    def test_required_calling_role_permissions(self):
        perms = SMTJDataPrepRuntimeManager.required_calling_role_permissions(
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
        )
        actions = [p[0] if isinstance(p, tuple) else p for p in perms]
        self.assertIn("sagemaker:CreateTrainingJob", actions)
        self.assertIn("sagemaker:DescribeTrainingJob", actions)
        self.assertIn("sagemaker:StopTrainingJob", actions)
        self.assertIn("iam:GetRole", actions)
        self.assertIn("iam:PassRole", actions)
        # Base S3 permissions should also be present
        self.assertIn("s3:GetObject", actions)

    def test_execute_input_data_config_has_code_channel(self):
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        input_data = call_kwargs["InputDataConfig"]
        self.assertEqual(len(input_data), 2)
        channel_names = [ch["ChannelName"] for ch in input_data]
        self.assertIn("code", channel_names)
        self.assertIn("deps", channel_names)
        code_channel = input_data[0]
        self.assertIn(
            "invoke_smtj_pipeline.py",
            code_channel["DataSource"]["S3DataSource"]["S3Uri"],
        )
        deps_channel = input_data[1]
        self.assertTrue(deps_channel["DataSource"]["S3DataSource"]["S3Uri"].endswith("/deps"))

    def test_execute_pip_install_hyperparameter_injects_baseline(self):
        """Baseline DLC packages are appended when caller passes no extras."""
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        hp = call_kwargs["HyperParameters"]
        self.assertIn("pip_install", hp)
        import json as _json

        pip_list = _json.loads(hp["pip_install"])
        self.assertTrue(any(p.startswith("ray") for p in pip_list))
        self.assertTrue(any(p.startswith("pandas") for p in pip_list))

    def test_execute_pip_install_merges_caller_extras_with_baseline(self):
        """Caller-supplied extras are merged with the DLC baseline."""
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "language_detection"},
            extra_pip_packages=["fasttext-wheel"],
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        hp = call_kwargs["HyperParameters"]
        self.assertIn("pip_install", hp)
        import json as _json

        pip_list = _json.loads(hp["pip_install"])
        self.assertIn("fasttext-wheel", pip_list)
        self.assertTrue(any(p.startswith("ray") for p in pip_list))
        self.assertTrue(any(p.startswith("pandas") for p in pip_list))

    def test_execute_pip_install_caller_pin_wins_over_baseline(self):
        """Caller's ray pin suppresses the default baseline ray entry."""
        manager = self._create_manager()
        manager.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
        }

        config = DataPrepJobConfig(
            job_name="nova-forge-test",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
            extra_pip_packages=["ray[default]==2.40.0"],
        )

        manager.execute(config)
        call_kwargs = manager.sagemaker_client.create_training_job.call_args.kwargs
        hp = call_kwargs["HyperParameters"]
        import json as _json

        pip_list = _json.loads(hp["pip_install"])
        ray_entries = [p for p in pip_list if p.startswith("ray")]
        self.assertEqual(ray_entries, ["ray[default]==2.40.0"])


class TestSMTJDataPrepRoleResolution(unittest.TestCase):
    """Test that SMTJDataPrepRuntimeManager resolves execution role from name or ARN."""

    @patch("amzn_nova_forge.manager.runtime_manager.boto3")
    def test_setup_resolves_role_name_to_arn(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.session.Session.return_value = mock_session

        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/MyCustomRole"}
        }
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_factory(service, **kwargs):
            if service == "sagemaker":
                return MagicMock()
            if service == "s3":
                return mock_s3
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto3.client.side_effect = client_factory

        with patch.object(SMTJDataPrepRuntimeManager, "_upload_entry_script"):
            with patch.object(SMTJDataPrepRuntimeManager, "_upload_bundled_whls"):
                manager = SMTJDataPrepRuntimeManager(
                    instance_type="ml.g5.2xlarge",
                    image_uri="test:latest",
                    execution_role_name="MyCustomRole",
                    s3_artifact_bucket="test-bucket",
                )

        self.assertEqual(manager.execution_role, "arn:aws:iam::123456789012:role/MyCustomRole")
        mock_iam.get_role.assert_called_once_with(RoleName="MyCustomRole")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3")
    def test_setup_uses_arn_directly(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.session.Session.return_value = mock_session

        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_factory(service, **kwargs):
            if service == "sagemaker":
                return MagicMock()
            if service == "s3":
                return mock_s3
            return MagicMock()

        mock_boto3.client.side_effect = client_factory

        role_arn = "arn:aws:iam::123456789012:role/PreExistingRole"
        with patch.object(SMTJDataPrepRuntimeManager, "_upload_entry_script"):
            with patch.object(SMTJDataPrepRuntimeManager, "_upload_bundled_whls"):
                manager = SMTJDataPrepRuntimeManager(
                    instance_type="ml.g5.2xlarge",
                    image_uri="test:latest",
                    execution_role_name=role_arn,
                    s3_artifact_bucket="test-bucket",
                )

        self.assertEqual(manager.execution_role, role_arn)
        # Should NOT call get_role when an ARN is passed
        for call in mock_boto3.client.call_args_list:
            if call[0] == ("iam",):
                self.fail("IAM client should not be created when ARN is passed directly")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3")
    def test_setup_creates_role_when_not_found(self, mock_boto3):
        mock_session = MagicMock()
        mock_session.region_name = "us-east-1"
        mock_boto3.session.Session.return_value = mock_session

        # Build a real IAM client to get the proper exception class, then mock
        import botocore.session

        real_iam = botocore.session.get_session().create_client("iam", region_name="us-east-1")
        NoSuchEntity = real_iam.exceptions.NoSuchEntityException
        real_iam.close()

        mock_iam = MagicMock()
        mock_iam.exceptions.NoSuchEntityException = NoSuchEntity
        mock_iam.get_role.side_effect = NoSuchEntity(
            {"Error": {"Code": "NoSuchEntity", "Message": "not found"}},
            "GetRole",
        )
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_factory(service, **kwargs):
            if service == "sagemaker":
                return MagicMock()
            if service == "s3":
                return mock_s3
            if service == "iam":
                return mock_iam
            return MagicMock()

        mock_boto3.client.side_effect = client_factory

        with patch.object(SMTJDataPrepRuntimeManager, "_upload_entry_script"):
            with patch.object(SMTJDataPrepRuntimeManager, "_upload_bundled_whls"):
                with patch(
                    "amzn_nova_forge.iam.iam_role_creator.create_smtj_dataprep_execution_role",
                    return_value={
                        "Role": {"Arn": "arn:aws:iam::123456789012:role/SmtjDataPrepExecutionRole"}
                    },
                ) as mock_create:
                    manager = SMTJDataPrepRuntimeManager(
                        instance_type="ml.g5.2xlarge",
                        image_uri="test:latest",
                        s3_artifact_bucket="test-bucket",
                    )

        self.assertEqual(
            manager.execution_role,
            "arn:aws:iam::123456789012:role/SmtjDataPrepExecutionRole",
        )
        mock_create.assert_called_once()

    def _build_manager_with_resolver(self, mock_boto3, instance_type, resolved_uri):
        mock_session = MagicMock()
        mock_session.region_name = "us-west-2"
        mock_boto3.session.Session.return_value = mock_session

        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        def client_factory(service, **kwargs):
            if service == "sagemaker":
                return MagicMock()
            if service == "s3":
                return mock_s3
            return MagicMock()

        mock_boto3.client.side_effect = client_factory

        role_arn = "arn:aws:iam::123456789012:role/PreExistingRole"
        with patch("sagemaker.core.image_uris.retrieve", return_value=resolved_uri) as mock_ret:
            with patch.object(SMTJDataPrepRuntimeManager, "_upload_entry_script"):
                with patch.object(SMTJDataPrepRuntimeManager, "_upload_bundled_whls"):
                    manager = SMTJDataPrepRuntimeManager(
                        instance_type=instance_type,
                        execution_role_name=role_arn,
                        s3_artifact_bucket="test-bucket",
                    )
        return manager, mock_ret

    @patch("amzn_nova_forge.manager.runtime_manager.boto3")
    def test_setup_defaults_image_uri_uses_caller_cpu_instance(self, mock_boto3):
        resolved_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
            "pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker"
        )
        manager, mock_retrieve = self._build_manager_with_resolver(
            mock_boto3, instance_type="ml.m5.2xlarge", resolved_uri=resolved_uri
        )

        self.assertEqual(manager.image_uri, resolved_uri)
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args.kwargs
        self.assertEqual(call_kwargs["framework"], "pytorch")
        self.assertEqual(call_kwargs["region"], "us-west-2")
        self.assertEqual(call_kwargs["version"], "2.8")
        self.assertEqual(call_kwargs["py_version"], "py312")
        self.assertEqual(call_kwargs["image_scope"], "training")
        # Caller-supplied CPU instance type is forwarded verbatim.
        self.assertEqual(call_kwargs["instance_type"], "ml.m5.2xlarge")

    @patch("amzn_nova_forge.manager.runtime_manager.boto3")
    def test_setup_defaults_image_uri_passes_gpu_instance_through(self, mock_boto3):
        # The CPU vs GPU DLC is picked by sagemaker.image_uris.retrieve based on
        # the instance_type it receives — we forward the caller's choice
        # verbatim and let the SDK map it to the right image.
        resolved_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
            "pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
        )
        _, mock_retrieve = self._build_manager_with_resolver(
            mock_boto3, instance_type="ml.g5.2xlarge", resolved_uri=resolved_uri
        )

        call_kwargs = mock_retrieve.call_args.kwargs
        self.assertEqual(call_kwargs["instance_type"], "ml.g5.2xlarge")


class TestSmtjEntryScriptSummaryJson(unittest.TestCase):
    """SMTJ entry script writes _summary.json when metrics['result'] is a dict."""

    def _run_smtj_entry_script(
        self, metrics_result, status="success", s3_side_effect=None, mock_s3=None
    ):
        """Execute the SMTJ entry script main() with mocked ForgeWorkflows."""
        script_src = SMTJDataPrepRuntimeManager._SMTJ_ENTRY_SCRIPT

        if mock_s3 is None:
            mock_s3 = MagicMock()
        if s3_side_effect is not None:
            mock_s3.put_object.side_effect = s3_side_effect
        mock_boto3_module = MagicMock()
        mock_boto3_module.client.return_value = mock_s3

        mock_forge_cls = MagicMock()
        mock_forge_instance = mock_forge_cls.return_value
        mock_forge_instance.execute.return_value = {
            "identifier": "exact_dedup",
            "status": status,
            "elapsed_seconds": 10.0,
            "elapsed_minutes": 0.17,
            "result": metrics_result,
        }

        hp = {
            "pipeline_id": "exact_dedup",
            "input_path": "s3://bucket/input/",
            "output_path": "s3://bucket/output/",
            "input_format": "parquet",
            "output_format": "parquet",
            "text_field": "text",
            "extra_args": "{}",
        }

        hp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, hp_dir)
        hp_path = os.path.join(hp_dir, "hyperparameters.json")
        with open(hp_path, "w") as f:
            json.dump(hp, f)

        env = {
            "SM_HOSTS": "",
            "SM_CURRENT_HOST": "",
        }

        script_globals = {"__name__": "__main__"}
        with patch.dict("os.environ", env, clear=False):
            with patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3_module,
                    "ray": MagicMock(),
                    "agi_data_curator": MagicMock(),
                    "agi_data_curator.workflows": MagicMock(),
                    "agi_data_curator.workflows.forge_workflows": MagicMock(
                        ForgeWorkflows=mock_forge_cls
                    ),
                },
            ):
                patched_src = script_src.replace(
                    "/opt/ml/input/config/hyperparameters.json", hp_path
                ).replace("/opt/ml/input/data/deps", "/nonexistent/deps")
                exec(compile(patched_src, "<smtj_entry>", "exec"), script_globals)

        return mock_s3, mock_boto3_module

    def test_writes_summary_when_result_is_dict(self):
        result_dict = {"input_count": 1000, "num_duplicates": 150}
        mock_s3, _ = self._run_smtj_entry_script(result_dict)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        self.assertEqual(call_kwargs["Bucket"], "bucket")
        self.assertEqual(call_kwargs["Key"], "output/_summary.json")

        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        self.assertEqual(body, result_dict)

    def test_skips_summary_when_result_is_not_dict(self):
        mock_s3, _ = self._run_smtj_entry_script("not-a-dict")
        mock_s3.put_object.assert_not_called()

    def test_skips_summary_on_error_status(self):
        result_dict = {"input_count": 100, "num_duplicates": 5}
        mock_s3 = MagicMock()
        with self.assertRaises(RuntimeError):
            self._run_smtj_entry_script(result_dict, status="error", mock_s3=mock_s3)
        mock_s3.put_object.assert_not_called()

    def test_put_object_failure_does_not_raise(self):
        result_dict = {"input_count": 1000, "num_duplicates": 150}
        mock_s3, _ = self._run_smtj_entry_script(
            result_dict, s3_side_effect=Exception("access denied")
        )
        mock_s3.put_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()
