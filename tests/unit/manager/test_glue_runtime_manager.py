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
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from botocore.exceptions import ClientError

from amzn_nova_forge.core.enums import Platform
from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.base import OperationResult
from amzn_nova_forge.manager.glue_runtime_manager import (
    GlueRuntimeManager,
    _ensure_artifact_bucket,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig, JobConfig


class TestDataPrepJobConfig(unittest.TestCase):
    def test_glue_job_config_is_job_config(self):
        config = DataPrepJobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://bucket/input",
            output_s3_path="s3://bucket/output",
            extra_args={"pipeline_id": "default_text_filter"},
        )
        self.assertIsInstance(config, JobConfig)

    def test_defaults(self):
        config = DataPrepJobConfig(job_name="j", image_uri="", recipe_path="")
        self.assertEqual(config.input_format, "parquet")
        self.assertEqual(config.output_format, "parquet")
        self.assertEqual(config.text_field, "text")
        self.assertEqual(config.extra_args, {})


class TestEnsureArtifactBucket(unittest.TestCase):
    def test_returns_explicit_bucket(self):
        self.assertEqual(_ensure_artifact_bucket("us-east-1", "my-bucket"), "my-bucket")

    @patch("amzn_nova_forge.manager.glue_runtime_manager.ensure_bucket_exists")
    @patch("amzn_nova_forge.manager.glue_runtime_manager.get_dataprep_bucket_name")
    def test_creates_default_bucket(self, mock_get_bucket, mock_ensure):
        mock_get_bucket.return_value = "sagemaker-forge-dataprep-123456789012-us-west-2"

        bucket = _ensure_artifact_bucket("us-west-2")
        self.assertEqual(bucket, "sagemaker-forge-dataprep-123456789012-us-west-2")
        mock_ensure.assert_called_once_with(
            "sagemaker-forge-dataprep-123456789012-us-west-2", region="us-west-2"
        )

    @patch("amzn_nova_forge.manager.glue_runtime_manager.ensure_bucket_exists")
    @patch("amzn_nova_forge.manager.glue_runtime_manager.get_dataprep_bucket_name")
    def test_uses_existing_default_bucket(self, mock_get_bucket, mock_ensure):
        mock_get_bucket.return_value = "sagemaker-forge-dataprep-123456789012-us-east-1"

        bucket = _ensure_artifact_bucket("us-east-1")
        self.assertEqual(bucket, "sagemaker-forge-dataprep-123456789012-us-east-1")
        mock_ensure.assert_called_once()


class TestGlueRuntimeManager(unittest.TestCase):
    def _create_manager(self):
        with patch.object(GlueRuntimeManager, "setup", return_value=None):
            manager = GlueRuntimeManager(
                glue_role_name="TestRole",
                s3_artifact_bucket="test-bucket",
                worker_type="Z.2X",
                num_workers=2,
                region="us-east-1",
            )
        manager.region = "us-east-1"
        manager.s3_client = MagicMock()
        manager.glue_client = MagicMock()
        manager.iam_client = MagicMock()
        manager.s3_artifact_bucket = "test-bucket"
        manager.script_s3_path = "s3://test-bucket/scripts/invoke_glue_pipeline.py"
        manager.whl_s3_path = "s3://test-bucket/modules/agi_data_curator-1.0.0.zip"
        return manager

    def test_platform(self):
        manager = self._create_manager()
        self.assertEqual(manager.platform, Platform.GLUE)

    def test_instance_type_and_count_are_none(self):
        manager = self._create_manager()
        self.assertIsNone(manager.instance_type)
        self.assertIsNone(manager.instance_count)

    def test_execute_requires_glue_job_config(self):
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
        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }
        manager.glue_client.start_job_run.return_value = {"JobRunId": "jr_123"}
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}
        # AlreadyExistsException for create_job
        manager.glue_client.exceptions.AlreadyExistsException = type(
            "AlreadyExistsException", (Exception,), {}
        )
        manager.glue_client.exceptions.IdempotentParameterMismatchException = type(
            "IdempotentParameterMismatchException", (Exception,), {}
        )

        config = DataPrepJobConfig(
            job_name="nova-forge-default_text_filter",
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

        job_run_id = manager.execute(config)

        self.assertEqual(job_run_id, "jr_123")
        manager.glue_client.create_job.assert_called_once()
        manager.glue_client.start_job_run.assert_called_once()

        # Verify run args
        run_call = manager.glue_client.start_job_run.call_args
        run_args = run_call.kwargs["Arguments"]
        self.assertEqual(run_args["--pipeline_id"], "default_text_filter")
        self.assertEqual(run_args["--input_path"], "s3://bucket/input")
        self.assertEqual(run_args["--output_path"], "s3://bucket/output")
        self.assertEqual(run_args["--text_field"], "doc_text")
        self.assertIn("--extra_args", run_args)

    def test_execute_updates_existing_job(self):
        manager = self._create_manager()
        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }
        # create_job raises AlreadyExistsException
        exc_cls = type("AlreadyExistsException", (Exception,), {})
        manager.glue_client.exceptions.AlreadyExistsException = exc_cls
        manager.glue_client.exceptions.IdempotentParameterMismatchException = type(
            "IdempotentParameterMismatchException", (Exception,), {}
        )
        manager.glue_client.create_job.side_effect = exc_cls()
        manager.glue_client.start_job_run.return_value = {"JobRunId": "jr_456"}
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}

        config = DataPrepJobConfig(
            job_name="nova-forge-default_text_filter",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )

        job_run_id = manager.execute(config)
        self.assertEqual(job_run_id, "jr_456")
        manager.glue_client.update_job.assert_called_once()

    def test_execute_raises_on_failure(self):
        manager = self._create_manager()
        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }
        manager.glue_client.exceptions.AlreadyExistsException = type(
            "AlreadyExistsException", (Exception,), {}
        )
        manager.glue_client.exceptions.IdempotentParameterMismatchException = type(
            "IdempotentParameterMismatchException", (Exception,), {}
        )
        manager.glue_client.start_job_run.return_value = {"JobRunId": "jr_789"}
        manager.glue_client.get_job_run.return_value = {
            "JobRun": {"JobRunState": "FAILED", "ErrorMessage": "OOM"}
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
        self.assertIn("FAILED", str(ctx.exception))
        self.assertIn("OOM", str(ctx.exception))

    def test_execute_defaults_job_name_from_pipeline_id(self):
        manager = self._create_manager()
        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }
        manager.glue_client.exceptions.AlreadyExistsException = type(
            "AlreadyExistsException", (Exception,), {}
        )
        manager.glue_client.exceptions.IdempotentParameterMismatchException = type(
            "IdempotentParameterMismatchException", (Exception,), {}
        )
        manager.glue_client.start_job_run.return_value = {"JobRunId": "jr_abc"}
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}

        config = DataPrepJobConfig(
            job_name="",  # empty -> should default
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "exact_dedup_filter"},
        )

        manager.execute(config)
        create_call = manager.glue_client.create_job.call_args
        self.assertEqual(create_call.kwargs["Name"], "nova-forge-exact_dedup_filter")

    def test_cleanup_success(self):
        manager = self._create_manager()
        manager._last_job_name = "nova-forge-default_text_filter"

        manager.cleanup("jr_123")

        manager.glue_client.batch_stop_job_run.assert_called_once_with(
            JobName="nova-forge-default_text_filter", JobRunIds=["jr_123"]
        )

    def test_cleanup_raises_without_job_name(self):
        manager = self._create_manager()
        # _last_job_name not set
        with self.assertRaises(ValueError) as ctx:
            manager.cleanup("jr_123")
        self.assertIn("no job_name", str(ctx.exception))

    def test_cleanup_propagates_error(self):
        manager = self._create_manager()
        manager._last_job_name = "nova-forge-test"
        manager.glue_client.batch_stop_job_run.side_effect = Exception("API error")

        with self.assertRaises(Exception) as ctx:
            manager.cleanup("jr_123")
        self.assertEqual(str(ctx.exception), "API error")

    def test_required_calling_role_permissions(self):
        perms = GlueRuntimeManager.required_calling_role_permissions(
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
        )
        actions = [p[0] if isinstance(p, tuple) else p for p in perms]
        self.assertIn("glue:CreateJob", actions)
        self.assertIn("glue:StartJobRun", actions)
        self.assertIn("glue:GetJobRun", actions)
        self.assertIn("glue:BatchStopJobRun", actions)
        self.assertIn("iam:GetRole", actions)
        # Base S3 permissions should also be present
        self.assertIn("s3:GetObject", actions)


class TestGlueRoleResolution(unittest.TestCase):
    """Test that GlueRuntimeManager resolves execution role from name or ARN."""

    def _create_manager(self, glue_role_name="TestGlueRole"):
        with patch.object(GlueRuntimeManager, "setup", return_value=None):
            manager = GlueRuntimeManager(
                glue_role_name=glue_role_name,
                s3_artifact_bucket="test-bucket",
                region="us-east-1",
            )
        manager.region = "us-east-1"
        manager.s3_client = MagicMock()
        manager.glue_client = MagicMock()
        manager.iam_client = MagicMock()
        manager.s3_artifact_bucket = "test-bucket"
        manager.script_s3_path = "s3://test-bucket/scripts/invoke_glue_pipeline.py"
        manager.whl_s3_path = "s3://test-bucket/modules/agi_data_curator-1.0.0.zip"
        return manager

    def test_execute_resolves_role_name_to_arn(self):
        manager = self._create_manager(glue_role_name="MyGlueRole")
        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/MyGlueRole"}
        }
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}

        config = DataPrepJobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )
        manager.execute(config)

        manager.iam_client.get_role.assert_called_once_with(RoleName="MyGlueRole")
        call_kwargs = manager.glue_client.create_job.call_args.kwargs
        self.assertEqual(call_kwargs["Role"], "arn:aws:iam::123456789012:role/MyGlueRole")

    def test_execute_uses_arn_directly(self):
        role_arn = "arn:aws:iam::123456789012:role/PreExistingRole"
        manager = self._create_manager(glue_role_name=role_arn)
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}

        config = DataPrepJobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
        )
        manager.execute(config)

        # Should NOT call get_role when an ARN is passed
        manager.iam_client.get_role.assert_not_called()
        call_kwargs = manager.glue_client.create_job.call_args.kwargs
        self.assertEqual(call_kwargs["Role"], role_arn)


class TestPathValidation(unittest.TestCase):
    """Tests for path validation in filter operations (validated at execute time)."""

    def _create_glue_manager(self):
        with patch.object(GlueRuntimeManager, "setup", return_value=None):
            manager = GlueRuntimeManager(
                glue_role_name="TestRole",
                s3_artifact_bucket="test-bucket",
                region="us-east-1",
            )
        return manager

    def test_default_text_filter_rejects_local_input_with_glue(self):
        from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
            DefaultTextFilterOperation,
        )

        manager = self._create_glue_manager()
        op = DefaultTextFilterOperation()
        state = DataState(
            path="/tmp/raw/data.parquet", format="parquet", location=DataLocation.LOCAL
        )
        with patch.object(op, "prepare_input", return_value=state):
            with self.assertRaises(ValueError) as ctx:
                op.execute(
                    loader=None,
                    state=state,
                    output_path="s3://bucket/output/",
                    runtime_manager=manager,
                )
        self.assertIn("input_path", str(ctx.exception))
        self.assertIn("Remote runtime managers require S3 paths", str(ctx.exception))

    def test_default_text_filter_rejects_local_output_with_glue(self):
        from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
            DefaultTextFilterOperation,
        )

        manager = self._create_glue_manager()
        op = DefaultTextFilterOperation()
        state = DataState(path="s3://bucket/input/", format="parquet", location=DataLocation.S3)
        with self.assertRaises(ValueError) as ctx:
            op.execute(
                loader=None,
                state=state,
                output_path="/tmp/output/",
                runtime_manager=manager,
            )
        self.assertIn("output_path", str(ctx.exception))

    def test_default_text_filter_rejects_both_local_paths_with_glue(self):
        from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
            DefaultTextFilterOperation,
        )

        manager = self._create_glue_manager()
        op = DefaultTextFilterOperation()
        state = DataState(path="/tmp/raw/", format="parquet", location=DataLocation.LOCAL)
        with patch.object(op, "prepare_input", return_value=state):
            with self.assertRaises(ValueError) as ctx:
                op.execute(
                    loader=None,
                    state=state,
                    output_path="/tmp/output/",
                    runtime_manager=manager,
                )
        self.assertIn("input_path", str(ctx.exception))
        self.assertIn("output_path", str(ctx.exception))

    def test_default_text_filter_accepts_s3_paths_with_glue(self):
        from amzn_nova_forge.dataset.operations.default_text_filter_operation import (
            DefaultTextFilterOperation,
        )

        manager = self._create_glue_manager()
        manager.execute = MagicMock(return_value="run-123")
        op = DefaultTextFilterOperation()
        state = DataState(path="s3://bucket/input/", format="parquet", location=DataLocation.S3)
        result = op.execute(
            loader=None,
            state=state,
            output_path="s3://bucket/output/",
            runtime_manager=manager,
        )
        self.assertEqual(result.output_state.path, "s3://bucket/output/")

    def test_exact_dedup_rejects_local_input_with_glue(self):
        from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import (
            ExactDedupFilterOperation,
        )

        manager = self._create_glue_manager()
        op = ExactDedupFilterOperation()
        state = DataState(
            path="/tmp/raw/data.parquet", format="parquet", location=DataLocation.LOCAL
        )
        with patch.object(op, "prepare_input", return_value=state):
            with self.assertRaises(ValueError) as ctx:
                op.execute(
                    loader=None,
                    state=state,
                    output_path="s3://bucket/output/",
                    runtime_manager=manager,
                )
        self.assertIn("input_path", str(ctx.exception))
        self.assertIn("Remote runtime managers require S3 paths", str(ctx.exception))

    def test_exact_dedup_rejects_local_output_with_glue(self):
        from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import (
            ExactDedupFilterOperation,
        )

        manager = self._create_glue_manager()
        op = ExactDedupFilterOperation()
        state = DataState(path="s3://bucket/input/", format="parquet", location=DataLocation.S3)
        with self.assertRaises(ValueError) as ctx:
            op.execute(
                loader=None,
                state=state,
                output_path="/tmp/output/",
                runtime_manager=manager,
            )
        self.assertIn("output_path", str(ctx.exception))

    def test_exact_dedup_accepts_s3_paths_with_glue(self):
        from amzn_nova_forge.dataset.operations.exact_dedup_filter_operation import (
            ExactDedupFilterOperation,
        )

        manager = self._create_glue_manager()
        manager.execute = MagicMock(return_value="run-789")
        op = ExactDedupFilterOperation()
        state = DataState(path="s3://bucket/input/", format="parquet", location=DataLocation.S3)
        result = op.execute(
            loader=None,
            state=state,
            output_path="s3://bucket/output/",
            runtime_manager=manager,
        )
        self.assertEqual(result.output_state.path, "s3://bucket/output/")


class TestLazyFilterExecution(unittest.TestCase):
    """Tests that filter() is lazy and execute() triggers the operations."""

    def _load_stub(self, loader, path):
        """Set _load_path without triggering real file I/O."""
        loader._load_path = path
        loader._session_id = "2026-04-20_14-30-22"
        loader.dataset = lambda: iter([])

    def test_execute_without_load_raises(self):
        """execute() should raise ValueError when load() was not called."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        loader = JSONLDatasetLoader()
        loader.filter(method=FilterMethod.DEFAULT_TEXT_FILTER)
        with self.assertRaises(ValueError) as ctx:
            loader.execute()
        self.assertIn("No data source provided", str(ctx.exception))
        self.assertIn("load()", str(ctx.exception))

    def test_filter_does_not_execute_immediately(self):
        """Calling filter() should only queue — no execution."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        loader = JSONLDatasetLoader()
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/output/",
        )
        # Should have 1 pending filter, no execution yet
        self.assertEqual(len(loader._pending_operations), 1)
        self.assertEqual(loader._pending_operations[0][0], "filter")
        self.assertEqual(loader._pending_operations[0][1], FilterMethod.DEFAULT_TEXT_FILTER)

    def test_multiple_filters_are_queued(self):
        """Multiple filter() calls should queue without executing."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        loader = JSONLDatasetLoader()
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/filtered/",
        ).filter(
            method=FilterMethod.EXACT_DEDUP,
            output_path="s3://bucket/deduped/",
        )
        self.assertEqual(len(loader._pending_operations), 2)
        self.assertEqual(loader._pending_operations[0][1], FilterMethod.DEFAULT_TEXT_FILTER)
        self.assertEqual(loader._pending_operations[1][1], FilterMethod.EXACT_DEDUP)

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_runs_pending_operations(self, mock_get_op):
        """execute() should run all queued filters then clear the queue."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/output/", format="parquet", location=DataLocation.S3
            ),
        )
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/input/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/output/",
        )
        # Not executed yet
        mock_op.execute.assert_not_called()

        # Now execute
        result = loader.execute()

        mock_op.execute.assert_called_once()
        self.assertEqual(len(loader._pending_operations), 0)
        self.assertIs(result, loader)

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_chains_multiple_filters(self, mock_get_op):
        """execute() should run filters in order."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/output/", format="parquet", location=DataLocation.S3
            ),
        )
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/input/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/filtered/",
        ).filter(
            method=FilterMethod.EXACT_DEDUP,
            output_path="s3://bucket/deduped/",
        ).execute()

        self.assertEqual(mock_op.execute.call_count, 2)
        self.assertEqual(len(loader._pending_operations), 0)

    def test_execute_with_no_pending_is_noop(self):
        """execute() with no pending filters should be a no-op."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader

        loader = JSONLDatasetLoader()
        result = loader.execute()
        self.assertIs(result, loader)

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_auto_chains_input_from_previous_output(self, mock_get_op):
        """execute() should set input_path from previous output for second filter."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.side_effect = [
            OperationResult(
                status="SUCCEEDED",
                output_state=DataState(
                    path="s3://bucket/filtered/",
                    format="parquet",
                    location=DataLocation.S3,
                ),
            ),
            OperationResult(
                status="SUCCEEDED",
                output_state=DataState(
                    path="s3://bucket/deduped/",
                    format="parquet",
                    location=DataLocation.S3,
                ),
            ),
        ]
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/raw/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/filtered/",
        ).filter(
            method=FilterMethod.EXACT_DEDUP,
            output_path="s3://bucket/deduped/",
        ).execute()

        self.assertEqual(mock_op.execute.call_count, 2)

        # First call: state from load()
        first_call_kwargs = mock_op.execute.call_args_list[0]
        self.assertEqual(first_call_kwargs.kwargs["state"].path, "s3://bucket/raw/")

        # Second call: state from first filter's output
        second_call_kwargs = mock_op.execute.call_args_list[1]
        self.assertEqual(second_call_kwargs.kwargs["state"].path, "s3://bucket/filtered/")

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_auto_generates_output_path(self, mock_get_op):
        """output_path should be auto-generated as <parent>/<stem>/<session>/<method>_output/."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/raw/2026-04-20_14-30-22/default_text_filter_output/",
                format="parquet",
                location=DataLocation.S3,
            ),
        )
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/raw/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            # output_path intentionally omitted
        ).execute()

        call_kwargs = mock_op.execute.call_args_list[0]
        self.assertEqual(
            call_kwargs.kwargs.get("output_path"),
            "s3://bucket/raw/2026-04-20_14-30-22/default_text_filter_output/",
        )

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_auto_generates_chained_paths(self, mock_get_op):
        """Full auto-chain: only input_path on first filter, everything else derived."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.side_effect = [
            OperationResult(
                status="SUCCEEDED",
                output_state=DataState(
                    path="s3://bucket/raw/2026-04-20_14-30-22/default_text_filter_output/",
                    format="parquet",
                    location=DataLocation.S3,
                ),
            ),
            OperationResult(
                status="SUCCEEDED",
                output_state=DataState(
                    path="s3://bucket/raw/2026-04-20_14-30-22/exact_dedup_filter_output/",
                    format="parquet",
                    location=DataLocation.S3,
                ),
            ),
        ]
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/raw/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
        ).filter(
            method=FilterMethod.EXACT_DEDUP,
        ).execute()

        self.assertEqual(mock_op.execute.call_count, 2)

        # First filter: state from load(), output auto-generated
        first_kwargs = mock_op.execute.call_args_list[0]
        self.assertEqual(first_kwargs.kwargs["state"].path, "s3://bucket/raw/")
        self.assertEqual(
            first_kwargs.kwargs.get("output_path"),
            "s3://bucket/raw/2026-04-20_14-30-22/default_text_filter_output/",
        )

        # Second filter: state from first output, output auto-generated within same session
        second_kwargs = mock_op.execute.call_args_list[1]
        self.assertEqual(
            second_kwargs.kwargs["state"].path,
            "s3://bucket/raw/2026-04-20_14-30-22/default_text_filter_output/",
        )
        self.assertEqual(
            second_kwargs.kwargs.get("output_path"),
            "s3://bucket/raw/2026-04-20_14-30-22/exact_dedup_filter_output/",
        )

    @patch("amzn_nova_forge.dataset.dataset_loader.get_filter_operation")
    def test_execute_explicit_output_not_overridden(self, mock_get_op):
        """Explicit output_path should not be overridden by auto-generation."""
        from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
        from amzn_nova_forge.dataset.operations.filter_operation import FilterMethod

        mock_op = MagicMock()
        mock_op.execute.return_value = OperationResult(
            status="SUCCEEDED",
            output_state=DataState(
                path="s3://bucket/my-custom-output/", format="parquet", location=DataLocation.S3
            ),
        )
        mock_get_op.return_value = mock_op

        loader = JSONLDatasetLoader()
        self._load_stub(loader, "s3://bucket/raw/")
        loader.filter(
            method=FilterMethod.DEFAULT_TEXT_FILTER,
            output_path="s3://bucket/my-custom-output/",  # explicit
        ).execute()

        call_kwargs = mock_op.execute.call_args_list[0]
        self.assertEqual(call_kwargs.kwargs.get("output_path"), "s3://bucket/my-custom-output/")


class TestGlueEntryScriptSummaryJson(unittest.TestCase):
    """Entry scripts write _summary.json when metrics['result'] is a dict."""

    def _run_glue_entry_script(
        self, metrics_result, status="success", s3_side_effect=None, mock_s3=None
    ):
        """Execute the Glue entry script main() with mocked ForgeWorkflows."""
        from amzn_nova_forge.manager.glue_runtime_manager import _GLUE_ENTRY_SCRIPT

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

        env = {
            "pipeline_id": "exact_dedup",
            "input_path": "s3://bucket/input/",
            "output_path": "s3://bucket/output/",
            "input_format": "parquet",
            "output_format": "parquet",
            "text_field": "text",
            "extra_args": "{}",
        }

        script_globals = {"__name__": "__main__"}
        with patch.dict("os.environ", env, clear=False):
            with patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3_module,
                    "agi_data_curator": MagicMock(),
                    "agi_data_curator.workflows": MagicMock(),
                    "agi_data_curator.workflows.forge_workflows": MagicMock(
                        ForgeWorkflows=mock_forge_cls
                    ),
                },
            ):
                exec(compile(_GLUE_ENTRY_SCRIPT, "<glue_entry>", "exec"), script_globals)

        return mock_s3, mock_boto3_module

    def test_writes_summary_when_result_is_dict(self):
        result_dict = {"input_count": 500, "duplicates_removed": 42}
        mock_s3, _ = self._run_glue_entry_script(result_dict)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        self.assertEqual(call_kwargs["Bucket"], "bucket")
        self.assertEqual(call_kwargs["Key"], "output/_summary.json")

        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        self.assertEqual(body, result_dict)

    def test_skips_summary_when_result_is_not_dict(self):
        mock_s3, _ = self._run_glue_entry_script("not-a-dict")
        mock_s3.put_object.assert_not_called()

    def test_skips_summary_on_error_status(self):
        result_dict = {"input_count": 100, "duplicates_removed": 5}
        mock_s3 = MagicMock()
        with self.assertRaises(RuntimeError):
            self._run_glue_entry_script(result_dict, status="error", mock_s3=mock_s3)
        mock_s3.put_object.assert_not_called()

    def test_put_object_failure_does_not_raise(self):
        result_dict = {"input_count": 500, "duplicates_removed": 42}
        mock_s3, _ = self._run_glue_entry_script(
            result_dict, s3_side_effect=Exception("access denied")
        )
        mock_s3.put_object.assert_called_once()


class TestPipInstallList(unittest.TestCase):
    """Tests for the --pip-install list construction in execute()."""

    def _create_manager(self):
        with patch.object(GlueRuntimeManager, "setup", return_value=None):
            manager = GlueRuntimeManager(
                glue_role_name="TestRole",
                s3_artifact_bucket="test-bucket",
                region="us-east-1",
            )
        manager.region = "us-east-1"
        manager.s3_client = MagicMock()
        manager.glue_client = MagicMock()
        manager.iam_client = MagicMock()
        manager.s3_artifact_bucket = "test-bucket"
        manager.script_s3_path = "s3://test-bucket/scripts/invoke_glue_pipeline.py"
        manager.whl_s3_path = "s3://test-bucket/modules/agi_data_curator-1.0.0.zip"

        manager.iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }
        manager.glue_client.exceptions.AlreadyExistsException = type(
            "AlreadyExistsException", (Exception,), {}
        )
        manager.glue_client.exceptions.IdempotentParameterMismatchException = type(
            "IdempotentParameterMismatchException", (Exception,), {}
        )
        manager.glue_client.start_job_run.return_value = {"JobRunId": "jr_pip"}
        manager.glue_client.get_job_run.return_value = {"JobRun": {"JobRunState": "SUCCEEDED"}}
        return manager

    def _pip_install_value(self, manager) -> str:
        create_call = manager.glue_client.create_job.call_args
        return create_call.kwargs["DefaultArguments"]["--pip-install"]

    def _run_with_extra_pip(self, extra_pip_packages):
        manager = self._create_manager()
        config = DataPrepJobConfig(
            job_name="test-job",
            image_uri="",
            recipe_path="",
            data_s3_path="s3://b/in",
            output_s3_path="s3://b/out",
            extra_args={"pipeline_id": "default_text_filter"},
            extra_pip_packages=extra_pip_packages,
        )
        manager.execute(config)
        return self._pip_install_value(manager)

    def test_baseline_only_when_no_extra_packages(self):
        """With no extra_pip_packages, --pip-install is just the baseline."""
        value = self._run_with_extra_pip(None)
        self.assertEqual(value, "s3fs,loguru")

    def test_baseline_only_when_empty_list(self):
        """Empty list is treated the same as None."""
        value = self._run_with_extra_pip([])
        self.assertEqual(value, "s3fs,loguru")

    def test_extra_packages_appended(self):
        """Extra packages are appended after baseline deps."""
        value = self._run_with_extra_pip(["fasttext-wheel"])
        self.assertEqual(value, "s3fs,loguru,fasttext-wheel")

    def test_multiple_extra_packages_preserve_order(self):
        """Multiple extras keep their declaration order after the baseline."""
        value = self._run_with_extra_pip(["fasttext-wheel", "datasketch", "xxhash"])
        self.assertEqual(value, "s3fs,loguru,fasttext-wheel,datasketch,xxhash")

    def test_duplicate_extras_are_deduplicated(self):
        """Repeated packages in extra_pip_packages appear only once."""
        value = self._run_with_extra_pip(["fasttext-wheel", "fasttext-wheel"])
        self.assertEqual(value, "s3fs,loguru,fasttext-wheel")

    def test_extra_package_overlapping_baseline_is_not_duplicated(self):
        """A package that overlaps with a baseline dep is kept only once
        (in its baseline position)."""
        value = self._run_with_extra_pip(["s3fs", "fasttext-wheel"])
        self.assertEqual(value, "s3fs,loguru,fasttext-wheel")


if __name__ == "__main__":
    unittest.main()
