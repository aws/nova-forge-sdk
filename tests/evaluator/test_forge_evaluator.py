import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

from amzn_nova_forge.core.enums import EvaluationTask, Model, Platform, TrainingMethod
from amzn_nova_forge.core.result import (
    SMHPEvaluationResult,
    SMTJEvaluationResult,
)
from amzn_nova_forge.core.result.training_result import TrainingResult
from amzn_nova_forge.core.types import ForgeConfig, ModelArtifacts
from amzn_nova_forge.evaluator.forge_evaluator import EvalTaskConfig, ForgeEvaluator
from amzn_nova_forge.manager.runtime_manager import (
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)


class TestForgeEvaluatorInit(unittest.TestCase):
    """Tests for ForgeEvaluator constructor."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_init_happy_path(self, mock_client, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            data_s3_path="s3://bucket/data",
        )
        self.assertEqual(evaluator.model, Model.NOVA_MICRO)
        self.assertEqual(evaluator.region, "us-east-1")
        self.assertEqual(evaluator.data_s3_path, "s3://bucket/data")
        self.assertEqual(evaluator.output_s3_path, "s3://bucket/output")

    @patch("boto3.session.Session")
    def test_init_unsupported_region_raises(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(
            return_value="unsupported-region"
        )
        with self.assertRaises(ValueError) as ctx:
            ForgeEvaluator(
                model=self.model,
                infra=self.mock_infra,
            )
        self.assertIn("unsupported-region", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
        return_value="s3://bucket/output",
    )
    @patch("boto3.session.Session")
    @patch("boto3.client")
    def test_platform_resolved_from_infra(self, mock_client, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.platform = Platform.SMHP
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2

        evaluator = ForgeEvaluator(model=self.model, infra=smhp_infra)
        self.assertEqual(evaluator._platform, Platform.SMHP)


class TestForgeEvaluatorEvaluate(unittest.TestCase):
    """Tests for ForgeEvaluator.evaluate()."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ
        self.mock_infra.rft_lambda_arn = None
        self.mock_infra.execute.return_value = "job-123"

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        self.evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            data_s3_path="s3://bucket/data",
        )

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    def test_bedrock_platform_raises_not_implemented(self):
        self.evaluator._platform = Platform.BEDROCK
        with self.assertRaises(NotImplementedError) as ctx:
            self.evaluator.evaluate(
                job_name="test-eval",
                eval_task=EvaluationTask.MMLU,
            )
        self.assertIn("Bedrock", str(ctx.exception))

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_smtj_evaluate_returns_smtj_result(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        self.assertIsInstance(result, SMTJEvaluationResult)
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.eval_task, EvaluationTask.MMLU)
        self.mock_infra.execute.assert_called_once()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_smhp_evaluate_returns_smhp_result(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMHP
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2
        smhp_infra.platform = Platform.SMHP
        smhp_infra.rft_lambda_arn = None
        smhp_infra.execute.return_value = "smhp-job-456"
        smhp_infra.cluster_name = "my-cluster"
        smhp_infra.namespace = "kubeflow"

        self.evaluator.infra = smhp_infra
        self.evaluator._platform = Platform.SMHP

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        self.assertIsInstance(result, SMHPEvaluationResult)
        self.assertEqual(result.job_id, "smhp-job-456")
        self.assertEqual(result.cluster_name, "my-cluster")
        self.assertEqual(result.namespace, "kubeflow")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_dry_run_returns_none(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        result = self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            dry_run=True,
        )

        self.assertIsNone(result)
        self.mock_infra.execute.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_model_path_passed_to_resolve(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_resolve.return_value = "s3://bucket/my-checkpoint"
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://bucket/my-checkpoint",
        )

        mock_resolve.assert_called_once_with(
            model_path="s3://bucket/my-checkpoint",
            job_result=None,
            customizer_job_id=None,
            customizer_output_s3_path="s3://bucket/output",
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_job_result_used_for_checkpoint(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_resolve.return_value = "s3://bucket/from-result"
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        mock_job_result = MagicMock(spec=TrainingResult)
        mock_job_result.model_artifacts = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            job_result=mock_job_result,
        )

        mock_resolve.assert_called_once_with(
            model_path=None,
            job_result=mock_job_result,
            customizer_job_id=None,
            customizer_output_s3_path="s3://bucket/output",
            customizer_model_path=None,
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_override_data_s3_path_used(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://custom/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(override_data_s3_path="s3://custom/data")
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.GEN_QA,
            task_config=tc,
        )

        # RecipeBuilder should receive the override data path
        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(kwargs["data_s3_path"], "s3://custom/data")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_byod_eval_uses_constructor_data_s3_path(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        # GEN_QA is a BYOD task, so constructor's data_s3_path should be used
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.GEN_QA,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(kwargs["data_s3_path"], "s3://bucket/data")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_non_byod_eval_ignores_data_s3_path(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        # MMLU is NOT a BYOD task, data_s3_path should be None
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertIsNone(kwargs["data_s3_path"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_rft_eval_processor_lambda_arn_to_rl_env(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(
            processor={"lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"}
        )
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
            task_config=tc,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(
            kwargs["rl_env_config"],
            {"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"},
        )
        # processor should be cleared
        self.assertIsNone(kwargs["processor_config"])

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_infra_rft_lambda_arn_auto_populates_rl_env(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        self.mock_infra.rft_lambda_arn = "arn:aws:lambda:us-east-1:123:function:InfraLambda"

        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
        )

        _, kwargs = mock_recipe_cls.call_args
        self.assertEqual(
            kwargs["rl_env_config"],
            {"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:InfraLambda"},
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_rft_lambda_name")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value=None,
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    def test_rl_env_reward_lambda_arn_validated(
        self,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
        mock_validate_lambda,
    ):
        mock_detect.return_value = None
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            None,
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        tc = EvalTaskConfig(
            rl_env={"reward_lambda_arn": "arn:aws:lambda:us-east-1:123:function:MyFunc"}
        )
        self.evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.RFT_EVAL,
            task_config=tc,
        )

        mock_validate_lambda.assert_called_once_with("MyFunc", Platform.SMTJ)


class TestForgeEvaluatorCaching(unittest.TestCase):
    """Tests for caching integration in ForgeEvaluator.evaluate()."""

    def setUp(self):
        self.model = Model.NOVA_MICRO
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ
        self.mock_infra.rft_lambda_arn = None
        self.mock_infra.execute.return_value = "eval-job-123"

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.load_existing_result")
    def test_cached_result_short_circuits_evaluate(self, mock_load):
        mock_cached = MagicMock(spec=SMTJEvaluationResult)
        mock_load.return_value = mock_cached

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        result = evaluator.evaluate(job_name="cached-eval", eval_task=EvaluationTask.MMLU)

        self.assertIs(result, mock_cached)
        self.mock_infra.execute.assert_not_called()
        mock_load.assert_called_once()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.persist_result")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.validate_platform_compatibility")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.detect_platform_from_path")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.resolve_model_checkpoint_path",
        return_value="s3://bucket/checkpoint",
    )
    @patch("amzn_nova_forge.evaluator.forge_evaluator.RecipeBuilder")
    @patch(
        "amzn_nova_forge.evaluator.forge_evaluator.load_existing_result",
        return_value=None,
    )
    def test_persist_called_after_successful_evaluate(
        self,
        mock_load,
        mock_recipe_cls,
        mock_resolve,
        mock_detect,
        mock_validate_compat,
        mock_persist,
    ):
        mock_detect.return_value = Platform.SMTJ
        mock_builder = MagicMock()
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        mock_recipe_cls.return_value = mock_builder

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        evaluator.evaluate(job_name="test-eval", eval_task=EvaluationTask.MMLU)

        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        self.assertEqual(call_kwargs[1]["job_name"], "test-eval")
        self.assertEqual(call_kwargs[1]["job_type"], "eval")

    @patch("amzn_nova_forge.evaluator.forge_evaluator.load_existing_result")
    def test_cache_load_receives_model_path(self, mock_load):
        mock_load.return_value = MagicMock(spec=SMTJEvaluationResult)

        evaluator = ForgeEvaluator(
            model=self.model,
            infra=self.mock_infra,
            config=ForgeConfig(enable_job_caching=True),
        )
        evaluator.evaluate(
            job_name="test-eval",
            eval_task=EvaluationTask.MMLU,
            model_path="s3://bucket/my-checkpoint",
        )

        call_kwargs = mock_load.call_args[1]
        self.assertEqual(call_kwargs["model_path"], "s3://bucket/my-checkpoint")


class TestForgeEvaluatorGetLogs(unittest.TestCase):
    """Tests for ForgeEvaluator.get_logs()."""

    def setUp(self):
        self.mock_infra = create_autospec(SMTJRuntimeManager)
        self.mock_infra.kms_key_id = None
        self.mock_infra.instance_type = "ml.p5.48xlarge"
        self.mock_infra.instance_count = 2
        self.mock_infra.platform = Platform.SMTJ

        self._patcher_set_output = patch(
            "amzn_nova_forge.evaluator.forge_evaluator.set_output_s3_path",
            return_value="s3://bucket/output",
        )
        self._patcher_session = patch("boto3.session.Session")
        self._patcher_client = patch("boto3.client")

        self._patcher_set_output.start()
        mock_session = self._patcher_session.start()
        self._patcher_client.start()

        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        self.evaluator = ForgeEvaluator(
            model=Model.NOVA_MICRO,
            infra=self.mock_infra,
        )

    def tearDown(self):
        self._patcher_client.stop()
        self._patcher_session.stop()
        self._patcher_set_output.stop()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_with_job_result(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        job_result = MagicMock()
        job_result.job_id = "eval-job-123"
        job_result.started_time = started

        self.evaluator.get_logs(job_result=job_result)

        mock_monitor_cls.assert_called_once_with(
            job_id="eval-job-123",
            platform=Platform.SMTJ,
            started_time=int(started.timestamp() * 1000),
        )
        mock_monitor.show_logs.assert_called_once_with(
            limit=None, start_from_head=False, end_time=None
        )

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    @patch("amzn_nova_forge.evaluator.forge_evaluator.logger")
    def test_get_logs_missing_params_logs_info(self, mock_logger, mock_monitor_cls):
        self.evaluator.get_logs()

        mock_logger.info.assert_called_once()
        self.assertIn("job_result", mock_logger.info.call_args[0][0])
        mock_monitor_cls.assert_not_called()

    @patch("amzn_nova_forge.evaluator.forge_evaluator.CloudWatchLogMonitor")
    def test_get_logs_smhp_includes_cluster_namespace(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        smhp_infra = create_autospec(SMHPRuntimeManager)
        smhp_infra.kms_key_id = None
        smhp_infra.instance_type = "ml.p5.48xlarge"
        smhp_infra.instance_count = 2
        smhp_infra.platform = Platform.SMHP
        smhp_infra.cluster_name = "my-cluster"
        smhp_infra.namespace = "kubeflow"

        self.evaluator.infra = smhp_infra
        self.evaluator._platform = Platform.SMHP

        started = datetime(2025, 1, 1, tzinfo=timezone.utc)
        job_result = MagicMock()
        job_result.job_id = "smhp-eval-789"
        job_result.started_time = started

        self.evaluator.get_logs(job_result=job_result)

        mock_monitor_cls.assert_called_once_with(
            job_id="smhp-eval-789",
            platform=Platform.SMHP,
            started_time=int(started.timestamp() * 1000),
            cluster_name="my-cluster",
            namespace="kubeflow",
        )


if __name__ == "__main__":
    unittest.main()
