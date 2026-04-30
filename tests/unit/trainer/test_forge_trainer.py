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
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

from amzn_nova_forge.core.enums import Model, Platform, TrainingMethod
from amzn_nova_forge.core.result import (
    BedrockTrainingResult,
    SMHPTrainingResult,
    SMTJTrainingResult,
)
from amzn_nova_forge.core.types import ForgeConfig, ModelArtifacts
from amzn_nova_forge.manager.runtime_manager import (
    BedrockRuntimeManager,
    SMHPRuntimeManager,
    SMTJRuntimeManager,
)
from amzn_nova_forge.trainer.forge_trainer import ForgeTrainer

FIXED_OUTPUT_PATH = "s3://sagemaker-nova-123456789012-us-east-1/output"


def _make_smtj_infra():
    infra = create_autospec(SMTJRuntimeManager)
    infra.instance_type = "ml.p5.48xlarge"
    infra.instance_count = 2
    infra.kms_key_id = None
    infra.platform = Platform.SMTJ
    infra.rft_lambda_arn = None
    return infra


def _make_bedrock_infra():
    infra = create_autospec(BedrockRuntimeManager)
    infra.instance_type = None
    infra.instance_count = None
    infra.kms_key_id = None
    infra.platform = Platform.BEDROCK
    infra.rft_lambda_arn = None
    return infra


def _make_smhp_infra():
    infra = create_autospec(SMHPRuntimeManager)
    infra.instance_type = "ml.p5.48xlarge"
    infra.instance_count = 2
    infra.kms_key_id = None
    infra.platform = Platform.SMHP
    infra.cluster_name = "my-cluster"
    infra.namespace = "kubeflow"
    infra.rft_lambda_arn = None
    return infra


class TestForgeTrainerInit(unittest.TestCase):
    """Tests for ForgeTrainer.__init__."""

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_happy_path(self, mock_session, mock_set_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_smtj_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )

        self.assertEqual(trainer.model, Model.NOVA_MICRO)
        self.assertEqual(trainer.method, TrainingMethod.SFT_LORA)
        self.assertEqual(trainer.region, "us-east-1")
        self.assertEqual(trainer.output_s3_path, FIXED_OUTPUT_PATH)
        self.assertFalse(trainer._is_multimodal)
        self.assertIsNone(trainer.data_mixing)
        mock_set_output.assert_called_once()

    @patch("boto3.session.Session")
    def test_unsupported_region_raises(self, mock_session):
        type(mock_session.return_value).region_name = PropertyMock(return_value="ap-southeast-99")
        infra = _make_smtj_infra()

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
            )
        self.assertIn("ap-southeast-99", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_smtj_serverless_requires_sagemaker_arn(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = create_autospec(SMTJRuntimeManager)
        infra.instance_type = None
        infra.instance_count = None
        infra.kms_key_id = None
        infra.platform = Platform.SMTJServerless

        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_s3_path="s3://bucket/checkpoint/",
            )
        self.assertIn("model package ARN", str(ctx.exception))

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_bedrock_warns_on_model_s3_path(self, mock_session, _mock_output):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        infra = _make_bedrock_infra()

        with patch("amzn_nova_forge.trainer.forge_trainer.logger") as mock_logger:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                model_s3_path="s3://bucket/model",
            )
            mock_logger.warning.assert_called_once()
            self.assertIn(
                "model_path is not used for Bedrock",
                mock_logger.warning.call_args[0][0],
            )

    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_raises_for_unsupported_platform_or_method(
        self, mock_session, _mock_output
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        # SMTJ (non-SMHP) should raise
        infra = _make_smtj_infra()
        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.CPT,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                data_mixing_enabled=True,
            )
        self.assertIn("SageMaker HyperPod", str(ctx.exception))

        # SMHP with unsupported method (DPO_LORA) should raise
        infra_smhp = _make_smhp_infra()
        with self.assertRaises(ValueError) as ctx:
            ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.DPO_LORA,
                infra=infra_smhp,
                training_data_s3_path="s3://bucket/data",
                data_mixing_enabled=True,
            )
        self.assertIn("Data mixing is only supported", str(ctx.exception))

    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_data_mixing_smhp_cpt_works(self, mock_session, _mock_output, mock_load_templates):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = _make_smhp_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.CPT,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertIsNotNone(trainer.data_mixing)
        mock_load_templates.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.is_multimodal_data", return_value=True)
    @patch("amzn_nova_forge.trainer.forge_trainer.load_recipe_templates")
    @patch(
        "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
        return_value=FIXED_OUTPUT_PATH,
    )
    @patch("boto3.session.Session")
    def test_is_multimodal_auto_detection(
        self, mock_session, _mock_output, mock_load_templates, mock_is_mm
    ):
        type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
        mock_load_templates.return_value = ({}, {}, None, "image:latest")
        infra = _make_smhp_infra()

        trainer = ForgeTrainer(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.CPT,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
            data_mixing_enabled=True,
        )
        self.assertTrue(trainer._is_multimodal)
        mock_is_mm.assert_called_once_with("s3://bucket/data")


class TestForgeTrainerTrain(unittest.TestCase):
    """Tests for ForgeTrainer.train()."""

    def setUp(self):
        self._boto3_client_patcher = patch("boto3.client")
        self._mock_boto3_client = self._boto3_client_patcher.start()

    def tearDown(self):
        self._boto3_client_patcher.stop()

    def _make_trainer(self, infra=None, **kwargs):
        infra = infra or _make_smtj_infra()
        defaults = dict(
            model=Model.NOVA_MICRO,
            method=TrainingMethod.SFT_LORA,
            infra=infra,
            training_data_s3_path="s3://bucket/data",
        )
        defaults.update(kwargs)
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(**defaults)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_train_returns_smtj_result(self, MockRecipeBuilder, mock_get_artifacts):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )

        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, SMTJTrainingResult)
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.method, TrainingMethod.SFT_LORA)
        self.assertEqual(result.model_type, Model.NOVA_MICRO)
        infra.execute.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_dry_run_returns_none(self, MockRecipeBuilder):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )

        infra = _make_smtj_infra()
        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job", dry_run=True)

        self.assertIsNone(result)
        infra.execute.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_rft_lambda_arn_validated_when_provided(
        self, MockRecipeBuilder, mock_get_artifacts, mock_validate_lambda
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path=None, output_s3_path="s3://bucket/output"
        )

        trainer = self._make_trainer(infra=infra)
        trainer.train(
            job_name="my-job",
            rft_lambda_arn="arn:aws:lambda:us-east-1:123:function:my-func",
        )

        mock_validate_lambda.assert_called_once_with("my-func", Platform.SMTJ)

    @patch("amzn_nova_forge.trainer.forge_trainer.validate_rft_lambda_name")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_rft_lambda_arn_falls_back_to_infra(
        self, MockRecipeBuilder, mock_get_artifacts, mock_validate_lambda
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.rft_lambda_arn = "arn:aws:lambda:us-east-1:123:function:infra-func"
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path=None, output_s3_path="s3://bucket/output"
        )

        trainer = self._make_trainer(infra=infra)
        trainer.train(job_name="my-job")

        mock_validate_lambda.assert_called_once_with("infra-func", Platform.SMTJ)

    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_bedrock_training_returns_bedrock_result(self, MockRecipeBuilder):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_bedrock_infra()
        infra.execute.return_value = "bedrock-job-123"

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, BedrockTrainingResult)
        self.assertEqual(result.job_id, "bedrock-job-123")
        self.assertIsNone(result.model_artifacts.checkpoint_s3_path)

    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_smhp_training_returns_smhp_result(self, MockRecipeBuilder, mock_get_artifacts):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smhp_infra()
        infra.execute.return_value = "smhp-job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        trainer = self._make_trainer(infra=infra)
        result = trainer.train(job_name="my-job")

        self.assertIsInstance(result, SMHPTrainingResult)
        self.assertEqual(result.job_id, "smhp-job-123")
        self.assertEqual(result.cluster_name, "my-cluster")
        self.assertEqual(result.namespace, "kubeflow")


class TestForgeTrainerCaching(unittest.TestCase):
    """Tests for caching integration in ForgeTrainer.train()."""

    def setUp(self):
        self._boto3_client_patcher = patch("boto3.client")
        self._boto3_client_patcher.start()

    def tearDown(self):
        self._boto3_client_patcher.stop()

    def _make_trainer(self, enable_caching=False):
        infra = _make_smtj_infra()
        config = ForgeConfig(enable_job_caching=enable_caching) if enable_caching else None
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                config=config,
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result")
    def test_cached_result_short_circuits_train(self, mock_load):
        mock_cached = MagicMock(spec=SMTJTrainingResult)
        mock_load.return_value = mock_cached

        trainer = self._make_trainer(enable_caching=True)
        result = trainer.train(job_name="cached-job")

        self.assertIs(result, mock_cached)
        mock_load.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result", return_value=None)
    def test_persist_called_after_successful_train(
        self, mock_load, MockRecipeBuilder, mock_get_artifacts, mock_persist
    ):
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/checkpoint",
            output_s3_path="s3://bucket/output",
        )

        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            trainer = ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
                config=ForgeConfig(enable_job_caching=True),
            )

        trainer.train(job_name="my-job")
        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        self.assertEqual(call_kwargs[1]["job_name"], "my-job")
        self.assertEqual(call_kwargs[1]["job_type"], "train")

    @patch("amzn_nova_forge.trainer.forge_trainer.persist_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.load_existing_result")
    @patch("amzn_nova_forge.trainer.forge_trainer.get_model_artifacts")
    @patch("amzn_nova_forge.trainer.forge_trainer.RecipeBuilder")
    def test_caching_noop_when_disabled(
        self, MockRecipeBuilder, mock_get_artifacts, mock_load, mock_persist
    ):
        mock_load.return_value = None
        mock_builder = MockRecipeBuilder.return_value
        mock_builder.build_and_validate.return_value = (
            "/tmp/recipe.yaml",
            "s3://bucket/output",
            "s3://bucket/data",
            "image:latest",
        )
        infra = _make_smtj_infra()
        infra.execute.return_value = "job-123"
        mock_get_artifacts.return_value = ModelArtifacts(
            checkpoint_s3_path="s3://bucket/ckpt", output_s3_path="s3://bucket/out"
        )

        trainer = self._make_trainer(enable_caching=False)
        trainer.train(job_name="my-job")

        # Cache functions are still called but context has caching disabled
        mock_load.assert_called_once()
        mock_persist.assert_called_once()
        self.assertFalse(trainer._cache_context.enable_job_caching)


class TestForgeTrainerGetLogs(unittest.TestCase):
    """Tests for ForgeTrainer.get_logs()."""

    def _make_trainer(self, infra=None):
        infra = infra or _make_smtj_infra()
        with (
            patch(
                "amzn_nova_forge.trainer.forge_trainer.set_output_s3_path",
                return_value=FIXED_OUTPUT_PATH,
            ),
            patch("boto3.session.Session") as mock_session,
        ):
            type(mock_session.return_value).region_name = PropertyMock(return_value="us-east-1")
            return ForgeTrainer(
                model=Model.NOVA_MICRO,
                method=TrainingMethod.SFT_LORA,
                infra=infra,
                training_data_s3_path="s3://bucket/data",
            )

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_with_job_result(self, MockMonitor):
        trainer = self._make_trainer()
        mock_result = MagicMock()
        mock_result.job_id = "job-abc"
        mock_result.started_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_result=mock_result)

        MockMonitor.assert_called_once_with(
            job_id="job-abc",
            platform=Platform.SMTJ,
            started_time=int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
        )
        MockMonitor.return_value.show_logs.assert_called_once()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_with_explicit_ids(self, MockMonitor):
        trainer = self._make_trainer()
        started = datetime(2025, 6, 15, tzinfo=timezone.utc)

        trainer.get_logs(job_id="explicit-job", started_time=started)

        MockMonitor.assert_called_once_with(
            job_id="explicit-job",
            platform=Platform.SMTJ,
            started_time=int(started.timestamp() * 1000),
        )

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_missing_info_returns_early(self, MockMonitor):
        trainer = self._make_trainer()

        with patch("amzn_nova_forge.trainer.forge_trainer.logger") as mock_logger:
            trainer.get_logs()
            mock_logger.info.assert_called_once()

        MockMonitor.assert_not_called()

    @patch("amzn_nova_forge.trainer.forge_trainer.CloudWatchLogMonitor")
    def test_get_logs_smhp_includes_cluster_kwargs(self, MockMonitor):
        infra = _make_smhp_infra()
        trainer = self._make_trainer(infra=infra)
        started = datetime(2025, 1, 1, tzinfo=timezone.utc)

        trainer.get_logs(job_id="smhp-job", started_time=started)

        MockMonitor.assert_called_once_with(
            job_id="smhp-job",
            platform=Platform.SMHP,
            started_time=int(started.timestamp() * 1000),
            cluster_name="my-cluster",
            namespace="kubeflow",
        )


if __name__ == "__main__":
    unittest.main()
