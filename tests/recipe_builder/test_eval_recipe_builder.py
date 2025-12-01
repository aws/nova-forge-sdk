import os
import tempfile
import unittest

import yaml

from amzn_nova_customization_sdk.model.model_enums import Model, Platform
from amzn_nova_customization_sdk.recipe_builder.eval_recipe_builder import (
    EvalRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.eval_config import (
    Aggregation,
    EvalRecipeConfig,
    EvaluationMetric,
    EvaluationStrategy,
    EvaluationTask,
    ProcessorLambdaType,
)


class TestEvalRecipeBuilder(unittest.TestCase):
    def setUp(self):
        self.job_name = "test-job"
        self.platform = Platform.SMTJ
        self.data_s3_path = "s3://test-bucket/data"
        self.output_s3_path = "s3://test-bucket/output"

    def test_eval_recipe_builder_initialization(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        self.assertEqual(builder.platform, Platform.SMTJ)
        self.assertEqual(builder.model, Model.NOVA_LITE)
        self.assertEqual(builder.eval_task, EvaluationTask.MMLU)
        self.assertEqual(builder.instance_type, "ml.g5.48xlarge")
        self.assertEqual(builder.instance_count, 1)
        self.assertEqual(builder.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(builder.model_path, "nova-lite/prod")

    def test_create_eval_run_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        run_config = builder._create_base_run_config()

        self.assertEqual(run_config.name, self.job_name)
        self.assertEqual(run_config.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(run_config.data_s3_path, self.data_s3_path)
        self.assertEqual(run_config.output_s3_path, self.output_s3_path)
        self.assertEqual(run_config.replicas, 1)

    def test_create_public_benchmark_eval_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, "")
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertIsNone(eval_config.evaluation.subtask)
        self.assertEqual(eval_config.inference.max_new_tokens, 8196)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 0.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 8196,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )

    def test_create_public_benchmark_eval_with_full_overrides_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={
                "max_new_tokens": 2048,
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 10.0,
            },
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, "")
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertIsNone(eval_config.evaluation.subtask)
        self.assertEqual(eval_config.inference.max_new_tokens, 2048)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 10.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 10.0,
                },
            },
        )

    def test_create_public_benchmark_eval_with_partial_overrides_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={"max_new_tokens": 2048},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, "")
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertIsNone(eval_config.evaluation.subtask)
        self.assertEqual(eval_config.inference.max_new_tokens, 2048)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 0.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )

    def test_create_public_benchmark_eval_with_unknown_overrides_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={"max_new_tokens": 2048, "unknown_fields": "abc"},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, "")
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertIsNone(eval_config.evaluation.subtask)
        self.assertEqual(eval_config.inference.max_new_tokens, 2048)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 0.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 2048,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )

    def test_create_public_benchmark_with_subtask_eval_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="anatomy",
            overrides={},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, "")
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertEqual(eval_config.evaluation.subtask, "anatomy")
        self.assertEqual(eval_config.inference.max_new_tokens, 8196)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 0.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                    "subtask": "anatomy",
                },
                "inference": {
                    "max_new_tokens": 8196,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )

    def test_build_public_benchmark_eval_yaml(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.24xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path)

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["replicas"], 1)
            self.assertEqual(config["evaluation"]["task"], "mmlu")
            self.assertEqual(config["evaluation"]["strategy"], "zs_cot")
            self.assertEqual(config["evaluation"]["metric"], "accuracy")
            self.assertNotIn("subtask", config["evaluation"])
            self.assertEqual(config["inference"]["max_new_tokens"], 8196)
            self.assertEqual(config["inference"]["top_k"], -1)
            self.assertEqual(config["inference"]["top_p"], 1.0)
            self.assertEqual(config["inference"]["temperature"], 0.0)

    def test_build_public_benchmark_with_overrides_eval_yaml(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.24xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={"max_new_tokens": 2048},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path)

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["replicas"], 1)
            self.assertEqual(config["evaluation"]["task"], "mmlu")
            self.assertEqual(config["evaluation"]["strategy"], "zs_cot")
            self.assertEqual(config["evaluation"]["metric"], "accuracy")
            self.assertNotIn("subtask", config["evaluation"])
            self.assertEqual(config["inference"]["max_new_tokens"], 2048)
            self.assertEqual(config["inference"]["top_k"], -1)
            self.assertEqual(config["inference"]["top_p"], 1.0)
            self.assertEqual(config["inference"]["temperature"], 0.0)

    def test_build_public_benchmark_with_subtask_eval_yaml(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.24xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="anatomy",
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path)

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["replicas"], 1)
            self.assertEqual(config["evaluation"]["task"], "mmlu")
            self.assertEqual(config["evaluation"]["strategy"], "zs_cot")
            self.assertEqual(config["evaluation"]["metric"], "accuracy")
            self.assertEqual(config["evaluation"]["subtask"], "anatomy")
            self.assertEqual(config["inference"]["max_new_tokens"], 8196)
            self.assertEqual(config["inference"]["top_k"], -1)
            self.assertEqual(config["inference"]["top_p"], 1.0)
            self.assertEqual(config["inference"]["temperature"], 0.0)

    def test_create_byod_eval_config(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertEqual(eval_config.run.name, self.job_name)
        self.assertEqual(eval_config.run.model_type, "amazon.nova-lite-v1:0:300k")
        self.assertEqual(eval_config.run.data_s3_path, self.data_s3_path)
        self.assertEqual(eval_config.run.output_s3_path, self.output_s3_path)
        self.assertEqual(eval_config.run.replicas, 1)
        self.assertEqual(eval_config.evaluation.task, EvaluationTask.MMLU)
        self.assertEqual(
            eval_config.evaluation.strategy, EvaluationStrategy.ZERO_SHOT_COT
        )
        self.assertEqual(eval_config.evaluation.metric, EvaluationMetric.ACCURACY)
        self.assertIsNone(eval_config.evaluation.subtask)
        self.assertEqual(eval_config.inference.max_new_tokens, 8196)
        self.assertEqual(eval_config.inference.top_k, -1)
        self.assertEqual(eval_config.inference.top_p, 1.0)
        self.assertEqual(eval_config.inference.temperature, 0.0)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": self.data_s3_path,
                    "output_s3_path": self.output_s3_path,
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 8196,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )

    def test_build_byod_eval_yaml(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.24xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.GEN_QA,
            overrides={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_recipe.yaml")
            builder.build(file_path)

            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            self.assertIn("run", config)
            self.assertEqual(config["run"]["replicas"], 1)
            self.assertEqual(config["run"]["data_s3_path"], self.data_s3_path)
            self.assertEqual(config["evaluation"]["task"], "gen_qa")
            self.assertEqual(config["evaluation"]["strategy"], "gen_qa")
            self.assertEqual(config["evaluation"]["metric"], "all")
            self.assertNotIn("subtask", config["evaluation"])
            self.assertEqual(config["inference"]["max_new_tokens"], 8196)
            self.assertEqual(config["inference"]["top_k"], -1)
            self.assertEqual(config["inference"]["top_p"], 1.0)
            self.assertEqual(config["inference"]["temperature"], 0.0)

    def test_eval_build_with_invalid_instance_type(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.invalid.instance",
            instance_count=1,
            data_s3_path="",
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="anatomy",
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build()

        self.assertIn("not supported", str(context.exception))

    def test_eval_build_with_invalid_instance_count(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.12xlarge",
            instance_count=1000,
            data_s3_path="",
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="anatomy",
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build()

        self.assertIn("not supported", str(context.exception))

    def test_build_byod_eval_with_invalid_task(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="anatomy",
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build()
        self.assertIn(
            "BYOD evaluation must use following eval task", str(context.exception)
        )

    def test_build_public_eval_with_non_supported_subtask(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU_PRO,
            subtask="anatomy",
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build()

        self.assertIn("does not support subtasks", str(context.exception))

    def test_build_public_eval_with_invalid_subtask(self):
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.12xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            subtask="invalid-subtask",
            overrides={},
        )

        with self.assertRaises(ValueError) as context:
            builder.build()

        self.assertIn("Invalid subtask ", str(context.exception))

    def test_eval_recipe_builder_with_processor_config_v1_full_override(self):
        """Test EvalRecipeBuilder with processor_config for v1 model"""
        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "lambda_type": "custom_metrics",
            "preprocessing": {"enabled": "False"},
            "postprocessing": {"enabled": "True"},
            "aggregation": "average",
        }

        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
            processor_config=processor_config,
        )

        eval_config = builder._build_recipe_config()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertIsNotNone(eval_config.processor)
        self.assertEqual(
            eval_config.processor.lambda_arn,
            "arn:aws:lambda:us-east-1:123456789012:function:test",
        )
        self.assertEqual(
            eval_config.processor.lambda_type,
            ProcessorLambdaType.CUSTOM_METRICS,
        )
        self.assertFalse(eval_config.processor.preprocessing.enabled)
        self.assertTrue(eval_config.processor.postprocessing.enabled)
        self.assertEqual(eval_config.processor.aggregation, Aggregation.AVERAGE)

    def test_eval_recipe_builder_with_processor_config_v1_with_default_value(self):
        """Test EvalRecipeBuilder with processor_config for v1 model"""
        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
        }

        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
            processor_config=processor_config,
        )

        eval_config = builder._build_recipe_config()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertIsNotNone(eval_config.processor)
        self.assertEqual(
            eval_config.processor.lambda_arn,
            "arn:aws:lambda:us-east-1:123456789012:function:test",
        )
        self.assertEqual(
            eval_config.processor.lambda_type,
            ProcessorLambdaType.CUSTOM_METRICS,
        )
        self.assertTrue(eval_config.processor.preprocessing.enabled)
        self.assertTrue(eval_config.processor.postprocessing.enabled)
        self.assertEqual(eval_config.processor.aggregation, Aggregation.AVERAGE)

    def test_eval_recipe_builder_with_processor_config_v1_with_partial_override(self):
        """Test EvalRecipeBuilder with processor_config for v1 model"""
        processor_config = {
            "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "postprocessing": {"enabled": "False"},
        }

        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
            processor_config=processor_config,
        )

        eval_config = builder._build_recipe_config()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertIsNotNone(eval_config.processor)
        self.assertEqual(
            eval_config.processor.lambda_arn,
            "arn:aws:lambda:us-east-1:123456789012:function:test",
        )
        self.assertEqual(
            eval_config.processor.lambda_type,
            ProcessorLambdaType.CUSTOM_METRICS,
        )
        self.assertTrue(eval_config.processor.preprocessing.enabled)
        self.assertFalse(eval_config.processor.postprocessing.enabled)
        self.assertEqual(eval_config.processor.aggregation, Aggregation.AVERAGE)

    def test_eval_recipe_builder_with_rl_env_config_v1(self):
        """Test EvalRecipeBuilder with rl_env_config for v1 model"""
        rl_env_config = {
            "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:reward"
        }

        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=self.data_s3_path,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
            rl_env_config=rl_env_config,
        )

        eval_config = builder._build_recipe_config()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertIsNotNone(eval_config.rl_env)
        self.assertEqual(
            eval_config.rl_env.reward_lambda_arn,
            "arn:aws:lambda:us-east-1:123456789012:function:reward",
        )

    def test_eval_recipe_builder_empty_configs(self):
        """Test EvalRecipeBuilder with empty processor_config and rl_env_config"""
        builder = EvalRecipeBuilder(
            job_name=self.job_name,
            platform=self.platform,
            model=Model.NOVA_LITE,
            model_path="nova-lite/prod",
            instance_type="ml.g5.48xlarge",
            instance_count=1,
            data_s3_path=None,
            output_s3_path=self.output_s3_path,
            eval_task=EvaluationTask.MMLU,
            overrides={},
        )

        eval_config = builder._build_recipe_config()
        eval_config_dict = eval_config.to_dict()

        self.assertIsInstance(eval_config, EvalRecipeConfig)
        self.assertIsNone(eval_config.processor)
        self.assertIsNone(eval_config.rl_env)
        self.assertEqual(
            eval_config_dict,
            {
                "run": {
                    "name": "test-job",
                    "model_type": "amazon.nova-lite-v1:0:300k",
                    "model_name_or_path": "nova-lite/prod",
                    "data_s3_path": "",
                    "output_s3_path": "s3://test-bucket/output",
                    "replicas": 1,
                },
                "evaluation": {
                    "task": "mmlu",
                    "strategy": "zs_cot",
                    "metric": "accuracy",
                },
                "inference": {
                    "max_new_tokens": 8196,
                    "top_k": -1,
                    "top_p": 1.0,
                    "temperature": 0.0,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
