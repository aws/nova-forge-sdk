"""Unit tests for amzn_nova_forge.core.job_cache."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.core.job_cache import (
    JobCacheContext,
    build_cache_context,
    collect_all_parameters,
    generate_job_hash,
    load_existing_result,
    matches_job_cache_criteria,
    persist_result,
    should_persist_results,
)
from amzn_nova_forge.core.result.job_result import JobStatus
from amzn_nova_forge.core.types import ForgeConfig


class TestJobCacheContext(unittest.TestCase):
    def test_default_config(self):
        ctx = JobCacheContext(enable_job_caching=True)
        self.assertIsNotNone(ctx.job_caching_config)
        self.assertTrue(ctx.job_caching_config["include_core"])
        self.assertTrue(ctx.job_caching_config["include_recipe"])
        self.assertFalse(ctx.job_caching_config["include_infra"])
        self.assertEqual(ctx.job_caching_config["include_params"], [])
        self.assertEqual(ctx.job_caching_config["exclude_params"], [])
        self.assertIn(JobStatus.COMPLETED, ctx.job_caching_config["allowed_statuses"])
        self.assertIn(JobStatus.IN_PROGRESS, ctx.job_caching_config["allowed_statuses"])

    def test_custom_config_preserved(self):
        custom = {"include_core": False, "allowed_statuses": [JobStatus.COMPLETED]}
        ctx = JobCacheContext(enable_job_caching=True, job_caching_config=custom)
        self.assertFalse(ctx.job_caching_config["include_core"])


class TestShouldPersistResults(unittest.TestCase):
    def test_enabled_with_valid_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = JobCacheContext(enable_job_caching=True, job_cache_dir=tmp)
            self.assertTrue(should_persist_results(ctx))

    def test_disabled(self):
        ctx = JobCacheContext(enable_job_caching=False)
        self.assertFalse(should_persist_results(ctx))

    def test_empty_dir(self):
        ctx = JobCacheContext(enable_job_caching=True, job_cache_dir="")
        self.assertFalse(should_persist_results(ctx))

    def test_creates_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = str(Path(tmp) / "new_cache")
            ctx = JobCacheContext(enable_job_caching=True, job_cache_dir=cache_dir)
            self.assertTrue(should_persist_results(ctx))
            self.assertTrue(Path(cache_dir).exists())


class TestGenerateJobHash(unittest.TestCase):
    def test_consistency(self):
        ctx = JobCacheContext(
            enable_job_caching=True,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            data_s3_path="s3://bucket/data",
        )
        h1 = generate_job_hash(ctx, "job1", "train")
        h2 = generate_job_hash(ctx, "job1", "train")
        self.assertEqual(h1, h2)

    def test_different_model(self):
        ctx1 = JobCacheContext(enable_job_caching=True, model=Model.NOVA_LITE_2)
        ctx2 = JobCacheContext(enable_job_caching=True, model=Model.NOVA_PRO)
        h1 = generate_job_hash(ctx1, "job", "train")
        h2 = generate_job_hash(ctx2, "job", "train")
        self.assertNotEqual(h1, h2)

    def test_overrides_affect_hash(self):
        ctx = JobCacheContext(enable_job_caching=True, model=Model.NOVA_LITE_2)
        h1 = generate_job_hash(ctx, "job", "train", overrides={"lr": 0.01})
        h2 = generate_job_hash(ctx, "job", "train", overrides={"lr": 0.001})
        self.assertNotEqual(h1, h2)

    def test_infra_fields_included(self):
        ctx1 = JobCacheContext(
            enable_job_caching=True,
            model=Model.NOVA_LITE_2,
            instance_type="ml.p5.48xlarge",
            instance_count=2,
        )
        ctx2 = JobCacheContext(
            enable_job_caching=True,
            model=Model.NOVA_LITE_2,
            instance_type="ml.g5.12xlarge",
            instance_count=1,
        )
        h1 = generate_job_hash(ctx1, "job", "train")
        h2 = generate_job_hash(ctx2, "job", "train")
        self.assertNotEqual(h1, h2)
        self.assertIn("instance_type:", h1)
        self.assertIn("instance_count:", h1)


class TestMatchesJobCacheCriteria(unittest.TestCase):
    def test_exact_match(self):
        h = "model:abc,method:def,job_type:ghi"
        config = {"include_core": True, "include_recipe": True, "include_infra": False}
        self.assertTrue(matches_job_cache_criteria(config, h, h))

    def test_mismatch(self):
        h1 = "model:abc,method:def"
        h2 = "model:xyz,method:def"
        config = {"include_core": True}
        self.assertFalse(matches_job_cache_criteria(config, h1, h2))

    def test_exclude_params(self):
        h1 = "model:abc,method:def,custom:111"
        h2 = "model:abc,method:def,custom:222"
        config = {"include_core": True, "exclude_params": ["custom"]}
        self.assertTrue(matches_job_cache_criteria(config, h1, h2))

    def test_include_infra(self):
        h1 = "model:abc,instance_type:p5"
        h2 = "model:abc,instance_type:g5"
        config = {"include_core": True, "include_infra": True}
        self.assertFalse(matches_job_cache_criteria(config, h1, h2))


class TestPersistAndLoad(unittest.TestCase):
    @patch("amzn_nova_forge.core.job_cache.BaseJobResult.load")
    def test_roundtrip(self, mock_load):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = JobCacheContext(
                enable_job_caching=True,
                job_cache_dir=tmp,
                model=Model.NOVA_LITE_2,
                method=TrainingMethod.SFT_LORA,
                data_s3_path="s3://bucket/data",
            )

            mock_result = MagicMock()
            mock_result.__class__.__name__ = "SMTJTrainingResult"
            mock_result._to_dict.return_value = {
                "job_id": "job-123",
                "started_time": "2024-01-01T00:00:00",
            }

            persist_result(ctx, mock_result, job_name="test-job", job_type="train")

            files = list(Path(tmp).glob("test-job_train_*.json"))
            self.assertEqual(len(files), 1)

            with open(files[0]) as f:
                data = json.load(f)
            self.assertIn("_job_cache_hash", data)
            self.assertIn("__class_name__", data)

    def test_load_returns_none_when_disabled(self):
        ctx = JobCacheContext(enable_job_caching=False)
        result = load_existing_result(ctx, "job", "train")
        self.assertIsNone(result)

    def test_load_returns_none_when_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = JobCacheContext(
                enable_job_caching=True,
                job_cache_dir=tmp,
                model=Model.NOVA_LITE_2,
            )
            result = load_existing_result(ctx, "nonexistent", "train")
            self.assertIsNone(result)


class TestCollectAllParameters(unittest.TestCase):
    def test_collects_fields(self):
        ctx = JobCacheContext(
            enable_job_caching=True,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            data_s3_path="s3://bucket/data",
            instance_type="ml.p5.48xlarge",
            instance_count=2,
        )
        params = collect_all_parameters(ctx, "job", "train", extra_key="extra_val")
        self.assertEqual(params["model"], Model.NOVA_LITE_2.value)
        self.assertEqual(params["method"], TrainingMethod.SFT_LORA.value)
        self.assertEqual(params["data_s3_path"], "s3://bucket/data")
        self.assertEqual(params["infra_instance_type"], "ml.p5.48xlarge")
        self.assertEqual(params["infra_instance_count"], 2)
        self.assertEqual(params["extra_key"], "extra_val")


class TestBuildCacheContext(unittest.TestCase):
    def test_from_forge_config(self):
        config = ForgeConfig(enable_job_caching=True, job_cache_dir="/tmp/cache")
        ctx = build_cache_context(
            config,
            model=Model.NOVA_LITE_2,
            method=TrainingMethod.SFT_LORA,
            data_s3_path="s3://bucket/data",
            instance_type="ml.p5.48xlarge",
        )
        self.assertTrue(ctx.enable_job_caching)
        self.assertEqual(ctx.job_cache_dir, "/tmp/cache")
        self.assertEqual(ctx.model, Model.NOVA_LITE_2)
        self.assertEqual(ctx.method, TrainingMethod.SFT_LORA)
        self.assertEqual(ctx.instance_type, "ml.p5.48xlarge")

    def test_disabled_by_default(self):
        config = ForgeConfig()
        ctx = build_cache_context(config)
        self.assertFalse(ctx.enable_job_caching)

    def test_job_caching_config_forwarded(self):
        custom_config = {
            "include_core": True,
            "include_recipe": False,
            "include_infra": True,
            "allowed_statuses": [JobStatus.COMPLETED],
        }
        config = ForgeConfig(
            enable_job_caching=True,
            job_caching_config=custom_config,
        )
        ctx = build_cache_context(config, model=Model.NOVA_LITE_2)
        self.assertIs(ctx.job_caching_config, custom_config)
        self.assertFalse(ctx.job_caching_config["include_recipe"])
        self.assertTrue(ctx.job_caching_config["include_infra"])

    def test_job_caching_config_defaults_when_none(self):
        config = ForgeConfig(enable_job_caching=True)
        ctx = build_cache_context(config)
        self.assertIsNotNone(ctx.job_caching_config)
        self.assertTrue(ctx.job_caching_config["include_core"])


class TestLoadExistingResultValidation(unittest.TestCase):
    """Tests for the allowed_statuses validation fix (assert → if check)."""

    def test_invalid_allowed_statuses_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = JobCacheContext(
                enable_job_caching=True,
                job_cache_dir=tmp,
                job_caching_config={
                    "include_core": True,
                    "allowed_statuses": "not-a-list",
                },
            )
            result = load_existing_result(ctx, "job", "train")
            self.assertIsNone(result)

    def test_none_allowed_statuses_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = JobCacheContext(
                enable_job_caching=True,
                job_cache_dir=tmp,
                job_caching_config={
                    "include_core": True,
                    "allowed_statuses": None,
                },
            )
            result = load_existing_result(ctx, "job", "train")
            self.assertIsNone(result)


class TestModelPathInCacheKey(unittest.TestCase):
    """Tests that model_path passed as job_param affects the cache hash."""

    def test_different_model_path_produces_different_hash(self):
        ctx = JobCacheContext(
            enable_job_caching=True,
            model=Model.NOVA_LITE_2,
        )
        h1 = generate_job_hash(ctx, "job", "eval", model_path="s3://bucket/checkpoint-a")
        h2 = generate_job_hash(ctx, "job", "eval", model_path="s3://bucket/checkpoint-b")
        self.assertNotEqual(h1, h2)
        self.assertIn("model_path:", h1)
        self.assertIn("model_path:", h2)


if __name__ == "__main__":
    unittest.main()
