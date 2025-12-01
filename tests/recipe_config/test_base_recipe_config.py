import unittest
from dataclasses import dataclass
from enum import Enum

from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
    to_primitive,
)


class DummyEnum(Enum):
    A = "value-a"
    B = "value-b"


@dataclass
class NestedDataclass:
    x: int
    y: str
    z: None = None


class TestToPrimitive(unittest.TestCase):
    def test_to_primitive_enum(self):
        self.assertEqual(to_primitive(DummyEnum.A), "value-a")

    def test_to_primitive_dataclass_simple(self):
        result = to_primitive(NestedDataclass(x=1, y="test"))
        self.assertEqual(result, {"x": 1, "y": "test"})  # z omitted because None

    def test_to_primitive_nested_dataclass(self):
        outer = {"config": NestedDataclass(x=5, y="inner")}
        result = to_primitive(outer)
        self.assertEqual(result, {"config": {"x": 5, "y": "inner"}})

    def test_to_primitive_list(self):
        result = to_primitive([1, DummyEnum.B, None])
        self.assertEqual(result, [1, "value-b"])

    def test_to_primitive_tuple(self):
        result = to_primitive((DummyEnum.A, None, 3))
        self.assertEqual(result, ["value-a", 3])

    def test_to_primitive_dict(self):
        result = to_primitive({"a": 1, "b": None, "c": DummyEnum.A})
        self.assertEqual(result, {"a": 1, "c": "value-a"})

    def test_to_primitive_scalar(self):
        self.assertEqual(to_primitive(123), 123)
        self.assertEqual(to_primitive("abc"), "abc")


class TestBaseRunConfig(unittest.TestCase):
    def test_run_config_basic(self):
        config = BaseRunConfig(
            name="test-job",
            model_type="amazon.nova-micro-v1:0:128k",
            model_name_or_path="nova-micro/prod",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=2,
        )

        self.assertEqual(config.name, "test-job")
        self.assertEqual(config.model_type, "amazon.nova-micro-v1:0:128k")
        self.assertEqual(config.model_name_or_path, "nova-micro/prod")
        self.assertEqual(config.data_s3_path, "s3://bucket/data")
        self.assertEqual(config.output_s3_path, "s3://bucket/output")
        self.assertEqual(config.replicas, 2)

    def test_run_config_to_dict_excludes_none(self):
        config = BaseRunConfig(
            name="job",
            model_type="t",
            model_name_or_path="path",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=1,
        )

        result = config.to_dict()
        self.assertEqual(result["name"], "job")
        self.assertEqual(result["model_type"], "t")
        self.assertEqual(result["model_name_or_path"], "path")
        self.assertEqual(result["data_s3_path"], "s3://bucket/data")
        self.assertEqual(result["output_s3_path"], "s3://bucket/output")
        self.assertEqual(result["replicas"], 1)
        self.assertNotIn("fake_field", result)


class TestBaseRecipeConfig(unittest.TestCase):
    def test_recipe_config_basic(self):
        run_config = BaseRunConfig(
            name="job",
            model_type="type",
            model_name_or_path="path",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=1,
        )

        recipe = BaseRecipeConfig(run=run_config)
        self.assertIsInstance(recipe.run, BaseRunConfig)

    def test_recipe_config_to_dict(self):
        run_config = BaseRunConfig(
            name="job",
            model_type="type",
            model_name_or_path="path",
            data_s3_path="s3://bucket/data",
            output_s3_path="s3://bucket/output",
            replicas=1,
        )
        recipe = BaseRecipeConfig(run=run_config)

        result = recipe.to_dict()

        self.assertIn("run", result)
        self.assertEqual(result["run"]["name"], "job")
        self.assertEqual(result["run"]["model_type"], "type")
        self.assertEqual(result["run"]["model_name_or_path"], "path")
        self.assertEqual(result["run"]["data_s3_path"], "s3://bucket/data")
        self.assertEqual(result["run"]["output_s3_path"], "s3://bucket/output")
        self.assertEqual(result["run"]["replicas"], 1)
        self.assertNotIn("fake", result)


if __name__ == "__main__":
    unittest.main()
