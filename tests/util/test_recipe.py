import unittest
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from yaml import YAMLError

from amzn_nova_customization_sdk.util.recipe import (
    RecipeLoadError,
    _load_file_content,
    _parse_s3_uri,
    _validate_extension,
    get_all_key_names,
    get_all_type_hints,
    merge_overrides_with_input_recipe,
    resolve_overrides,
)


# Dummy enums and dataclasses for testing
class DummyEnum(Enum):
    A = "a"
    B = "b"


@dataclass
class Nested:
    field2: int
    enum_field: DummyEnum


@dataclass
class Root:
    field1: str
    nested: Nested


class TestRecipeUtil(unittest.TestCase):
    def test_parse_s3_uri_valid(self):
        uri = "s3://my-bucket/path/to/file.yaml"
        bucket, key = _parse_s3_uri(uri)
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.yaml")

    def test_parse_s3_uri_invalid(self):
        self.assertIsNone(_parse_s3_uri("not-an-s3-uri"))

    def test_validate_extension__no_exception_raised(self):
        _validate_extension("file.yaml", ".yaml")

    def test_validate_extension_failure(self):
        with self.assertRaises(RecipeLoadError):
            _validate_extension("file.txt", ".yaml")

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    def test_load_file_content_s3(self, mock_boto_client):
        content = "key: value"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": BytesIO(content.encode("utf-8"))}
        mock_boto_client.return_value = mock_s3

        with patch(
            "amzn_nova_customization_sdk.util.recipe._parse_s3_uri",
            return_value=("bucket", "key.yaml"),
        ):
            result = _load_file_content("s3://bucket/key.yaml")
            self.assertEqual(result, content)

    @patch("amzn_nova_customization_sdk.util.recipe.Path.read_text")
    def test_load_file_content_local(self, mock_read_text):
        mock_read_text.return_value = "field: value"
        with patch(
            "amzn_nova_customization_sdk.util.recipe._parse_s3_uri",
            return_value=None,
        ):
            result = _load_file_content("/tmp/file.yaml")
            self.assertEqual(result, "field: value")

    @patch("amzn_nova_customization_sdk.util.recipe.boto3.client")
    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri")
    def test_load_file_content_s3_client_error(
        self, mock_parse_s3_uri, mock_boto_client
    ):
        mock_parse_s3_uri.return_value = ("bucket", "key.yaml")
        mock_s3 = MagicMock()

        mock_s3.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            operation_name="GetObject",
        )
        mock_boto_client.return_value = mock_s3

        with self.assertRaises(RecipeLoadError) as context:
            _load_file_content("s3://bucket/key.yaml")

        self.assertIn("Failed to load S3 file", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri", return_value=None)
    def test_load_file_content_file_not_found(self, mock_parse_s3_uri):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            with self.assertRaises(RecipeLoadError) as context:
                _load_file_content("local_file.yaml")

            self.assertIn("File not found", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._parse_s3_uri", return_value=None)
    def test_load_file_content_os_error(self, mock_parse_s3_uri):
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            with self.assertRaises(RecipeLoadError) as context:
                _load_file_content("local_file.yaml")

            self.assertIn("Failed to read file", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._load_file_content")
    def test_merge_overrides(self, mock_load_file):
        mock_load_file.return_value = (
            "field1: hello\nnested:\n  field2: 42\n  enum_field: a"
        )
        overrides = {"field1": "override"}
        result = merge_overrides_with_input_recipe(
            recipe_path="dummy.yaml", recipe_class=Root, overrides=overrides
        )
        self.assertEqual(result["field1"], "override")  # Override takes priority
        self.assertEqual(result["field2"], 42)
        self.assertEqual(result["enum_field"], DummyEnum.A)

    def test_merge_overrides_invalid_enum(self) -> None:
        @dataclass
        class EnumRecipe:
            enum_field: DummyEnum

        overrides = {"enum_field": "invalid_value"}
        with patch(
            "amzn_nova_customization_sdk.util.recipe._load_file_content",
            return_value="enum_field: invalid_value",
        ):
            with self.assertRaises(ValueError) as context:
                merge_overrides_with_input_recipe("dummy.yaml", EnumRecipe, overrides)
            self.assertIn("Valid options are", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._load_file_content")
    @patch("amzn_nova_customization_sdk.util.recipe.yaml.safe_load")
    def test_yaml_error_raises_recipe_load_error(self, mock_safe_load, mock_load_file):
        mock_load_file.return_value = "invalid: yaml: content"
        mock_safe_load.side_effect = YAMLError("parsing failed")

        with self.assertRaises(RecipeLoadError) as context:
            merge_overrides_with_input_recipe(
                recipe_path="fake_path.yaml",
                recipe_class=dict,
                overrides={},
            )

        self.assertIn("Invalid YAML in fake_path.yaml", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe._load_file_content")
    def test_yaml_not_a_dict_raises_recipe_load_error(self, mock_load_file):
        # YAML content that is a list instead of a dict
        mock_load_file.return_value = "- item1\n- item2\n"

        with self.assertRaises(RecipeLoadError) as context:
            merge_overrides_with_input_recipe(
                recipe_path="fake_path.yaml",
                recipe_class=dict,
                overrides={},
            )

        self.assertIn("YAML must be a dictionary, got list", str(context.exception))

    @patch("amzn_nova_customization_sdk.util.recipe.merge_overrides_with_input_recipe")
    def test_resolve_overrides_with_recipe_path(self, mock_merge):
        mock_merge.return_value = {"field1": "hello"}
        result = resolve_overrides(
            Root, recipe_path="dummy.yaml", overrides={"field1": "value"}
        )
        self.assertEqual(result, {"field1": "hello"})
        mock_merge.assert_called_once()

    @patch("amzn_nova_customization_sdk.util.recipe.merge_overrides_with_input_recipe")
    @patch("amzn_nova_customization_sdk.util.recipe.logger")
    def test_resolve_overrides_with_recipe_path_no_overrides(
        self, mock_logger, mock_merge
    ):
        mock_merge.return_value = {"some_key": "some_value"}
        recipe_path = "fake_path.yaml"
        overrides = {}

        result = resolve_overrides(
            recipe_class=dict, recipe_path=recipe_path, overrides=overrides
        )

        mock_logger.info.assert_called_once_with(
            f"Recipe provided at {recipe_path}. Ignoring other user input in favor of the recipe content."
        )
        self.assertEqual(result, {"some_key": "some_value"})

    def test_resolve_overrides_without_recipe_path(self):
        result = resolve_overrides(Root, overrides={"field1": "value"})
        self.assertEqual(result, {"field1": "value"})

    def test_get_all_type_hints_flat(self):
        hints = get_all_type_hints(Root)
        self.assertIn("field1", hints)
        self.assertIn("field2", hints)
        self.assertIn("enum_field", hints)
        self.assertEqual(hints["enum_field"], DummyEnum)

    def test_get_all_key_names_flat(self):
        keys = get_all_key_names(Root)
        self.assertIn("field1", keys)
        self.assertIn("field2", keys)
        self.assertIn("enum_field", keys)


if __name__ == "__main__":
    unittest.main()
