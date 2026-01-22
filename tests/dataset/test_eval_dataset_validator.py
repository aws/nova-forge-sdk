import json
import tempfile
import unittest

from amzn_nova_customization_sdk.dataset.dataset_loader import (
    JSONLDatasetLoader,
)
from amzn_nova_customization_sdk.model.model_enums import Model, TrainingMethod


class TestEvalDatasetValidator(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory before each test for storing test data."""
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_file(self, name: str, data):
        temp_file_name = f"{self.temp_dir}/{name}.jsonl"
        with open(temp_file_name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return temp_file_name

    def test_eval_text_success(self):
        eval_text_success = [
            {
                "system": "You are an English major with top marks in class who likes to give minimal word responses: ",
                "query": "What is the symbol that ends the sentence as a question",
                "response": "?",
            }
        ]

        test_file = self.create_temp_file("eval_text_success", eval_text_success)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.EVALUATION, Model.NOVA_LITE
        )

    def test_eval_text_fail(self):
        eval_text_fail = [
            {
                "system": "You are an English major with top marks in class who likes to give minimal word responses: ",
                "query": "What is the symbol that ends the sentence as a question",
                "not right": "?",
            }
        ]

        test_file = self.create_temp_file("eval_text_fail", eval_text_fail)
        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.EVALUATION, Model.NOVA_LITE)

    def test_eval_img_success(self):
        eval_img_success = [
            {
                "system": "Image inference: ",
                "query": "What is the number in the image? Please just use one English word to answer.",
                "response": "two",
                "images": [{"data": "data:image/png;Base64,iVBORw0KGgoA"}],
            }
        ]
        test_file = self.create_temp_file("eval_img_success", eval_img_success)
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.EVALUATION, Model.NOVA_LITE
        )

    def test_eval_img_fail(self):
        eval_img_fail = [
            {
                "system": "Image inference: ",
                "query": "What is the number in the image? Please just use one English word to answer.",
                "response": "two",
                "images": [{"data": "data:image/png;WRONG,iVBORw0KGgoA"}],
            }
        ]

        test_file = self.create_temp_file("eval_img_fail", eval_img_fail)
        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.EVALUATION, Model.NOVA_LITE)

    def test_eval_metadata_success(self):
        eval_metadata_success = [
            {
                "system": "Image inference: ",
                "query": "What is the number in the image? Please just use one English word to answer.",
                "response": "two",
                "metadata": "hello",
            }
        ]
        test_file = self.create_temp_file(
            "eval_metadata_success", eval_metadata_success
        )
        JSONLDatasetLoader().load(test_file).validate(
            TrainingMethod.EVALUATION, Model.NOVA_LITE
        )

    def test_eval_metadata_fail(self):
        eval_metadata_fail = [
            {
                "system": "Image inference: ",
                "query": "What is the number in the image? Please just use one English word to answer.",
                "response": "two",
                "metadata": "<image> bad response",
            }
        ]
        test_file = self.create_temp_file("eval_metadata_fail", eval_metadata_fail)
        dataset_loader = JSONLDatasetLoader().load(test_file)
        with self.assertRaises(ValueError):
            dataset_loader.validate(TrainingMethod.EVALUATION, Model.NOVA_LITE)

    def tearDown(self):
        """Clean up temporary files created during each unit test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
