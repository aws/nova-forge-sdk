import json
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_customization_sdk.util.bedrock import invoke_model


class TestBedrock(unittest.TestCase):
    def test_invoke_model_success(self):
        mock_bedrock_runtime = MagicMock()
        mock_body = MagicMock()
        mock_response = {
            "body": mock_body,
            "ResponseMetadata": {"RequestId": "test-request-id"},
        }
        mock_body.read.return_value = b'{"response": "test output"}'
        mock_bedrock_runtime.invoke_model.return_value = mock_response

        model_id = "test-model"
        request_body = {"input": "test"}

        result = invoke_model(model_id, request_body, mock_bedrock_runtime)

        mock_bedrock_runtime.invoke_model.assert_called_once_with(
            modelId=model_id, body=json.dumps(request_body)
        )
        self.assertEqual(result.job_id, "test-request-id")
        self.assertEqual(result._nonstreaming_response, '{"response": "test output"}')

    def test_invoke_model_exception(self):
        mock_bedrock_runtime = MagicMock()
        mock_bedrock_runtime.invoke_model.side_effect = Exception("Test error")

        model_id = "test-model"
        request_body = {"input": "test"}

        with self.assertRaises(Exception) as context:
            invoke_model(model_id, request_body, mock_bedrock_runtime)

        self.assertTrue("Failed invoke Bedrock model" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
