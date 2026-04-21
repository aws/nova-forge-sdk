import json
import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.core.enums import ModelStatus
from amzn_nova_forge.util.bedrock import invoke_model, wait_for_model_ready


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


class TestWaitForModelReady(unittest.TestCase):
    MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/test"

    def test_already_active(self):
        client = MagicMock()
        client.get_custom_model.return_value = {"modelStatus": "Active"}
        result = wait_for_model_ready(client, self.MODEL_ARN)
        self.assertEqual(result, ModelStatus.ACTIVE)
        client.get_custom_model.assert_called_once()

    @patch("amzn_nova_forge.util.bedrock.time.sleep")
    def test_creating_then_active(self, mock_sleep):
        client = MagicMock()
        client.get_custom_model.side_effect = [
            {"modelStatus": "Creating"},
            {"modelStatus": "Active"},
        ]
        result = wait_for_model_ready(client, self.MODEL_ARN, poll_interval=1)
        self.assertEqual(result, ModelStatus.ACTIVE)
        self.assertEqual(client.get_custom_model.call_count, 2)

    def test_failed_raises(self):
        client = MagicMock()
        client.get_custom_model.return_value = {"modelStatus": "Failed"}
        with self.assertRaises(ValueError) as ctx:
            wait_for_model_ready(client, self.MODEL_ARN)
        self.assertIn("FAILED", str(ctx.exception))

    @patch("amzn_nova_forge.util.bedrock.time.sleep")
    @patch("amzn_nova_forge.util.bedrock.time.time")
    def test_timeout(self, mock_time, mock_sleep):
        mock_time.side_effect = [0, 0, 0, 1000]
        client = MagicMock()
        client.get_custom_model.return_value = {"modelStatus": "Creating"}
        with self.assertRaises(TimeoutError):
            wait_for_model_ready(client, self.MODEL_ARN, timeout=10)


if __name__ == "__main__":
    unittest.main()
