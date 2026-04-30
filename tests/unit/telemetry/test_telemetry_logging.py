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
"""Tests for telemetry logging module."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from amzn_nova_forge.core.enums import Platform, TrainingMethod
from amzn_nova_forge.telemetry.constants import (
    DEFAULT_AWS_REGION,
    PLATFORM_TO_CODE,
    TELEMETRY_OPT_OUT_MESSAGING,
    TRAINING_METHOD_TO_CODE,
    Feature,
    Status,
)

# Opt out of the global autouse telemetry mock — these tests exercise
# the telemetry module itself and manage their own mocks.
pytestmark = pytest.mark.no_auto_mock_telemetry
from amzn_nova_forge.telemetry.telemetry_logging import (
    _construct_url,
    _get_accountId,
    _is_telemetry_opted_out,
    _requests_helper,
    _send_telemetry_request,
    _telemetry_emitter,
)


class TestIsTelemetryOptedOut(unittest.TestCase):
    """Tests for _is_telemetry_opted_out."""

    @patch.dict("os.environ", {"TELEMETRY_OPT_OUT": "true"})
    def test_opted_out_lowercase_true(self):
        """Returns True when env var is 'true'."""
        self.assertTrue(_is_telemetry_opted_out())

    @patch.dict("os.environ", {"TELEMETRY_OPT_OUT": "True"})
    def test_opted_out_capitalized_true(self):
        """Returns True when env var is 'True'."""
        self.assertTrue(_is_telemetry_opted_out())

    @patch.dict("os.environ", {"TELEMETRY_OPT_OUT": "FALSE"})
    def test_not_opted_out_false(self):
        """Returns False when env var is 'FALSE'."""
        self.assertFalse(_is_telemetry_opted_out())

    @patch.dict("os.environ", {"TELEMETRY_OPT_OUT": "False"})
    def test_not_opted_out_default(self):
        """Returns False when env var is 'False'."""
        self.assertFalse(_is_telemetry_opted_out())

    @patch.dict("os.environ", {}, clear=True)
    def test_not_opted_out_when_env_var_missing(self):
        """Returns False when env var is not set."""
        self.assertFalse(_is_telemetry_opted_out())

    @patch.dict("os.environ", {"TELEMETRY_OPT_OUT": "yes"})
    def test_not_opted_out_for_non_true_string(self):
        """Returns False for non-'true' strings like 'yes'."""
        self.assertFalse(_is_telemetry_opted_out())


class TestConstructUrl(unittest.TestCase):
    """Tests for _construct_url."""

    def test_basic_url_construction(self):
        """Constructs URL with required parameters only."""
        url = _construct_url(
            accountId="123456789012",
            region="us-east-1",
            status="1",
            feature="1",
        )
        self.assertIn("x-accountId=123456789012", url)
        self.assertIn("x-status=1", url)
        self.assertIn("x-feature=1", url)
        self.assertIn("us-east-1", url)
        self.assertNotIn("x-failureType", url)

    def test_url_with_failure_info(self):
        """Includes failure type when provided."""
        url = _construct_url(
            accountId="123456789012",
            region="us-west-2",
            status="0",
            feature="11",
            failure_type="ValueError",
        )
        self.assertIn("x-failureType=ValueError", url)

    def test_url_with_extra_info(self):
        """Appends extra info string to URL."""
        extra = "&x-func=train&x-sdkVersion=1.3"
        url = _construct_url(
            accountId="123456789012",
            region="us-east-1",
            status="1",
            feature="1",
            extra_info=extra,
        )
        self.assertIn("x-func=train", url)
        self.assertIn("x-sdkVersion=1.3", url)

    def test_url_without_optional_params(self):
        """Does not include optional params when None."""
        url = _construct_url(
            accountId="123456789012",
            region="us-east-1",
            status="1",
            feature="1",
            failure_type=None,
            extra_info=None,
        )
        self.assertNotIn("failureType", url)


class TestRequestsHelper(unittest.TestCase):
    """Tests for _requests_helper."""

    @patch("amzn_nova_forge.telemetry.telemetry_logging.requests")
    def test_successful_request(self, mock_requests):
        """Returns response on successful GET."""
        mock_response = Mock()
        mock_requests.get.return_value = mock_response

        result = _requests_helper("https://example.com", 2)

        mock_requests.get.assert_called_once_with("https://example.com", timeout=2)
        self.assertEqual(result, mock_response)

    @patch("amzn_nova_forge.telemetry.telemetry_logging.requests")
    def test_request_exception_returns_none(self, mock_requests):
        """Returns None when request raises RequestException."""
        import requests as real_requests

        mock_requests.exceptions.RequestException = real_requests.exceptions.RequestException
        mock_requests.get.side_effect = real_requests.exceptions.RequestException("timeout")

        result = _requests_helper("https://example.com", timeout=2)

        self.assertIsNone(result)


class TestGetAccountId(unittest.TestCase):
    """Tests for _get_accountId."""

    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_returns_account_id(self, mock_boto3):
        """Returns account ID from STS."""
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto3.client.return_value = mock_sts

        result = _get_accountId()

        self.assertEqual(result, "123456789012")
        mock_boto3.client.assert_called_once_with("sts")

    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_returns_none_on_exception(self, mock_boto3):
        """Returns None when STS call fails."""
        mock_boto3.client.side_effect = Exception("no credentials")

        result = _get_accountId()

        self.assertIsNone(result)


class TestSendTelemetryRequest(unittest.TestCase):
    """Tests for _send_telemetry_request."""

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._construct_url")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        return_value="123456789012",
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_sends_request_for_supported_region(self, mock_boto3, mock_get_id, mock_url, mock_req):
        """Sends telemetry when region is supported."""
        mock_boto3.session.Session.return_value.region_name = "us-east-1"
        mock_url.return_value = "https://telemetry-url"

        _send_telemetry_request(1, 1)

        mock_url.assert_called_once()
        mock_req.assert_called_once_with("https://telemetry-url", 2)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        return_value="123456789012",
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_skips_unsupported_region(self, mock_boto3, mock_get_id, mock_req):
        """Does not send telemetry for unsupported region."""
        mock_boto3.session.Session.return_value.region_name = "ap-southeast-1"
        _send_telemetry_request(1, 1)

        mock_req.assert_not_called()

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._construct_url")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        return_value="123456789012",
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-west-2": "123456789012"},
    )
    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_resolves_region_from_boto3_session(self, mock_boto3, mock_get_id, mock_url, mock_req):
        """Resolves region dynamically from boto3 session."""
        mock_boto3.session.Session.return_value.region_name = "us-west-2"
        mock_url.return_value = "https://telemetry-url"

        _send_telemetry_request(1, 1)

        call_args = mock_url.call_args
        self.assertEqual(call_args[0][1], "us-west-2")

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._construct_url")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        return_value="123456789012",
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_falls_back_to_default_when_session_has_no_region(
        self, mock_boto3, mock_get_id, mock_url, mock_req
    ):
        """Falls back to DEFAULT_AWS_REGION when boto3 session region is None."""
        mock_boto3.session.Session.return_value.region_name = None
        mock_url.return_value = "https://telemetry-url"

        _send_telemetry_request(1, 1)

        call_args = mock_url.call_args
        self.assertEqual(call_args[0][1], DEFAULT_AWS_REGION)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        side_effect=Exception("boom"),
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    def test_does_not_raise_on_exception(self, mock_get_id, mock_req):
        """Silently handles exceptions without propagating."""
        # Should not raise
        _send_telemetry_request(1, 1)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._requests_helper")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._construct_url")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._get_accountId",
        return_value="123456789012",
    )
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging.REGION_TO_ESCROW_ACCOUNT_MAPPING",
        {"us-east-1": "123456789012"},
    )
    @patch("amzn_nova_forge.telemetry.telemetry_logging.boto3")
    def test_passes_failure_info_to_construct_url(
        self, mock_boto3, mock_get_id, mock_url, mock_req
    ):
        """Forwards failure_type to _construct_url."""
        mock_boto3.session.Session.return_value.region_name = "us-east-1"
        mock_url.return_value = "https://telemetry-url"

        _send_telemetry_request(0, 1, failure_type="ValueError")

        call_args = mock_url.call_args
        self.assertEqual(call_args[0][4], "ValueError")


class TestTelemetryEmitter(unittest.TestCase):
    """Tests for _telemetry_emitter decorator."""

    def setUp(self):
        # Reset the global notice flag before each test
        import amzn_nova_forge.telemetry.telemetry_logging as tl

        tl._telemetry_notice_shown = False

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_emits_success_telemetry(self, mock_opt_out, mock_send):
        """Emits success telemetry when wrapped function succeeds."""

        @_telemetry_emitter(Feature.TRAINING, "train")
        def my_func():
            return "result"

        result = my_func()

        self.assertEqual(result, "result")
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        self.assertEqual(call_args[0][0], Status.SUCCESS.value)
        self.assertEqual(call_args[0][1], Feature.TRAINING.value)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_emits_failure_telemetry_and_reraises(self, mock_opt_out, mock_send):
        """Emits failure telemetry and re-raises when wrapped function raises."""

        @_telemetry_emitter(Feature.EVAL, "evaluate")
        def my_func():
            raise ValueError("bad input")

        with self.assertRaises(ValueError) as ctx:
            my_func()

        self.assertIn("bad input", str(ctx.exception))
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        self.assertEqual(call_args[0][0], Status.FAILURE.value)
        self.assertEqual(call_args[0][2], "ValueError")

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=True,
    )
    def test_skips_telemetry_when_opted_out(self, mock_opt_out, mock_send):
        """Does not send telemetry when opted out."""

        @_telemetry_emitter(Feature.DEPLOY, "deploy")
        def my_func():
            return "deployed"

        result = my_func()

        self.assertEqual(result, "deployed")
        mock_send.assert_not_called()

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=True,
    )
    def test_skips_telemetry_on_failure_when_opted_out(self, mock_opt_out, mock_send):
        """Does not send telemetry on failure when opted out, but still re-raises."""

        @_telemetry_emitter(Feature.TRAINING, "train")
        def my_func():
            raise RuntimeError("fail")

        with self.assertRaises(RuntimeError):
            my_func()

        mock_send.assert_not_called()

    @patch("amzn_nova_forge.telemetry.telemetry_logging.logger")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_shows_notice_on_first_call(self, mock_opt_out, mock_send, mock_logger):
        """Logs telemetry opt-out notice on first invocation."""

        @_telemetry_emitter(Feature.TRAINING, "train")
        def my_func():
            return "ok"

        my_func()

        mock_logger.info.assert_any_call(TELEMETRY_OPT_OUT_MESSAGING)

    @patch("amzn_nova_forge.telemetry.telemetry_logging.logger")
    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_shows_notice_only_once(self, mock_opt_out, mock_send, mock_logger):
        """Logs telemetry notice only on the first call, not subsequent ones."""

        @_telemetry_emitter(Feature.TRAINING, "train")
        def my_func():
            return "ok"

        my_func()
        mock_logger.reset_mock()
        my_func()

        # Should not log the notice again
        for call in mock_logger.info.call_args_list:
            self.assertNotEqual(call[0][0], TELEMETRY_OPT_OUT_MESSAGING)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_preserves_function_args(self, mock_opt_out, mock_send):
        """Passes args and kwargs through to the wrapped function."""

        @_telemetry_emitter(Feature.DATA_PREP, "prep")
        def my_func(a, b, key=None):
            return (a, b, key)

        result = my_func(1, 2, key="val")

        self.assertEqual(result, (1, 2, "val"))

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_extra_info_contains_func_name(self, mock_opt_out, mock_send):
        """Extra info string includes the function name."""

        @_telemetry_emitter(Feature.MONITOR, "my_monitor_func")
        def my_func():
            return "ok"

        my_func()

        call_args = mock_send.call_args
        extra_info = call_args[0][3]
        self.assertIn("x-func=my_monitor_func", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_extra_info_fn_appends_custom_fields(self, mock_opt_out, mock_send):
        """extra_info_fn return values are appended to the extra string."""

        @_telemetry_emitter(
            Feature.TRAINING,
            "train",
            extra_info_fn=lambda method, model, platform: {
                "method": method,
                "model": model,
                "platform": platform,
            },
        )
        def my_func(method, model, platform):
            return "ok"

        my_func(TrainingMethod.SFT_LORA, "nova-micro", Platform.SMTJ)

        call_args = mock_send.call_args
        extra_info = call_args[0][3]
        self.assertIn("x-method=8", extra_info)
        self.assertIn("x-model=nova-micro", extra_info)
        self.assertIn("x-platform=1", extra_info)
        self.assertIn("x-func=train", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_extra_info_fn_exception_does_not_break_telemetry(self, mock_opt_out, mock_send):
        """If extra_info_fn raises, telemetry is still emitted without the extra fields."""

        @_telemetry_emitter(
            Feature.TRAINING,
            "train",
            extra_info_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        def my_func():
            return "ok"

        result = my_func()

        self.assertEqual(result, "ok")
        mock_send.assert_called_once()
        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-func=train", extra_info)
        self.assertNotIn("x-method", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_extra_info_fn_none_returns_no_extra_fields(self, mock_opt_out, mock_send):
        """When extra_info_fn returns None, no additional fields are appended."""

        @_telemetry_emitter(
            Feature.TRAINING,
            "train",
            extra_info_fn=lambda: None,
        )
        def my_func():
            return "ok"

        my_func()

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-func=train", extra_info)
        self.assertIn("x-sdkVersion=", extra_info)


class TestFeatureAndStatusMappings(unittest.TestCase):
    """Tests for enum value codes and code-mapping dictionaries."""

    def test_all_features_have_int_values(self):
        """Every Feature enum member has an integer value."""
        for feature in Feature:
            self.assertIsInstance(feature.value, int)

    def test_all_statuses_have_int_values(self):
        """Every Status enum member has an integer value."""
        for status in Status:
            self.assertIsInstance(status.value, int)

    def test_all_platforms_mapped(self):
        """Every Platform enum member has a mapping in PLATFORM_TO_CODE."""
        for plat in Platform:
            self.assertIn(plat, PLATFORM_TO_CODE)

    def test_all_deploy_platforms_mapped(self):
        """Every DeployPlatform enum member has a mapping in PLATFORM_TO_CODE."""
        from amzn_nova_forge.core.enums import DeployPlatform

        for plat in DeployPlatform:
            self.assertIn(plat, PLATFORM_TO_CODE)

    def test_all_training_methods_mapped(self):
        """Every TrainingMethod enum member has a mapping in TRAINING_METHOD_TO_CODE."""
        for method in TrainingMethod:
            self.assertIn(method, TRAINING_METHOD_TO_CODE)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_deploy_platform_bedrock_od_converted_to_bedrock_code(self, mock_opt_out, mock_send):
        """DeployPlatform.BEDROCK_OD is mapped to its own code (11)."""
        from amzn_nova_forge.core.enums import DeployPlatform

        @_telemetry_emitter(
            Feature.DEPLOY,
            "deploy",
            extra_info_fn=lambda platform: {"platform": platform},
        )
        def my_func(platform):
            return "ok"

        my_func(DeployPlatform.BEDROCK_OD)

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-platform=11", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_deploy_platform_bedrock_pt_converted_to_bedrock_code(self, mock_opt_out, mock_send):
        """DeployPlatform.BEDROCK_PT is mapped to its own code (12)."""
        from amzn_nova_forge.core.enums import DeployPlatform

        @_telemetry_emitter(
            Feature.DEPLOY,
            "deploy",
            extra_info_fn=lambda platform: {"platform": platform},
        )
        def my_func(platform):
            return "ok"

        my_func(DeployPlatform.BEDROCK_PT)

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-platform=12", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_deploy_platform_sagemaker_converted_to_sagemaker_code(self, mock_opt_out, mock_send):
        """DeployPlatform.SAGEMAKER is mapped to its own code (13)."""
        from amzn_nova_forge.core.enums import DeployPlatform

        @_telemetry_emitter(
            Feature.DEPLOY,
            "deploy",
            extra_info_fn=lambda platform: {"platform": platform},
        )
        def my_func(platform):
            return "ok"

        my_func(DeployPlatform.SAGEMAKER)

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-platform=13", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_dry_run_true_converted_to_1(self, mock_opt_out, mock_send):
        """dryRun=True is emitted as x-dryRun=1 on the wire."""

        @_telemetry_emitter(
            Feature.TRAINING,
            "train",
            extra_info_fn=lambda dry_run: {"dryRun": dry_run},
        )
        def my_func(dry_run):
            return "ok"

        my_func(True)

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-dryRun=1", extra_info)
        self.assertNotIn("x-dryRun=True", extra_info)

    @patch("amzn_nova_forge.telemetry.telemetry_logging._send_telemetry_request")
    @patch(
        "amzn_nova_forge.telemetry.telemetry_logging._is_telemetry_opted_out",
        return_value=False,
    )
    def test_dry_run_false_converted_to_0(self, mock_opt_out, mock_send):
        """dryRun=False is emitted as x-dryRun=0 on the wire."""

        @_telemetry_emitter(
            Feature.TRAINING,
            "train",
            extra_info_fn=lambda dry_run: {"dryRun": dry_run},
        )
        def my_func(dry_run):
            return "ok"

        my_func(False)

        extra_info = mock_send.call_args[0][3]
        self.assertIn("x-dryRun=0", extra_info)
        self.assertNotIn("x-dryRun=False", extra_info)
