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
import logging
import unittest

from amzn_nova_forge.util.subprocess_utils import _check_hyperpod_stderr


class TestCheckHyperpodStderr(unittest.TestCase):
    """Tests for _check_hyperpod_stderr utility function."""

    def test_empty_string(self):
        _check_hyperpod_stderr("")

    def test_none(self):
        _check_hyperpod_stderr(None)

    def test_whitespace_only(self):
        _check_hyperpod_stderr("   \n\n  ")

    def test_benign_not_openssl_warning_with_prefix(self):
        _check_hyperpod_stderr(
            "/opt/homebrew/lib/python3.12/site-packages/urllib3/__init__.py:35: "
            "NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+"
        )

    def test_benign_insecure_request_with_prefix(self):
        _check_hyperpod_stderr(
            "urllib3/connectionpool.py:1045: InsecureRequestWarning: Unverified HTTPS request"
        )

    def test_benign_deprecation_with_prefix(self):
        _check_hyperpod_stderr(
            "/path/to/kubernetes/client.py:10: DeprecationWarning: some old API is deprecated"
        )

    def test_benign_future_warning_with_prefix(self):
        _check_hyperpod_stderr(
            "/path/to/module.py:42: FutureWarning: this will change in a future version"
        )

    def test_benign_resource_warning_with_prefix(self):
        _check_hyperpod_stderr("/path/to/subprocess.py:99: ResourceWarning: unclosed file")

    def test_benign_user_warning_with_prefix(self):
        _check_hyperpod_stderr("/path/to/lib.py:5: UserWarning: some user warning")

    def test_benign_urllib3_subclass_warning_with_prefix(self):
        _check_hyperpod_stderr("/path/to/urllib3/__init__.py:35: urllib3.NotOpenSSLWarning: test")

    def test_benign_multiline_all_with_prefix(self):
        stderr = (
            "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+\n"
            "/path/to/module.py:10: DeprecationWarning: old API\n"
            "/path/to/other.py:5: FutureWarning: will change\n"
        )
        _check_hyperpod_stderr(stderr)

    def test_benign_bare_not_openssl_warning(self):
        _check_hyperpod_stderr("NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+")

    def test_benign_bare_deprecation_warning(self):
        _check_hyperpod_stderr("DeprecationWarning: some old API is deprecated")

    def test_benign_bare_future_warning(self):
        _check_hyperpod_stderr("FutureWarning: this will change in a future version")

    def test_benign_bare_insecure_request(self):
        _check_hyperpod_stderr("InsecureRequestWarning: Unverified HTTPS request")

    def test_benign_bare_resource_warning(self):
        _check_hyperpod_stderr("ResourceWarning: unclosed file")

    def test_benign_bare_user_warning(self):
        _check_hyperpod_stderr("UserWarning: some user warning")

    def test_real_error_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _check_hyperpod_stderr("Cluster with name not found")
        self.assertIn("Cluster with name not found", str(ctx.exception))

    def test_real_error_connection_failed(self):
        with self.assertRaises(RuntimeError):
            _check_hyperpod_stderr("Connection failed")

    def test_urllib3_error_not_suppressed(self):
        """urllib3 MaxRetryError should NOT be treated as benign."""
        with self.assertRaises(RuntimeError):
            _check_hyperpod_stderr(
                "urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='eks.us-east-1.amazonaws.com'): "
                "Max retries exceeded"
            )

    def test_warning_in_prose_not_suppressed(self):
        """A line mentioning 'DeprecationWarning' in prose should NOT be benign."""
        with self.assertRaises(RuntimeError):
            _check_hyperpod_stderr("Error: DeprecationWarning handling failed in module X")

    def test_two_line_warning_format(self):
        """Python warnings module outputs warning + indented source line."""
        stderr = (
            "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+\n"
            "  import urllib3\n"
        )
        _check_hyperpod_stderr(stderr)

    def test_two_line_warning_mixed_with_error(self):
        """Two-line warning followed by a real error should raise."""
        stderr = (
            "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+\n"
            "  import urllib3\n"
            "Error: cluster not found\n"
        )
        with self.assertRaises(RuntimeError) as ctx:
            _check_hyperpod_stderr(stderr)
        self.assertIn("Error: cluster not found", str(ctx.exception))

    def test_warning_then_blank_then_indented_error_not_suppressed(self):
        """An indented line after a blank should NOT be suppressed."""
        stderr = (
            "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: msg\n"
            "  import urllib3\n"
            "\n"
            "  real indented error\n"
        )
        with self.assertRaises(RuntimeError) as ctx:
            _check_hyperpod_stderr(stderr)
        self.assertIn("real indented error", str(ctx.exception))

    def test_mixed_benign_and_error_raises(self):
        stderr = (
            "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+\n"
            "Error: cluster not found\n"
        )
        with self.assertRaises(RuntimeError) as ctx:
            _check_hyperpod_stderr(stderr)
        self.assertIn("Error: cluster not found", str(ctx.exception))
        # Full stderr is returned for debugging (benign lines included)
        self.assertIn("NotOpenSSLWarning", str(ctx.exception))

    def test_mixed_non_warning_urllib3_line_raises(self):
        """A bare 'urllib3 warning line' without proper format is non-benign."""
        stderr = (
            "/path/to/module.py:10: DeprecationWarning: old API\n"
            "/path/to/other.py:5: FutureWarning: will change\n"
            "fatal: something went wrong\n"
            "urllib3 warning line\n"
        )
        with self.assertRaises(RuntimeError) as ctx:
            _check_hyperpod_stderr(stderr)
        error_msg = str(ctx.exception)
        self.assertIn("fatal: something went wrong", error_msg)
        self.assertIn("urllib3 warning line", error_msg)

    def test_benign_logs_debug(self):
        with self.assertLogs("amzn_nova_forge.util.subprocess_utils", level=logging.DEBUG) as cm:
            _check_hyperpod_stderr(
                "/path/to/urllib3/__init__.py:35: NotOpenSSLWarning: test warning"
            )
        self.assertTrue(any("benign" in msg.lower() for msg in cm.output))

    def test_error_logs_error(self):
        with self.assertLogs("amzn_nova_forge.util.subprocess_utils", level=logging.ERROR) as cm:
            with self.assertRaises(RuntimeError):
                _check_hyperpod_stderr("real error message")
        self.assertTrue(any("real error message" in msg for msg in cm.output))


if __name__ == "__main__":
    unittest.main()
