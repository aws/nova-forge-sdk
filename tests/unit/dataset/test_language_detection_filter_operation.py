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
"""Unit tests for LanguageDetectionFilterOperation."""

import unittest
from unittest.mock import MagicMock, patch

from amzn_nova_forge.dataset.data_state import DataLocation, DataState
from amzn_nova_forge.dataset.operations.filter_operation import (
    FilterMethod,
    get_filter_operation,
)
from amzn_nova_forge.dataset.operations.language_detection_filter_operation import (
    _LANGUAGE_DETECTION_PARAM_KEYS,
    _PIPELINE_ID,
    LanguageDetectionFilterOperation,
)
from amzn_nova_forge.manager.runtime_manager import DataPrepJobConfig


def _s3_state(path="s3://test-bucket/in", fmt="parquet"):
    return DataState(path=path, format=fmt, location=DataLocation.S3)


class TestLanguageDetectionFilterOperationFactory(unittest.TestCase):
    """Tests for FilterMethod.LANGUAGE_DETECTION and the factory."""

    def test_factory_returns_language_detection_instance(self):
        op = get_filter_operation(FilterMethod.LANGUAGE_DETECTION)
        self.assertIsInstance(op, LanguageDetectionFilterOperation)

    def test_enum_value(self):
        self.assertEqual(FilterMethod.LANGUAGE_DETECTION.value, "language_detection")


class TestLanguageDetectionFilterOperationSupportedRuntimes(unittest.TestCase):
    def test_supported_runtimes_includes_smtj_and_glue(self):
        from amzn_nova_forge.manager.glue_runtime_manager import GlueRuntimeManager
        from amzn_nova_forge.manager.runtime_manager import SMTJRuntimeManager

        op = LanguageDetectionFilterOperation()
        runtimes = op.get_supported_runtimes()
        self.assertIn(SMTJRuntimeManager, runtimes)
        self.assertIn(GlueRuntimeManager, runtimes)


class TestLanguageDetectionFilterOperationExecute(unittest.TestCase):
    """Tests for execute() parameter forwarding and config construction."""

    def _make_mock_manager(self):
        manager = MagicMock()
        manager.execute.return_value = "jr_test_run_id"
        return manager

    def _execute_kwargs(self, **extra):
        base = {
            "state": _s3_state(),
            "input_path": "s3://test-bucket/in",
            "output_path": "s3://test-bucket/out",
            "model_path": "s3://test-bucket/models/lid.176.bin",
            "languages": ["en"],
        }
        base.update(extra)
        return base

    def test_missing_model_path_auto_stages(self):
        """When model_path is omitted, SDK auto-stages the default model."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch.object(
                op,
                "_ensure_default_model",
                return_value="s3://auto-bucket/nova-forge/models/language_detection/lid.176.bin",
            ) as mock_ensure,
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
        ):
            op.execute(
                loader=None,
                state=_s3_state(),
                input_path="s3://test-bucket/in",
                output_path="s3://test-bucket/out",
                languages=["en"],
            )

        mock_ensure.assert_called_once_with()
        job_config = manager.execute.call_args[0][0]
        self.assertEqual(
            job_config.extra_args["model_path"],
            "s3://auto-bucket/nova-forge/models/language_detection/lid.176.bin",
        )

    def test_explicit_model_path_skips_auto_staging(self):
        """An explicit model_path must never trigger the auto-stage flow."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch.object(op, "_ensure_default_model") as mock_ensure,
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
        ):
            op.execute(loader=None, **self._execute_kwargs())

        # Critical: the auto-stage helper must not fire when the caller
        # provided their own model. Otherwise we'd make unnecessary S3
        # calls and potentially overwrite a user-staged model.
        mock_ensure.assert_not_called()
        job_config = manager.execute.call_args[0][0]
        self.assertEqual(
            job_config.extra_args["model_path"],
            "s3://test-bucket/models/lid.176.bin",
        )

    def test_missing_languages_raises(self):
        """Filter-mode-only: languages must be provided and non-empty."""
        op = LanguageDetectionFilterOperation()
        with self.assertRaisesRegex(ValueError, "languages"):
            op.execute(
                loader=None,
                state=_s3_state(),
                input_path="s3://test-bucket/in",
                output_path="s3://test-bucket/out",
                model_path="s3://test-bucket/models/lid.176.bin",
            )

    def test_empty_languages_raises(self):
        """An empty languages list should be rejected at queue time."""
        op = LanguageDetectionFilterOperation()
        with self.assertRaisesRegex(ValueError, "languages"):
            op.execute(
                loader=None,
                state=_s3_state(),
                input_path="s3://test-bucket/in",
                output_path="s3://test-bucket/out",
                model_path="s3://test-bucket/models/lid.176.bin",
                languages=[],
            )

    def test_min_score_above_one_raises(self):
        """FastText confidence is in [0, 1]; values > 1 silently drop
        everything at runtime, so we reject them at queue time."""
        op = LanguageDetectionFilterOperation()
        with self.assertRaisesRegex(ValueError, r"min_score.*\[0\.0, 1\.0\]"):
            op.execute(
                loader=None,
                state=_s3_state(),
                input_path="s3://test-bucket/in",
                output_path="s3://test-bucket/out",
                model_path="s3://test-bucket/models/lid.176.bin",
                languages=["en"],
                min_score=1.5,
            )

    def test_min_score_negative_raises(self):
        """Negative min_score is nonsensical — fail fast instead of
        running an expensive Glue job that treats it as disabled."""
        op = LanguageDetectionFilterOperation()
        with self.assertRaisesRegex(ValueError, r"min_score.*\[0\.0, 1\.0\]"):
            op.execute(
                loader=None,
                state=_s3_state(),
                input_path="s3://test-bucket/in",
                output_path="s3://test-bucket/out",
                model_path="s3://test-bucket/models/lid.176.bin",
                languages=["en"],
                min_score=-0.1,
            )

    def test_min_score_boundary_values_accepted(self):
        """0.0 and 1.0 are both valid endpoints of the probability range."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        for value in (0.0, 1.0):
            with (
                patch.object(op, "_resolve_runtime_manager", return_value=manager),
                patch(
                    "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
                ),
            ):
                op.execute(loader=None, **self._execute_kwargs(min_score=value))
            # Not raising is the assertion.

    def test_pipeline_id_cannot_be_overridden(self):
        """Verify pipeline_id is set after extra_args merge (not before)."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
        ):
            op.execute(
                loader=None,
                extra_args={"pipeline_id": "malicious_override"},
                **self._execute_kwargs(),
            )

        job_config = manager.execute.call_args[0][0]
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)

    def test_forwards_stage_params(self):
        """Verify all _LANGUAGE_DETECTION_PARAM_KEYS are forwarded to extra_args."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        kwargs = self._execute_kwargs(
            languages=["en", "fr"],
            min_score=0.5,
            keep_undetected=True,
            lang_field="detected_lang",
            score_field="detected_score",
        )

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
        ):
            op.execute(loader=None, **kwargs)

        job_config = manager.execute.call_args[0][0]
        for key in _LANGUAGE_DETECTION_PARAM_KEYS:
            self.assertEqual(
                job_config.extra_args[key],
                kwargs[key],
                f"Parameter {key!r} not forwarded correctly",
            )

    def test_builds_correct_job_config(self):
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._resolve_s3_directory_to_jsonl",
                side_effect=lambda p: p,
            ),
        ):
            result = op.execute(
                loader=None,
                state=_s3_state("s3://bucket/input/", fmt="jsonl"),
                input_path="s3://bucket/input/",
                output_path="s3://bucket/output/",
                model_path="s3://bucket/models/lid.176.bin",
                languages=["en"],
                input_format="jsonl",
                output_format="jsonl",
                text_field="body",
            )

        job_config = manager.execute.call_args[0][0]
        self.assertIsInstance(job_config, DataPrepJobConfig)
        self.assertEqual(job_config.extra_args["pipeline_id"], _PIPELINE_ID)
        self.assertEqual(job_config.data_s3_path, "s3://bucket/input/")
        self.assertEqual(job_config.output_s3_path, "s3://bucket/output/")
        self.assertEqual(job_config.text_field, "body")
        self.assertEqual(job_config.input_format, "jsonl")
        self.assertEqual(job_config.output_format, "jsonl")
        # fasttext-wheel MUST be in extra_pip_packages so the dependency is
        # installed on Glue Ray workers at startup. Without this, language
        # detection silently fails at runtime.
        self.assertEqual(job_config.extra_pip_packages, ["fasttext-wheel"])
        self.assertEqual(result.status, "SUCCEEDED")

        # output_state drives downstream pipeline chaining — assert it
        # reflects the requested output path, format, and (S3) location.
        self.assertEqual(result.output_state.path, "s3://bucket/output/")
        self.assertEqual(result.output_state.format, "jsonl")
        self.assertEqual(result.output_state.location, DataLocation.S3)

    def test_output_state_uses_local_location_for_local_path(self):
        """A local output_path should produce output_state.location == LOCAL
        so downstream operations don't re-read from S3."""
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._resolve_s3_directory_to_jsonl",
                side_effect=lambda p: p,
            ),
        ):
            result = op.execute(
                loader=None,
                state=_s3_state("s3://bucket/input/", fmt="parquet"),
                input_path="s3://bucket/input/",
                output_path="/tmp/output/",
                model_path="s3://bucket/models/lid.176.bin",
                languages=["en"],
            )

        self.assertEqual(result.output_state.path, "/tmp/output/")
        self.assertEqual(result.output_state.location, DataLocation.LOCAL)

    def test_reloads_loader_when_provided(self):
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()
        mock_loader = MagicMock()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ) as mock_reload,
        ):
            op.execute(loader=mock_loader, **self._execute_kwargs())

        mock_reload.assert_called_once_with(mock_loader, "s3://test-bucket/out", "parquet")

    def test_skips_reload_when_loader_is_none(self):
        op = LanguageDetectionFilterOperation()
        manager = self._make_mock_manager()

        with (
            patch.object(op, "_resolve_runtime_manager", return_value=manager),
            patch(
                "amzn_nova_forge.dataset.operations.language_detection_filter_operation._reload_output_into_loader"
            ) as mock_reload,
        ):
            op.execute(loader=None, **self._execute_kwargs())

        mock_reload.assert_not_called()


class TestEnsureDefaultModel(unittest.TestCase):
    """Tests for _ensure_default_model (the auto-stage helper)."""

    _MOD = "amzn_nova_forge.dataset.operations.language_detection_filter_operation"

    def test_cache_hit_skips_download(self):
        """HeadObject 200 => no download, no upload, return cached URI."""
        mock_s3 = MagicMock()
        # head_object succeeds (no exception) == cache hit.

        with (
            patch(f"{self._MOD}.get_dataprep_bucket_name", return_value="dp-bucket"),
            patch(f"{self._MOD}.ensure_bucket_exists"),
            patch(f"{self._MOD}.boto3.client", return_value=mock_s3),
            patch(f"{self._MOD}.requests.get") as mock_download,
        ):
            uri = LanguageDetectionFilterOperation._ensure_default_model()

        self.assertEqual(uri, "s3://dp-bucket/nova-forge/models/language_detection/lid.176.bin")
        mock_s3.head_object.assert_called_once()
        mock_download.assert_not_called()
        mock_s3.upload_file.assert_not_called()

    def test_cache_miss_downloads_and_uploads(self):
        """HeadObject 404 => download from upstream, verify SHA, upload to S3, return URI."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        # Mock the requests.get context manager: returns a response whose
        # iter_content yields a single chunk of bytes.
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"fake model bytes"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        # SHA-256 of b"fake model bytes" — patched in so the integrity
        # check passes with our synthetic download payload.
        fake_sha = "355ac2cb838a71f81eda48f4fad7903af0c5e4276a86b8fd3dd845d173f58372"

        with (
            patch(f"{self._MOD}.get_dataprep_bucket_name", return_value="dp-bucket"),
            patch(f"{self._MOD}.ensure_bucket_exists"),
            patch(f"{self._MOD}.boto3.client", return_value=mock_s3),
            patch(f"{self._MOD}.requests.get", return_value=mock_resp) as mock_get,
            patch(f"{self._MOD}._DEFAULT_MODEL_SHA256", fake_sha),
            patch(f"{self._MOD}.os.remove"),
        ):
            uri = LanguageDetectionFilterOperation._ensure_default_model()

        self.assertEqual(uri, "s3://dp-bucket/nova-forge/models/language_detection/lid.176.bin")
        mock_get.assert_called_once()
        # First positional arg is the upstream URL. stream=True is critical
        # to avoid buffering 126 MB into memory.
        self.assertEqual(
            mock_get.call_args[0][0],
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        )
        self.assertEqual(mock_get.call_args[1].get("stream"), True)
        mock_resp.raise_for_status.assert_called_once()
        mock_s3.upload_file.assert_called_once()
        upload_args = mock_s3.upload_file.call_args[0]
        # upload_file(local_path, bucket, key)
        self.assertEqual(upload_args[1], "dp-bucket")
        self.assertEqual(upload_args[2], "nova-forge/models/language_detection/lid.176.bin")

    def test_cache_miss_sha_mismatch_aborts(self):
        """SHA mismatch => RuntimeError, nothing uploaded to S3.

        Guards against CDN compromise, MITM, and truncated downloads.
        """
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        # Fake download payload whose SHA does not match the pinned constant.
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"tampered model bytes"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(f"{self._MOD}.get_dataprep_bucket_name", return_value="dp-bucket"),
            patch(f"{self._MOD}.ensure_bucket_exists"),
            patch(f"{self._MOD}.boto3.client", return_value=mock_s3),
            patch(f"{self._MOD}.requests.get", return_value=mock_resp),
            patch(f"{self._MOD}.os.remove"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Integrity check failed"):
                LanguageDetectionFilterOperation._ensure_default_model()

        # Critical: nothing gets uploaded when the hash doesn't match.
        # Otherwise a tampered binary could be staged and loaded by FastText
        # on Glue workers.
        mock_s3.upload_file.assert_not_called()

    def test_unexpected_head_error_propagates(self):
        """AccessDenied, ThrottlingException, etc. must not be silently swallowed.

        Only 404/NoSuchKey/NotFound triggers the download path — other
        errors surface to the caller so permission / network issues are
        obvious in the terminal instead of turning into spurious downloads.
        """
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}},
            "HeadObject",
        )

        with (
            patch(f"{self._MOD}.get_dataprep_bucket_name", return_value="dp-bucket"),
            patch(f"{self._MOD}.ensure_bucket_exists"),
            patch(f"{self._MOD}.boto3.client", return_value=mock_s3),
            patch(f"{self._MOD}.requests.get") as mock_download,
        ):
            with self.assertRaises(ClientError):
                LanguageDetectionFilterOperation._ensure_default_model()

        mock_download.assert_not_called()
        mock_s3.upload_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
