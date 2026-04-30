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
"""Tests for TransformContext and S3 key helper in DatasetTransformer."""

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from amzn_nova_forge.core.enums import Model, TrainingMethod
from amzn_nova_forge.dataset.dataset_format_schema import (
    SFT_NOVA_ONE_CONVERSE_2024,
    SFT_NOVA_TWO_CONVERSE_2024,
)
from amzn_nova_forge.dataset.dataset_loader import JSONLDatasetLoader
from amzn_nova_forge.dataset.dataset_transformers import (
    MAX_IMAGE_SIZE,
    DatasetTransformer,
    TransformContext,
)
from amzn_nova_forge.dataset.operations.transform_operation import TransformMethod

# A minimal valid 1x1 PNG
TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ===========================================================================
# DatasetTransformer._build_s3_image_key tests
# ===========================================================================


class TestBuildS3ImageKey:
    """Tests for DatasetTransformer._build_s3_image_key static method."""

    def _make_ctx(self, prefix="prefix/", record_index=0, block_index=0):
        """Helper to create a minimal TransformContext for key generation tests."""
        return TransformContext(
            bucket="bucket",
            prefix=prefix,
            bucket_owner="owner",
            record_index=record_index,
            block_index=block_index,
        )

    @pytest.mark.parametrize(
        "record_index,block_index,fmt,expected",
        [
            (0, 0, "png", "prefix/record_0000_block_000.png"),
            (1, 2, "jpeg", "prefix/record_0001_block_002.jpeg"),
            (42, 1, "png", "prefix/record_0042_block_001.png"),
            (9999, 999, "gif", "prefix/record_9999_block_999.gif"),
        ],
        ids=["zero-zero-png", "small-indices-jpeg", "example-from-spec", "max-padding"],
    )
    def test_key_pattern(self, record_index, block_index, fmt, expected):
        """S3 key follows the pattern {prefix}record_{index:04d}_block_{block_index:03d}.{format}."""
        ctx = self._make_ctx(record_index=record_index, block_index=block_index)
        key = DatasetTransformer._build_s3_image_key(ctx, fmt)
        assert key == expected

    @pytest.mark.parametrize(
        "prefix,expected_prefix",
        [
            ("images/train/", "images/train/"),
            ("data/", "data/"),
            ("", ""),
        ],
        ids=["nested-prefix", "simple-prefix", "empty-prefix"],
    )
    def test_different_prefixes(self, prefix, expected_prefix):
        """Key correctly uses different prefix values."""
        ctx = self._make_ctx(prefix=prefix, record_index=5, block_index=3)
        key = DatasetTransformer._build_s3_image_key(ctx, "png")
        assert key == f"{expected_prefix}record_0005_block_003.png"


# ===========================================================================
# _parse_image_url S3 upload tests
# ===========================================================================


class TestParseImageUrlS3Upload:
    """Tests for _parse_image_url with S3 upload via transform_ctx and s3_client."""

    @staticmethod
    def _make_ctx(
        bucket="test-bucket",
        prefix="images/",
        bucket_owner="123456789012",
        record_index=0,
        block_index=0,
    ):
        """Helper to create a TransformContext."""
        return TransformContext(
            bucket=bucket,
            prefix=prefix,
            bucket_owner=bucket_owner,
            record_index=record_index,
            block_index=block_index,
        )

    def test_data_uri_put_object(self):
        """Data URI: decodes base64, calls put_object, returns s3Location block with bucketOwner."""
        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        data_uri = f"data:image/png;base64,{png_b64}"
        ctx = self._make_ctx()
        mock_s3 = MagicMock()

        result = DatasetTransformer._parse_image_url(data_uri, transform_ctx=ctx, s3_client=mock_s3)

        mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="images/record_0000_block_000.png",
            Body=TINY_PNG_BYTES,
        )
        assert result == {
            "image": {
                "format": "png",
                "source": {
                    "s3Location": {
                        "uri": "s3://test-bucket/images/record_0000_block_000.png",
                        "bucketOwner": "123456789012",
                    }
                },
            }
        }

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_http_fetch_put_object(self, mock_urlopen):
        """HTTP URL: fetches image, calls put_object, returns s3Location block."""
        mock_response = MagicMock()
        mock_response.read.return_value = TINY_PNG_BYTES
        mock_response.headers.get.return_value = "image/png"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        result = DatasetTransformer._parse_image_url(
            "https://example.com/photo.png", transform_ctx=ctx, s3_client=mock_s3
        )

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "images/record_0000_block_000.png"
        assert (
            result["image"]["source"]["s3Location"]["uri"]
            == "s3://test-bucket/images/record_0000_block_000.png"
        )
        assert result["image"]["source"]["s3Location"]["bucketOwner"] == "123456789012"

    def test_s3_copy_object(self):
        """S3 URI: calls copy_object, returns s3Location block with target bucket/key."""
        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        result = DatasetTransformer._parse_image_url(
            "s3://source-bucket/path/photo.png", transform_ctx=ctx, s3_client=mock_s3
        )

        mock_s3.copy_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="images/record_0000_block_000.png",
            CopySource={"Bucket": "source-bucket", "Key": "path/photo.png"},
        )
        assert (
            result["image"]["source"]["s3Location"]["uri"]
            == "s3://test-bucket/images/record_0000_block_000.png"
        )
        assert result["image"]["source"]["s3Location"]["bucketOwner"] == "123456789012"
        assert result["image"]["format"] == "png"

    def test_missing_transform_ctx_raises_valueerror(self):
        """Calling _parse_image_url without transform_ctx raises ValueError about multimodal_data_s3_path."""
        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        data_uri = f"data:image/png;base64,{png_b64}"

        with pytest.raises(ValueError, match="multimodal_data_s3_path is required"):
            DatasetTransformer._parse_image_url(data_uri)

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_http_50mb_size_limit(self, mock_urlopen):
        """HTTP image exceeding 50MB raises ValueError (enforced by _fetch_image_from_url)."""

        mock_response = MagicMock()
        mock_response.read.return_value = b"\x00" * (MAX_IMAGE_SIZE + 1)
        mock_response.headers.get.return_value = "image/png"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        with pytest.raises(ValueError, match="Image exceeds maximum allowed size"):
            DatasetTransformer._parse_image_url(
                "https://example.com/big.png", transform_ctx=ctx, s3_client=mock_s3
            )

    def test_unsupported_url_scheme_raises_valueerror(self):
        """Unsupported URL scheme (file://) raises ValueError listing supported schemes."""
        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        with pytest.raises(ValueError, match="Unsupported URL scheme in image_url"):
            DatasetTransformer._parse_image_url(
                "file:///etc/passwd", transform_ctx=ctx, s3_client=mock_s3
            )

    def test_block_index_increments(self):
        """block_index increments after each image is processed."""
        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        data_uri = f"data:image/png;base64,{png_b64}"
        ctx = self._make_ctx()
        mock_s3 = MagicMock()

        assert ctx.block_index == 0
        DatasetTransformer._parse_image_url(data_uri, transform_ctx=ctx, s3_client=mock_s3)
        assert ctx.block_index == 1
        DatasetTransformer._parse_image_url(data_uri, transform_ctx=ctx, s3_client=mock_s3)
        assert ctx.block_index == 2
        DatasetTransformer._parse_image_url(
            "s3://src/img.png", transform_ctx=ctx, s3_client=mock_s3
        )
        assert ctx.block_index == 3


# ===========================================================================
# End-to-end OpenAI → Converse conversion with S3 upload tests
# ===========================================================================


class TestEndToEndOpenaiToConverseWithS3Upload:
    """End-to-end tests: OpenAI multimodal record → Converse output with s3Location + bucketOwner."""

    @staticmethod
    def _make_ctx(bucket="test-bucket", prefix="images/", bucket_owner="123456789012"):
        return TransformContext(
            bucket=bucket,
            prefix=prefix,
            bucket_owner=bucket_owner,
        )

    def test_nova_one_multimodal_produces_s3location(self):
        """Nova One: multimodal OpenAI record produces Converse output with s3Location + bucketOwner."""

        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        rec = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    ],
                },
                {"role": "assistant", "content": "A tiny pixel."},
            ]
        }

        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        result = DatasetTransformer.convert_openai_to_converse_sft_nova_one(
            rec, column_mappings=None, transform_ctx=ctx, s3_client=mock_s3
        )

        jsonschema.validate(instance=result, schema=SFT_NOVA_ONE_CONVERSE_2024)

        user_content = result["messages"][0]["content"]
        image_block = [b for b in user_content if "image" in b][0]
        assert "s3Location" in image_block["image"]["source"]
        assert image_block["image"]["source"]["s3Location"]["bucketOwner"] == "123456789012"
        assert image_block["image"]["source"]["s3Location"]["uri"].startswith("s3://")

        assert ctx.record_index == 1
        assert ctx.block_index == 0

    def test_nova_two_multimodal_produces_s3location(self):
        """Nova Two: multimodal OpenAI record produces Converse output with s3Location + bucketOwner."""

        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        rec = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    ],
                },
                {"role": "assistant", "content": "A small image."},
            ]
        }

        ctx = self._make_ctx()
        mock_s3 = MagicMock()
        result = DatasetTransformer.convert_openai_to_converse_sft_nova_two(
            rec, column_mappings=None, transform_ctx=ctx, s3_client=mock_s3
        )

        jsonschema.validate(instance=result, schema=SFT_NOVA_TWO_CONVERSE_2024)

        user_content = result["messages"][0]["content"]
        image_block = [b for b in user_content if "image" in b][0]
        assert "s3Location" in image_block["image"]["source"]
        assert image_block["image"]["source"]["s3Location"]["bucketOwner"] == "123456789012"
        assert image_block["image"]["source"]["s3Location"]["uri"].startswith("s3://")

        assert ctx.record_index == 1
        assert ctx.block_index == 0

    def test_nova_one_multiple_records_track_indices(self):
        """Nova One: record_index increments across multiple records, block_index resets per record."""

        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        data_uri = f"data:image/png;base64,{png_b64}"

        rec1 = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
                {"role": "assistant", "content": "Answer 1"},
            ]
        }
        rec2 = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Second"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
                {"role": "assistant", "content": "Answer 2"},
            ]
        }

        ctx = self._make_ctx()
        mock_s3 = MagicMock()

        DatasetTransformer.convert_openai_to_converse_sft_nova_one(
            rec1, column_mappings=None, transform_ctx=ctx, s3_client=mock_s3
        )
        assert ctx.record_index == 1
        assert ctx.block_index == 0

        DatasetTransformer.convert_openai_to_converse_sft_nova_one(
            rec2, column_mappings=None, transform_ctx=ctx, s3_client=mock_s3
        )
        assert ctx.record_index == 2
        assert ctx.block_index == 0

        put_calls = mock_s3.put_object.call_args_list
        assert put_calls[0][1]["Key"] == "images/record_0000_block_000.png"
        assert put_calls[1][1]["Key"] == "images/record_0000_block_001.png"
        assert put_calls[2][1]["Key"] == "images/record_0001_block_000.png"

    def test_text_only_nova_one_validates_against_schema(self):
        """Text-only Nova One output validates against SFT_NOVA_ONE_CONVERSE_2024 schema."""

        rec = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        result = DatasetTransformer.convert_openai_to_converse_sft_nova_one(rec)
        jsonschema.validate(instance=result, schema=SFT_NOVA_ONE_CONVERSE_2024)

    def test_text_only_nova_two_validates_against_schema(self):
        """Text-only Nova Two output validates against SFT_NOVA_TWO_CONVERSE_2024 schema."""

        rec = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        result = DatasetTransformer.convert_openai_to_converse_sft_nova_two(rec)
        jsonschema.validate(instance=result, schema=SFT_NOVA_TWO_CONVERSE_2024)


class TestSaveBucketValidation:
    """Tests for S3 bucket mismatch validation on save()."""

    def test_save_raises_when_buckets_differ(self):
        """save() raises DataPrepError when save bucket differs from image bucket."""

        loader = JSONLDatasetLoader()
        loader._multimodal_image_bucket = "image-bucket"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
            tmp_path = f.name

        try:
            loader.load(tmp_path)
            with pytest.raises(Exception, match="image-bucket"):
                loader.save("s3://different-bucket/output/train.jsonl")
        finally:
            os.unlink(tmp_path)

    def test_save_succeeds_when_buckets_match(self):
        """save() does not raise when save bucket matches image bucket."""

        loader = JSONLDatasetLoader()
        loader._multimodal_image_bucket = "same-bucket"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
            tmp_path = f.name

        try:
            loader.load(tmp_path)
            with patch("amzn_nova_forge.util.dataset_writer.DatasetWriter.save_to_s3"):
                loader.save("s3://same-bucket/output/train.jsonl")
        finally:
            os.unlink(tmp_path)

    def test_save_skips_validation_without_image_bucket(self):
        """save() skips bucket validation when no multimodal transform was done."""

        loader = JSONLDatasetLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
            tmp_path = f.name

        try:
            loader.load(tmp_path)
            with patch("amzn_nova_forge.util.dataset_writer.DatasetWriter.save_to_s3"):
                loader.save("s3://any-bucket/output/train.jsonl")
        finally:
            os.unlink(tmp_path)

    def test_save_skips_validation_for_local_path(self):
        """save() skips bucket validation when saving to a local path."""

        loader = JSONLDatasetLoader()
        loader._multimodal_image_bucket = "image-bucket"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
            tmp_path = f.name

        try:
            loader.load(tmp_path)
            out_path = tmp_path + ".out.jsonl"
            loader.save(out_path)
            os.unlink(out_path)
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# transform() S3 path validation and bucket owner auto-resolution tests
# ===========================================================================


class TestTransformMultimodalS3PathValidation:
    """Tests for multimodal_data_s3_path validation and bucket_owner auto-resolution in transform()."""

    @staticmethod
    def _make_loader_with_openai_data():
        """Create a JSONLDatasetLoader with minimal OpenAI multimodal data."""

        png_b64 = base64.b64encode(TINY_PNG_BYTES).decode("utf-8")
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                        },
                    ],
                },
                {"role": "assistant", "content": "A pixel."},
            ]
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(json.dumps(record) + "\n")
        tmp.close()

        loader = JSONLDatasetLoader()
        loader.load(tmp.name)
        return loader, tmp.name

    def test_invalid_s3_path_raises_valueerror(self):
        """transform() with a local path as multimodal_data_s3_path raises ValueError."""

        loader, tmp_path = self._make_loader_with_openai_data()
        try:
            with pytest.raises(ValueError, match="must be a valid S3 prefix starting with 's3://'"):
                loader.transform(
                    method=TransformMethod.SCHEMA,
                    training_method=TrainingMethod.SFT_LORA,
                    model=Model.NOVA_LITE_2,
                    multimodal_data_s3_path="/tmp/local/images/",
                )
                loader.execute()
        finally:
            os.unlink(tmp_path)

    @patch("amzn_nova_forge.dataset.operations.transform_operation.boto3.client")
    def test_auto_resolve_bucket_owner_via_sts(self, mock_boto_client):
        """transform() auto-resolves bucket_owner via STS when not provided."""

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_s3 = MagicMock()

        def side_effect(service):
            if service == "sts":
                return mock_sts
            elif service == "s3":
                return mock_s3
            return MagicMock()

        mock_boto_client.side_effect = side_effect

        loader, tmp_path = self._make_loader_with_openai_data()
        try:
            loader.transform(
                method=TransformMethod.SCHEMA,
                training_method=TrainingMethod.SFT_LORA,
                model=Model.NOVA_LITE_2,
                multimodal_data_s3_path="s3://my-bucket/images/",
            )
            loader.execute()
            # Verify STS was called
            mock_sts.get_caller_identity.assert_called_once()
            # Verify bucket was set eagerly for save() validation
            assert loader._multimodal_image_bucket == "my-bucket"
        finally:
            os.unlink(tmp_path)

    @patch("amzn_nova_forge.dataset.operations.transform_operation.boto3.client")
    def test_sts_failure_raises_valueerror(self, mock_boto_client):
        """transform() raises ValueError when STS auto-resolution fails."""

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.side_effect = Exception("STS access denied")

        mock_boto_client.return_value = mock_sts

        loader, tmp_path = self._make_loader_with_openai_data()
        try:
            with pytest.raises(ValueError, match="Failed to resolve bucket owner"):
                loader.transform(
                    method=TransformMethod.SCHEMA,
                    training_method=TrainingMethod.SFT_LORA,
                    model=Model.NOVA_LITE_2,
                    multimodal_data_s3_path="s3://my-bucket/images/",
                )
                loader.execute()
        finally:
            os.unlink(tmp_path)
