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
"""Tests for image URL parsing helpers in DatasetTransformer."""

import base64
from http.client import HTTPResponse
from unittest.mock import MagicMock, patch

import pytest

from amzn_nova_forge.dataset.dataset_transformers import (
    DatasetTransformer,
    TransformContext,
)

TEST_IMAGE_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
TEST_IMAGE_B64 = base64.b64encode(TEST_IMAGE_BYTES).decode()


def _make_transform_ctx(bucket="test-bucket", prefix="images/", bucket_owner="123456789012"):
    """Helper to create a TransformContext."""
    return TransformContext(
        bucket=bucket,
        prefix=prefix,
        bucket_owner=bucket_owner,
    )


# ===========================================================================
# _parse_data_uri tests
# ===========================================================================


class TestParseDataUri:
    @pytest.mark.parametrize(
        "image_format",
        ["png", "jpeg", "gif", "webp"],
        ids=["png", "jpeg", "gif", "webp"],
    )
    def test_valid_formats(self, image_format):
        uri = f"data:image/{image_format};base64,{TEST_IMAGE_B64}"
        result_image_format, result_base64 = DatasetTransformer._parse_data_uri(uri)
        assert result_image_format == image_format
        assert result_base64 == TEST_IMAGE_B64

    def test_jpg_format(self):
        uri = f"data:image/jpg;base64,{TEST_IMAGE_B64}"
        result_image_format, result_base64 = DatasetTransformer._parse_data_uri(uri)
        assert result_image_format == "jpg"

    def test_unsupported_format_raises(self):
        uri = f"data:image/bmp;base64,{TEST_IMAGE_B64}"
        with pytest.raises(ValueError, match="Unsupported image format"):
            DatasetTransformer._parse_data_uri(uri)

    @pytest.mark.parametrize(
        "bad_uri",
        [
            "data:text/plain;base64,abc",
            "data:image/png;abc",
            "not-a-data-uri",
            "",
        ],
        ids=["wrong-mime", "missing-base64-marker", "not-data-uri", "empty"],
    )
    def test_malformed_uri_raises(self, bad_uri):
        with pytest.raises(ValueError):
            DatasetTransformer._parse_data_uri(bad_uri)


# ===========================================================================
# _fetch_image_from_url tests
# ===========================================================================


def _mock_response(data, status=200, content_type="image/png"):
    """Create a mock HTTP response."""
    resp = MagicMock(spec=HTTPResponse)
    resp.status = status
    resp.read.return_value = data
    resp.headers = {"Content-Type": content_type}
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestFetchImageFromUrl:
    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_infer_format_from_extension(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES)
        image_format, image_bytes = DatasetTransformer._fetch_image_from_url(
            "https://example.com/photo.png"
        )
        assert image_format == "png"
        assert image_bytes == TEST_IMAGE_BYTES

    @pytest.mark.parametrize(
        "url,expected_image_format",
        [
            ("https://example.com/img.jpeg", "jpeg"),
            ("https://example.com/img.jpg", "jpg"),
            ("https://example.com/img.gif", "gif"),
            ("https://example.com/img.webp", "webp"),
        ],
        ids=["jpeg", "jpg", "gif", "webp"],
    )
    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_various_extensions(self, mock_urlopen, url, expected_image_format):
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES)
        image_format, image_bytes = DatasetTransformer._fetch_image_from_url(url)
        assert image_format == expected_image_format

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_fallback_to_content_type(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES, content_type="image/jpeg")
        image_format, image_bytes = DatasetTransformer._fetch_image_from_url(
            "https://example.com/image_no_ext"
        )
        assert image_format == "jpeg"
        assert image_bytes == TEST_IMAGE_BYTES

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_network_error_on_get_raises(self, mock_urlopen):
        """Network error during GET (after successful HEAD) raises ValueError."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # HEAD succeeds
                return _mock_response(TEST_IMAGE_BYTES)
            # GET fails
            raise Exception("Connection refused")

        mock_urlopen.side_effect = side_effect
        with pytest.raises(ValueError, match="Failed to fetch image"):
            DatasetTransformer._fetch_image_from_url("https://example.com/img.png")

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_no_ext_no_image_content_type_raises(self, mock_urlopen):
        """Case a/b: no extension + non-image Content-Type → FAIL."""
        mock_urlopen.return_value = _mock_response(
            TEST_IMAGE_BYTES, content_type="application/octet-stream"
        )
        with pytest.raises(ValueError, match="Cannot determine image format"):
            DatasetTransformer._fetch_image_from_url("https://example.com/no_extension")

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_ext_with_non_image_header_warns_and_proceeds(self, mock_urlopen):
        """Case e: extension present + non-image Content-Type → WARN + use ext."""
        mock_urlopen.return_value = _mock_response(
            TEST_IMAGE_BYTES, content_type="application/octet-stream"
        )
        image_format, image_bytes = DatasetTransformer._fetch_image_from_url(
            "https://example.com/photo.png"
        )
        assert image_format == "png"
        assert image_bytes == TEST_IMAGE_BYTES

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_ext_header_mismatch_warns_and_uses_ext(self, mock_urlopen):
        """Case g: extension says png, header says jpeg → WARN + use ext."""
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES, content_type="image/jpeg")
        image_format, image_bytes = DatasetTransformer._fetch_image_from_url(
            "https://example.com/photo.png"
        )
        assert image_format == "png"
        assert image_bytes == TEST_IMAGE_BYTES


# ===========================================================================
# _parse_image_url tests
# ===========================================================================


class TestParseImageUrl:
    def test_data_uri_routes_correctly(self):
        uri = f"data:image/png;base64,{TEST_IMAGE_B64}"
        ctx = _make_transform_ctx()
        result = DatasetTransformer._parse_image_url(uri, transform_ctx=ctx, s3_client=MagicMock())
        assert result["image"]["format"] == "png"
        assert "s3Location" in result["image"]["source"]
        assert result["image"]["source"]["s3Location"]["bucketOwner"] == "123456789012"

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_https_routes_correctly(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES)
        ctx = _make_transform_ctx()
        result = DatasetTransformer._parse_image_url(
            "https://example.com/img.png", transform_ctx=ctx, s3_client=MagicMock()
        )
        assert result["image"]["format"] == "png"
        assert "s3Location" in result["image"]["source"]

    @patch("amzn_nova_forge.dataset.dataset_transformers.urllib.request.urlopen")
    def test_http_routes_correctly(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(TEST_IMAGE_BYTES)
        ctx = _make_transform_ctx()
        result = DatasetTransformer._parse_image_url(
            "http://example.com/img.jpeg", transform_ctx=ctx, s3_client=MagicMock()
        )
        assert result["image"]["format"] == "jpeg"
        assert "s3Location" in result["image"]["source"]

    def test_s3_routes_correctly(self):
        ctx = _make_transform_ctx()
        result = DatasetTransformer._parse_image_url(
            "s3://bucket/path/image.gif", transform_ctx=ctx, s3_client=MagicMock()
        )
        assert result["image"]["format"] == "gif"
        assert "s3Location" in result["image"]["source"]

    def test_unsupported_scheme_raises(self):
        ctx = _make_transform_ctx()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            DatasetTransformer._parse_image_url(
                "ftp://example.com/image.png", transform_ctx=ctx, s3_client=MagicMock()
            )

    def test_empty_url_raises(self):
        ctx = _make_transform_ctx()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            DatasetTransformer._parse_image_url("", transform_ctx=ctx, s3_client=MagicMock())

    def test_no_transform_ctx_raises(self):
        """Calling _parse_image_url without transform_ctx raises ValueError."""
        with pytest.raises(ValueError, match="multimodal_data_s3_path is required"):
            DatasetTransformer._parse_image_url(f"data:image/png;base64,{TEST_IMAGE_B64}")


# ===========================================================================
# _convert_content_blocks tests
# ===========================================================================


class TestConvertContentBlocks:
    def test_text_only_array(self):
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        result = DatasetTransformer._convert_content_blocks(blocks)
        assert result == [{"text": "Hello"}, {"text": "World"}]

    def test_empty_text_skipped(self):
        blocks = [
            {"type": "text", "text": "Keep me"},
            {"type": "text", "text": ""},
            {"type": "text"},
        ]
        result = DatasetTransformer._convert_content_blocks(blocks)
        assert result == [{"text": "Keep me"}]

    def test_data_uri_image(self):
        blocks = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
            }
        ]
        ctx = _make_transform_ctx()
        result = DatasetTransformer._convert_content_blocks(
            blocks, transform_ctx=ctx, s3_client=MagicMock()
        )
        assert len(result) == 1
        assert result[0]["image"]["format"] == "png"
        assert "s3Location" in result[0]["image"]["source"]

    def test_s3_uri_image(self):
        blocks = [
            {
                "type": "image_url",
                "image_url": {"url": "s3://my-bucket/images/photo.jpeg"},
            }
        ]
        ctx = _make_transform_ctx()
        result = DatasetTransformer._convert_content_blocks(
            blocks, transform_ctx=ctx, s3_client=MagicMock()
        )
        assert len(result) == 1
        assert result[0]["image"]["format"] == "jpeg"
        assert "s3Location" in result[0]["image"]["source"]

    def test_mixed_text_and_image_preserves_order(self):
        blocks = [
            {"type": "text", "text": "Look at this:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
            },
            {"type": "text", "text": "What do you see?"},
        ]
        ctx = _make_transform_ctx()
        result = DatasetTransformer._convert_content_blocks(
            blocks, transform_ctx=ctx, s3_client=MagicMock()
        )
        assert len(result) == 3
        assert result[0] == {"text": "Look at this:"}
        assert result[1]["image"]["format"] == "png"
        assert result[2] == {"text": "What do you see?"}

    def test_empty_blocks_returns_empty(self):
        result = DatasetTransformer._convert_content_blocks([])
        assert result == []


# ===========================================================================
# _convert_openai_messages_to_converse tests - Test simple text content,
# array-type content including multimodal types
# ===========================================================================


class TestConvertOpenaiMessagesToConverseMultimodal:
    def test_string_content_preserved(self):
        """String content still works as before"""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        system, converse = DatasetTransformer._convert_openai_messages_to_converse(messages)
        assert system == "You are helpful."
        assert converse[0]["role"] == "user"
        assert converse[0]["content"] == [{"text": "Hello"}]
        assert converse[1]["role"] == "assistant"
        assert converse[1]["content"] == [{"text": "Hi there"}]

    def test_array_content_conversion(self):
        """Array-type content in user message produces multimodal Converse content."""
        ctx = _make_transform_ctx()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"},
                    },
                ],
            },
            {"role": "assistant", "content": "It is a tiny pixel."},
        ]
        _, converse = DatasetTransformer._convert_openai_messages_to_converse(
            messages, transform_ctx=ctx, s3_client=MagicMock()
        )
        user_content = converse[0]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"text": "Describe this image:"}
        assert user_content[1]["image"]["format"] == "png"
        assert "s3Location" in user_content[1]["image"]["source"]

    def test_empty_array_content_skipped(self):
        """User message with empty array content should not produce a message."""
        messages = [
            {"role": "user", "content": []},
            {"role": "assistant", "content": "No input received."},
        ]
        _, converse = DatasetTransformer._convert_openai_messages_to_converse(messages)
        # Only the assistant message should be present
        assert len(converse) == 1
        assert converse[0]["role"] == "assistant"

    def test_null_content_no_user_message(self):
        """Null content should not produce a user message (backward compat)."""
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Ok."},
        ]
        _, converse = DatasetTransformer._convert_openai_messages_to_converse(messages)
        assert len(converse) == 1
        assert converse[0]["role"] == "assistant"

    def test_nova_two_multimodal(self):
        """Array-type content works with nova_version 2.0 as well."""
        ctx = _make_transform_ctx()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/gif;base64,{TEST_IMAGE_B64}"},
                    },
                ],
            },
            {"role": "assistant", "content": "An animation."},
        ]
        _, converse = DatasetTransformer._convert_openai_messages_to_converse(
            messages, nova_version="2.0", transform_ctx=ctx, s3_client=MagicMock()
        )
        user_content = converse[0]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"text": "Describe:"}
        assert user_content[1]["image"]["format"] == "gif"
