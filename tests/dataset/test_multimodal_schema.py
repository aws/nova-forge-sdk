"""Tests for multimodal content validation in OPENAI_FORMAT and RFT_OPENAI_FORMAT schemas."""

import jsonschema
import pytest

from amzn_nova_forge.dataset.dataset_format_schema import (
    OPENAI_FORMAT,
    RFT_OPENAI_FORMAT,
    SFT_NOVA_ONE_CONVERSE_2024,
    SFT_NOVA_TWO_CONVERSE_2024,
)


def _openai_record(content):
    """Build a minimal OpenAI-format record with the given user-message content."""
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "ok"},
        ]
    }


def _rft_record(content):
    """Build a minimal RFT OpenAI-format record with the given user-message content."""
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "ok"},
        ]
    }


# ---------------------------------------------------------------------------
# Valid content values (string, null, array)
# ---------------------------------------------------------------------------

VALID_CONTENT_CASES = [
    pytest.param("Hello, world!", id="string-content"),
    pytest.param(None, id="null-content"),
    pytest.param(
        [{"type": "text", "text": "Describe this image."}],
        id="array-single-text-block",
    ),
    pytest.param(
        [
            {"type": "text", "text": "What is in this picture?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBOR..."},
            },
        ],
        id="array-text-and-image-url",
    ),
    pytest.param(
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/photo.jpg"},
            },
        ],
        id="array-single-image-url",
    ),
    pytest.param(
        [
            {"type": "text", "text": "First"},
            {
                "type": "image_url",
                "image_url": {"url": "s3://bucket/path/image.png"},
            },
            {"type": "text", "text": "Second"},
        ],
        id="array-mixed-text-image-text",
    ),
]


# ---------------------------------------------------------------------------
# Invalid / malformed array content
# ---------------------------------------------------------------------------

INVALID_CONTENT_CASES = [
    pytest.param(
        [{"type": "text"}],  # missing required "text" field
        id="array-text-block-missing-text-field",
    ),
    pytest.param(
        [{"type": "image_url"}],  # missing required "image_url" field
        id="array-image-block-missing-image-url-field",
    ),
    pytest.param(
        [{"type": "image_url", "image_url": {}}],  # missing "url" in image_url
        id="array-image-url-missing-url",
    ),
    pytest.param(
        [{"type": "unknown", "data": "foo"}],  # unrecognised block type
        id="array-unknown-block-type",
    ),
    pytest.param(
        [{"text": "no type field"}],  # missing "type" key
        id="array-block-missing-type-key",
    ),
    pytest.param(
        [
            {"type": "text", "text": "ok", "extra": True},  # extra property
        ],
        id="array-text-block-extra-property",
    ),
]


# ---------------------------------------------------------------------------
# OPENAI_FORMAT tests
# ---------------------------------------------------------------------------


class TestOpenAIFormatMultimodalSchema:
    """Validate that OPENAI_FORMAT accepts multimodal content correctly."""

    @pytest.mark.parametrize("content", VALID_CONTENT_CASES)
    def test_valid_content_accepted(self, content):
        record = _openai_record(content)
        jsonschema.validate(instance=record, schema=OPENAI_FORMAT)

    @pytest.mark.parametrize("content", INVALID_CONTENT_CASES)
    def test_malformed_array_rejected(self, content):
        record = _openai_record(content)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=OPENAI_FORMAT)


# ---------------------------------------------------------------------------
# RFT_OPENAI_FORMAT tests
# ---------------------------------------------------------------------------


class TestRFTOpenAIFormatMultimodalSchema:
    """Validate that RFT_OPENAI_FORMAT accepts multimodal content correctly."""

    @pytest.mark.parametrize("content", VALID_CONTENT_CASES)
    def test_valid_content_accepted(self, content):
        record = _rft_record(content)
        jsonschema.validate(instance=record, schema=RFT_OPENAI_FORMAT)

    @pytest.mark.parametrize("content", INVALID_CONTENT_CASES)
    def test_malformed_array_rejected(self, content):
        record = _rft_record(content)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=RFT_OPENAI_FORMAT)


# ---------------------------------------------------------------------------
# Converse schema image source tests
# ---------------------------------------------------------------------------


def _converse_record_with_image(image_block):
    """Build a minimal Converse-format record with the given image content block."""
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "messages": [
            {
                "role": "user",
                "content": [{"image": image_block}],
            },
            {
                "role": "assistant",
                "content": [{"text": "I see the image."}],
            },
        ],
    }


# Valid image blocks for Converse schemas
VALID_CONVERSE_IMAGE_CASES = [
    pytest.param(
        {
            "format": "png",
            "source": {
                "s3Location": {
                    "uri": "s3://bucket/path/image.png",
                    "bucketOwner": "123456789012",
                },
            },
        },
        id="s3-source-with-bucketOwner",
    ),
]

# Invalid image blocks for Converse schemas
INVALID_CONVERSE_IMAGE_CASES = [
    pytest.param(
        {
            "format": "png",
            "source": {},
        },
        id="empty-source",
    ),
    pytest.param(
        {
            "format": "png",
        },
        id="missing-source",
    ),
    pytest.param(
        {
            "format": "png",
            "source": {
                "s3Location": {"bucketOwner": "123456789012"},
            },
        },
        id="s3-source-missing-uri",
    ),
    pytest.param(
        {
            "format": "png",
            "source": {
                "s3Location": {"uri": "s3://bucket/path/image.png"},
            },
        },
        id="s3-source-missing-bucketOwner",
    ),
]

CONVERSE_SCHEMAS = [
    pytest.param(SFT_NOVA_ONE_CONVERSE_2024, id="nova-one"),
    pytest.param(SFT_NOVA_TWO_CONVERSE_2024, id="nova-two"),
]


class TestConverseSchemaImageSources:
    """Validate Converse schemas require s3Location with bucketOwner and reject bytes source."""

    @pytest.mark.parametrize("schema", CONVERSE_SCHEMAS)
    @pytest.mark.parametrize("image_block", VALID_CONVERSE_IMAGE_CASES)
    def test_valid_image_block_accepted(self, schema, image_block):
        record = _converse_record_with_image(image_block)
        jsonschema.validate(instance=record, schema=schema)

    @pytest.mark.parametrize("schema", CONVERSE_SCHEMAS)
    @pytest.mark.parametrize("image_block", INVALID_CONVERSE_IMAGE_CASES)
    def test_invalid_image_block_rejected(self, schema, image_block):
        record = _converse_record_with_image(image_block)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=record, schema=schema)
