"""Tests for reward function verifier."""

import tempfile
from pathlib import Path

import pytest

from amzn_nova_forge_sdk.util.reward_verifier import verify_reward_function


def test_verify_with_valid_rft_format():
    """Test verification with valid RFT format data."""
    # Create a simple reward function
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        # Simple reward: length of response
        messages = sample.get("messages", [])
        last_msg = messages[-1] if messages else {}
        response_len = len(last_msg.get("content", ""))
        
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": float(response_len) / 10.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # Valid RFT format
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "reference_answer": "4",
            },
            {
                "id": "sample_2",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "reference_answer": "Hi there!",
            },
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    finally:
        Path(reward_file).unlink()


def test_verify_missing_messages_field():
    """Test that missing messages field raises error."""
    reward_code = """
def lambda_handler(event, context):
    return []
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # Missing messages field
        sample_data = [{"id": "sample_1", "reference_answer": "4"}]

        with pytest.raises(ValueError, match="field 'messages': Field required"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_missing_reference_answer():
    """Test that missing reference_answer field generates a warning (not error)."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # Missing reference_answer field
        sample_data = [
            {"id": "sample_1", "messages": [{"role": "user", "content": "test"}]}
        ]

        # Should not raise error, but should include warning
        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert len(result["warnings"]) == 1
        assert "reference_answer" in result["warnings"][0]
        assert "Without reference_answer" in result["warnings"][0]

    finally:
        Path(reward_file).unlink()


def test_verify_without_id_field():
    """Test that missing id field is allowed (no error)."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # No id field - should be fine
        sample_data = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_with_validation_disabled():
    """Test that validation can be disabled."""
    reward_code = """
def lambda_handler(event, context):
    return [{"id": "test", "aggregate_reward_score": 1.0}]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # Invalid format but validation disabled
        sample_data = [{"prompt": "test", "response": "answer"}]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data, validate_format=False
        )

        # Should not raise error
        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_output_format_validation():
    """Test that output format is validated."""
    reward_code = """
def lambda_handler(event, context):
    # Return wrong format
    return [{"wrong_field": "value"}]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # Should raise ValueError with errors about missing fields
        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        error_msg = str(exc_info.value)
        assert "Missing 'id' field" in error_msg
        assert "Missing 'aggregate_reward_score' field" in error_msg

    finally:
        Path(reward_file).unlink()


def test_verify_with_transformed_dataset_format():
    """Test verification with data that looks like SDK-transformed RFT dataset."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        # Extract prediction from last assistant message
        messages = sample.get("messages", [])
        prediction = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                prediction = msg.get("content", "")
                break
        
        # Compare with reference answer
        ref_answer = sample.get("reference_answer", "")
        reward = 1.0 if prediction.strip() == ref_answer.strip() else -1.0
        
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": reward
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        # Format that matches SDK transformation output
        sample_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "reference_answer": "Paris",
            },
            {
                "id": "custom_id_123",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "reference_answer": "4",
            },
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["total_samples"] == 2
        assert result["successful_samples"] == 2

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_with_metrics():
    """Test verification with metrics_list."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "Reward"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_without_metrics():
    """Test that metrics_list is optional."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_invalid_metrics_list():
    """Test that invalid metrics_list structure raises error."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": "not a list"
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="'metrics_list' should be a list"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_missing_metric_fields():
    """Test that missing metric fields raise errors."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy"}  # Missing value and type
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        error_msg = str(exc_info.value)
        assert "Missing 'value' field" in error_msg
        assert "Missing 'type' field" in error_msg

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_invalid_metric_type():
    """Test that invalid metric type raises error."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "InvalidType"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="'type' should be 'Metric' or 'Reward'"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_invalid_mode():
    """Test that empty results raises error."""
    reward_code = """
def lambda_handler(event, context):
    return []
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # Lambda returns empty array - should fail because we sent 1 sample but got 0 results
        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

        assert "1/1 sample(s) failed validation" in str(exc_info.value)
        assert "0/1 sample(s) passed" in str(exc_info.value)

    finally:
        Path(reward_file).unlink()


def test_verify_training_mode_ignores_metrics_list():
    """Test that metrics_list is validated if present."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # metrics_list should be validated if present
        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_smhp_platform_valid_lambda_arn():
    """Test that SMHP platform accepts Lambda ARN with 'SageMaker' in name."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # Local file should work regardless of platform
        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data, platform=Platform.SMHP
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_smhp_platform_invalid_lambda_arn():
    """Test that SMHP platform rejects Lambda ARN without 'SageMaker' in name."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # Lambda ARN without 'SageMaker' in function name
    invalid_arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-function"

    with pytest.raises(
        ValueError,
        match="Lambda ARN for SMHP.*must contain 'SageMaker'",
    ):
        verify_reward_function(
            reward_function=invalid_arn, sample_data=sample_data, platform=Platform.SMHP
        )


def test_verify_smhp_platform_valid_lambda_arn_case_insensitive():
    """Test that SMHP platform accepts Lambda ARN with 'sagemaker' (lowercase)."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # These should all be valid (case-insensitive)
    valid_arns = [
        "arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward",
        "arn:aws:lambda:us-east-1:123456789012:function:my-sagemaker-reward",
        "arn:aws:lambda:us-east-1:123456789012:function:SageMaker-reward-function",
        "arn:aws:lambda:us-east-1:123456789012:function:reward-Sagemaker",
    ]

    for arn in valid_arns:
        # Should not raise error (we're just validating ARN format, not invoking)
        # Since we can't actually invoke these ARNs, we'll catch the boto3 error
        try:
            verify_reward_function(
                reward_function=arn, sample_data=sample_data, platform=Platform.SMHP
            )
        except Exception as e:
            # We expect boto3 errors since these are fake ARNs
            # But we should NOT get the "must contain 'SageMaker'" error
            assert "must contain 'SageMaker'" not in str(e)
            # And we should NOT get the "platform parameter is required" error
            assert "platform' parameter is required" not in str(e)


def test_verify_smtj_platform_no_lambda_arn_validation():
    """Test that SMTJ platform doesn't validate Lambda ARN format."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # Lambda ARN without 'SageMaker' should be fine for SMTJ
    arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-function"

    # Should not raise the SageMaker validation error
    # (will fail with boto3 error since it's a fake ARN, but that's expected)
    try:
        verify_reward_function(
            reward_function=arn, sample_data=sample_data, platform=Platform.SMTJ
        )
    except Exception as e:
        # Should NOT get the "must contain 'SageMaker'" error
        assert "must contain 'SageMaker'" not in str(e)


def test_verify_lambda_arn_requires_platform():
    """Test that Lambda ARN requires platform parameter."""
    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # Lambda ARN without platform parameter
    arn = "arn:aws:lambda:us-east-1:123456789012:function:my-reward-function"

    with pytest.raises(
        ValueError, match="'platform' parameter is required when using a Lambda ARN"
    ):
        verify_reward_function(reward_function=arn, sample_data=sample_data)


def test_verify_evaluation_mode_with_multiple_metrics():
    """Test evaluation mode with multiple metrics of different types."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.82,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "Reward"},
                {"name": "coherence", "value": 0.78, "type": "Metric"},
                {"name": "relevance", "value": 0.95, "type": "Reward"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["output"]["metrics_list"] is not None
        assert len(result["results"][0]["output"]["metrics_list"]) == 4

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_empty_metrics_list():
    """Test that empty metrics_list is invalid."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": []
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # Empty metrics_list should be valid (it's just an empty array)
        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_metric_with_wrong_value_type():
    """Test that metric value must be a number."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": "0.85", "type": "Metric"}  # String instead of number
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="'value' should be a number"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_metric_with_wrong_name_type():
    """Test that metric name must be a string."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": 123, "value": 0.85, "type": "Metric"}  # Number instead of string
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="'name' should be a string"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_multiple_samples():
    """Test evaluation mode with multiple samples."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for idx, sample in enumerate(event):
        results.append({
            "id": sample.get("id", f"sample_{idx}"),
            "aggregate_reward_score": 0.5 + (idx * 0.1),
            "metrics_list": [
                {"name": "accuracy", "value": 0.6 + (idx * 0.1), "type": "Metric"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test1"}],
                "reference_answer": "answer1",
            },
            {
                "id": "sample_2",
                "messages": [{"role": "user", "content": "test2"}],
                "reference_answer": "answer2",
            },
            {
                "id": "sample_3",
                "messages": [{"role": "user", "content": "test3"}],
                "reference_answer": "answer3",
            },
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["total_samples"] == 3
        assert result["successful_samples"] == 3
        assert len(result["results"]) == 3

    finally:
        Path(reward_file).unlink()


def test_verify_smhp_platform_malformed_lambda_arn():
    """Test that malformed Lambda ARN raises appropriate error."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # Malformed ARN
    malformed_arn = "arn:aws:lambda:us-east-1:123456789012:invalid-format"

    with pytest.raises(ValueError, match="Invalid Lambda ARN format"):
        verify_reward_function(
            reward_function=malformed_arn,
            sample_data=sample_data,
            platform=Platform.SMHP,
        )


def test_verify_smhp_platform_with_local_file():
    """Test that SMHP platform validation doesn't affect local files."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        # Local file should work with SMHP platform (no ARN validation)
        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data, platform=Platform.SMHP
        )

        assert result["success"] is True

    finally:
        Path(reward_file).unlink()


def test_verify_combined_evaluation_mode_and_smhp_platform():
    """Test using both evaluation mode and SMHP platform together."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.85,
            "metrics_list": [
                {"name": "accuracy", "value": 0.90, "type": "Metric"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
            platform=Platform.SMHP,
        )

        assert result["success"] is True
        assert result["results"][0]["output"]["metrics_list"] is not None

    finally:
        Path(reward_file).unlink()


def test_verify_evaluation_mode_mixed_valid_invalid_metrics():
    """Test evaluation mode with mix of valid and invalid metrics."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},  # Valid
                {"name": "fluency", "value": 0.90, "type": "InvalidType"}  # Invalid type
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="'type' should be 'Metric' or 'Reward'"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_verify_aggregate_reward_score_as_int():
    """Test that aggregate_reward_score can be an integer."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 1  # Integer instead of float
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["results"][0]["output"]["aggregate_reward_score"] == 1

    finally:
        Path(reward_file).unlink()


def test_verify_metric_value_as_int():
    """Test that metric value can be an integer."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "count", "value": 5, "type": "Metric"}  # Integer value
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        result = verify_reward_function(
            reward_function=reward_file, sample_data=sample_data
        )

        assert result["success"] is True
        assert result["results"][0]["output"]["metrics_list"][0]["value"] == 5

    finally:
        Path(reward_file).unlink()


def test_verify_smhp_arn_with_version_qualifier():
    """Test SMHP validation with Lambda ARN that includes version/alias."""
    from amzn_nova_forge_sdk.model.model_enums import Platform

    sample_data = [
        {
            "id": "sample_1",
            "messages": [{"role": "user", "content": "test"}],
            "reference_answer": "answer",
        }
    ]

    # ARN with version qualifier - should still validate function name
    arn_with_version = (
        "arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward:1"
    )

    # Should not raise SageMaker validation error (function name has SageMaker)
    try:
        verify_reward_function(
            reward_function=arn_with_version,
            sample_data=sample_data,
            platform=Platform.SMHP,
        )
    except Exception as e:
        # Should NOT get the "must contain 'SageMaker'" error
        assert "must contain 'SageMaker'" not in str(e)
        # And we should NOT get the "platform parameter is required" error
        assert "platform' parameter is required" not in str(e)


def test_verify_evaluation_mode_metric_not_dict():
    """Test that metric must be a dict."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "no_id"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                "not a dict"  # String instead of dict
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [{"role": "user", "content": "test"}],
                "reference_answer": "answer",
            }
        ]

        with pytest.raises(ValueError, match="Expected dict, got str"):
            verify_reward_function(reward_function=reward_file, sample_data=sample_data)

    finally:
        Path(reward_file).unlink()


def test_training_mode_with_metrics_list():
    """Test that metrics_list is accepted and validated."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        # Return metrics_list
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy", "value": 0.85, "type": "Metric"},
                {"name": "fluency", "value": 0.90, "type": "Reward"}
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            }
        ]

        # metrics_list should be accepted
        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
        )

        assert result["success"] is True
        assert result["total_samples"] == 1
        assert result["successful_samples"] == 1

        # Verify metrics_list is in output
        output = result["results"][0]["output"]
        assert "metrics_list" in output
        assert len(output["metrics_list"]) == 2
        assert output["metrics_list"][0]["name"] == "accuracy"
        assert output["metrics_list"][1]["name"] == "fluency"

    finally:
        Path(reward_file).unlink()


def test_training_mode_validates_invalid_metrics_list():
    """Test that metrics_list structure is validated if present."""
    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        # Invalid metrics_list - missing required fields
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 0.75,
            "metrics_list": [
                {"name": "accuracy"}  # Missing 'value' and 'type'
            ]
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            }
        ]

        # metrics_list should be validated if present
        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(
                reward_function=reward_file,
                sample_data=sample_data,
            )

        error_msg = str(exc_info.value)
        assert "Missing 'value' field" in error_msg
        assert "Missing 'type' field" in error_msg

    finally:
        Path(reward_file).unlink()


def test_logging_shows_input_and_output_for_all_samples(caplog):
    """Test that logging shows input and output for all validated samples."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test1"},
                    {"role": "assistant", "content": "response1"},
                ],
                "reference_answer": "answer1",
            },
            {
                "id": "sample_2",
                "messages": [
                    {"role": "user", "content": "test2"},
                    {"role": "assistant", "content": "response2"},
                ],
                "reference_answer": "answer2",
            },
        ]

        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
        )

        assert result["success"] is True

        # Check that logs contain INPUT and OUTPUT for both samples
        log_text = caplog.text
        assert "Sample 0 INPUT:" in log_text
        assert "Sample 0 OUTPUT [PASS]:" in log_text
        assert "Sample 1 INPUT:" in log_text
        assert "Sample 1 OUTPUT [PASS]:" in log_text

        # Check that sample data appears in logs
        assert "sample_1" in log_text
        assert "sample_2" in log_text
        assert "test1" in log_text
        assert "test2" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_shows_validation_errors(caplog):
    """Test that logging shows validation errors for failed samples."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        # Missing required field 'aggregate_reward_score'
        results.append({
            "id": sample.get("id", "unknown"),
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            },
        ]

        with pytest.raises(ValueError):
            verify_reward_function(
                reward_function=reward_file,
                sample_data=sample_data,
            )

        # Check that logs show FAIL status and validation errors
        log_text = caplog.text
        assert "Sample 0 OUTPUT" in log_text
        assert "FAIL" in log_text
        assert "Sample 0 validation errors:" in log_text
        assert "aggregate_reward_score" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_shows_summary(caplog):
    """Test that logging shows summary of validation results."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    results = []
    for sample in event:
        results.append({
            "id": sample.get("id", "unknown"),
            "aggregate_reward_score": 1.0
        })
    return results
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": f"sample_{i}",
                "messages": [
                    {"role": "user", "content": f"test{i}"},
                    {"role": "assistant", "content": f"response{i}"},
                ],
                "reference_answer": f"answer{i}",
            }
            for i in range(5)
        ]

        result = verify_reward_function(
            reward_function=reward_file,
            sample_data=sample_data,
        )

        assert result["success"] is True

        # Check that logs contain summary
        log_text = caplog.text
        assert "Testing local Python file:" in log_text
        assert "Number of samples: 5" in log_text
        assert "Lambda returned list with 5 result(s)" in log_text
        assert "All 5 sample(s) passed validation" in log_text

    finally:
        Path(reward_file).unlink()


def test_logging_lambda_execution_error(caplog):
    """Test that logging shows Lambda execution errors clearly."""
    import logging

    caplog.set_level(logging.INFO)

    reward_code = """
def lambda_handler(event, context):
    # Simulate Lambda error by raising exception
    raise TypeError("RewardOutput.__init__() got an unexpected keyword argument 'aggregate_score'")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(reward_code)
        reward_file = f.name

    try:
        sample_data = [
            {
                "id": "sample_1",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ],
                "reference_answer": "answer",
            },
        ]

        with pytest.raises(ValueError) as exc_info:
            verify_reward_function(
                reward_function=reward_file,
                sample_data=sample_data,
            )

        # Check that error message is clear and contains the handler error
        error_msg = str(exc_info.value)
        assert "Handler execution failed" in error_msg
        assert "aggregate_score" in error_msg

    finally:
        Path(reward_file).unlink()
