# Copyright 2025 Amazon Inc

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
import json
from typing import Any, Dict

import boto3


def get_hub_content(
    hub_name: str,
    hub_content_name: str,
    hub_content_type: str,
    region: str,
) -> Dict[str, Any]:
    """
     Get hub content from SageMaker via the DescribeHubContent API

    Args:
        hub_name: Name of the SageMaker Hub
        hub_content_name: Name of the hub content
        hub_content_type: Type of hub content
        region: AWS region

    Returns:
        Dict containing hub content
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker_client.describe_hub_content(
            HubName=hub_name,
            HubContentType=hub_content_type,
            HubContentName=hub_content_name,
        )

        # Parse HubContentDocument if it's a JSON string
        if "HubContentDocument" in response:
            hub_content_document = response["HubContentDocument"]
            if isinstance(hub_content_document, str):
                try:
                    response["HubContentDocument"] = json.loads(hub_content_document)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, leave the string as is
                    pass

    except Exception as e:
        raise RuntimeError(
            f"Failed to get SageMaker hub content for '{hub_content_name}': {str(e)}"
        )

    return response
