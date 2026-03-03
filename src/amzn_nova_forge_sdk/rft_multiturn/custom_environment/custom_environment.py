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
"""
Utility to create custom RFT environments with boilerplate code.
"""

import os
import tarfile
from dataclasses import dataclass, field
from typing import Literal, Optional

import boto3

from amzn_nova_forge_sdk.util.logging import logger

from .custom_env_templates import (
    MULTI_TURN_TEMPLATE,
    PYPROJECT_TEMPLATE,
    README_TEMPLATE,
    SINGLE_TURN_TEMPLATE,
)


@dataclass
class CustomEnvironment:
    """Custom environment configuration"""

    env_id: str
    local_path: Optional[str] = None
    s3_uri: Optional[str] = None
    output_dir: Optional[str] = None
    env_type: Literal["single_turn", "multi_turn"] = "single_turn"

    def create(self, overwrite: bool = False) -> "CustomEnvironment":
        """
        Create custom environment files with boilerplate code.

        Args:
            overwrite: Whether to overwrite existing directory

        Returns:
            self for chaining

        Raises:
            ValueError: If output_dir is not set
        """
        if not self.output_dir:
            raise ValueError(
                "output_dir is required when creating a new custom environment. "
                "Example: CustomEnvironment(env_id='my_env', output_dir='~/custom_envs').create()"
            )

        if self.env_type not in ["single_turn", "multi_turn"]:
            raise ValueError(
                f"env_type must be 'single_turn' or 'multi_turn', got '{self.env_type}'"
            )

        env_name_safe = self.env_id
        module_name = env_name_safe.replace("-", "_")
        env_file = f"{module_name}.py"

        output_dir = os.path.expanduser(self.output_dir)
        env_dir = os.path.join(output_dir, env_name_safe)

        if os.path.exists(env_dir) and not overwrite:
            raise FileExistsError(
                f"Environment directory already exists: {env_dir}\nUse overwrite=True to replace it"
            )

        os.makedirs(env_dir, exist_ok=True)
        logger.info(f"Creating custom environment at: {env_dir}")

        # Write environment file directly at root (like built-in environments)
        template = (
            SINGLE_TURN_TEMPLATE
            if self.env_type == "single_turn"
            else MULTI_TURN_TEMPLATE
        )
        with open(os.path.join(env_dir, env_file), "w") as f:
            f.write(template)
        logger.info(f"Created {env_file}")
        logger.warning(
            f"IMPORTANT: Please customize the generated {env_file} file with your own environment logic:"
        )
        logger.warning("   - Replace dataset loading with your own data source")
        logger.warning("   - Implement your custom reward function")
        logger.warning("   - Configure parser and rubric for your specific task")
        logger.warning(f"   See TODO comments in {env_file}")

        # Write pyproject.toml
        with open(os.path.join(env_dir, "pyproject.toml"), "w") as f:
            f.write(
                PYPROJECT_TEMPLATE.format(
                    env_name=env_name_safe, module_name=module_name
                )
            )
        logger.info("Created pyproject.toml")

        # Write README
        with open(os.path.join(env_dir, "README.md"), "w") as f:
            f.write(README_TEMPLATE.format(env_name=env_name_safe, env_path=env_dir))
        logger.info("Created README.md")

        # Set local_path
        self.local_path = env_dir

        logger.info(f"\nCustom environment '{env_name_safe}' created successfully!")

        return self

    def load(self) -> "CustomEnvironment":
        """
        Load existing custom environment from local_path.

        If local_path points to a parent directory (no pyproject.toml),
        will look for a subdirectory matching env_id.

        Returns:
            self for chaining

        Raises:
            ValueError: If local_path not set or doesn't exist
        """
        if not self.local_path:
            raise ValueError("local_path required to load environment")

        # Use the path exactly as provided by the user
        env_path = os.path.expanduser(self.local_path)

        if not os.path.exists(env_path):
            raise ValueError(
                f"Environment path does not exist: {env_path}\n"
                f"Provide the correct path to the environment directory"
            )

        if not os.path.isdir(env_path):
            raise ValueError(f"Path is not a directory: {env_path}")

        # Check if this is an environment directory (has pyproject.toml)
        # or a parent directory containing environments
        if not os.path.exists(os.path.join(env_path, "pyproject.toml")):
            # This is a parent directory, look for env_id subdirectory
            env_subdir = os.path.join(env_path, self.env_id)
            if os.path.exists(env_subdir) and os.path.isdir(env_subdir):
                env_path = env_subdir
            else:
                raise ValueError(
                    f"Not an environment directory (no pyproject.toml): {env_path}\n"
                    f"Also tried: {env_subdir}\n"
                    f"Provide the full path to the environment directory"
                )

        # Update local_path to the actual environment directory
        self.local_path = env_path

        logger.info(f"Loaded custom environment from: {env_path}")
        return self

    def validate(self) -> bool:
        """Validate custom environment has required structure"""
        if not self.local_path:
            raise ValueError("local_path required for validation")

        env_path = os.path.expanduser(self.local_path)

        if not os.path.exists(env_path):
            raise ValueError(f"Environment path does not exist: {env_path}")

        if not os.path.isdir(env_path):
            raise ValueError(f"Environment path is not a directory: {env_path}")

        # Check for pyproject.toml
        if not os.path.exists(os.path.join(env_path, "pyproject.toml")):
            raise ValueError(f"Missing pyproject.toml in {env_path}")

        # Check for load_environment function in Python files
        # First check root level (new structure), then src/ directory (old structure)
        found_load_env = False

        # Check new structure: Python files at root level
        py_files_root = [
            f
            for f in os.listdir(env_path)
            if f.endswith(".py") and not f.startswith("__")
        ]

        for py_file in py_files_root:
            with open(os.path.join(env_path, py_file), "r") as f:
                if "def load_environment" in f.read():
                    found_load_env = True
                    break

        # Get the base folder name for later use
        base_name = os.path.basename(env_path.rstrip("/"))
        sanitized_name = base_name.replace("-", "_")

        # If not found, check alternative structures
        if not found_load_env:
            # Check src/ directory (old structure)
            src_path = os.path.join(env_path, "src")
            if os.path.exists(src_path) and os.path.isdir(src_path):
                py_files_src = [
                    f
                    for f in os.listdir(src_path)
                    if f.endswith(".py") and not f.startswith("__")
                ]

                for py_file in py_files_src:
                    with open(os.path.join(src_path, py_file), "r") as f:
                        if "def load_environment" in f.read():
                            found_load_env = True
                            break

        # If still not found, check subfolder with sanitized name (replace - with _)
        if not found_load_env:
            subfolder_path = os.path.join(env_path, sanitized_name)

            if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                py_files_subfolder = [
                    f
                    for f in os.listdir(subfolder_path)
                    if f.endswith(".py") and not f.startswith("__")
                ]

                for py_file in py_files_subfolder:
                    with open(os.path.join(subfolder_path, py_file), "r") as f:
                        if "def load_environment" in f.read():
                            found_load_env = True
                            break

        if not found_load_env:
            raise ValueError(
                f"No load_environment() function found in {env_path}\n"
                f"Checked root level, src/ directory, and {sanitized_name}/ subfolder"
            )

        logger.info(f"Environment validation passed: {env_path}")
        return True

    def package_and_upload(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "custom-envs",
        region: str = "us-east-1",
    ) -> str:
        """
        Package custom environment and upload to S3 for ECS/EC2 deployment.

        Args:
            s3_bucket: S3 bucket name (optional, defaults to SageMaker default bucket)
            s3_prefix: S3 prefix/folder (default: "custom-envs")
            region: AWS region

        Returns:
            str: S3 URI of uploaded package
        """
        if not self.local_path:
            raise ValueError("local_path required for packaging")

        env_path = os.path.expanduser(self.local_path)
        self.validate()

        # Use SageMaker default bucket if not specified
        if not s3_bucket:
            import sagemaker

            session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
            bucket_name = session.default_bucket()
            logger.info(f"Using SageMaker default bucket: {bucket_name}")
        else:
            if s3_bucket.startswith("s3://"):
                raise ValueError(
                    "s3_bucket should be bucket name only, not s3:// URI. Use s3_prefix for path."
                )
            bucket_name = s3_bucket

        tar_name = f"{self.env_id}.tar.gz"
        tar_path = os.path.join("/tmp", tar_name)

        logger.info(f"Packaging {env_path} to {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(env_path, arcname=self.env_id)

        s3_key = f"{s3_prefix.rstrip('/')}/{tar_name}" if s3_prefix else tar_name
        s3_uri = f"s3://{bucket_name}/{s3_key}"

        logger.info(f"Uploading to {s3_uri}")
        s3_client = boto3.client("s3", region_name=region)
        s3_client.upload_file(tar_path, bucket_name, s3_key)

        os.remove(tar_path)
        self.s3_uri = s3_uri

        logger.info(f"Custom environment uploaded: {s3_uri}")
        return s3_uri
