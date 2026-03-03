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
"""Templates for custom environment scaffolding"""

SINGLE_TURN_TEMPLATE = """from datasets import load_dataset, Dataset
import verifiers as vf

def load_environment(**kwargs) -> vf.Environment:
    # TODO: Replace with your own dataset loading logic
    # Load your dataset from HuggingFace, S3, or local files
    dataset = Dataset.from_list([
        {"question": "Example input", "answer": "Example answer"},
    ])
    
    # TODO: Replace with your own parser configuration
    # Define how to parse model responses (e.g., XML, JSON, plain text)
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"Respond in format: {parser.get_format_str()}"
    
    # TODO: Replace with your own reward function logic
    # Define how to score model responses based on your task requirements
    def custom_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip() == answer.strip() else 0.0
    
    # TODO: Adjust rubric weights and functions for your use case
    rubric = vf.Rubric(
        parser=parser,
        funcs=[custom_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )
    
    #TODO: Replace vf.SingleTurnEnv with custom env if required
    return vf.SingleTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,
    )
"""

MULTI_TURN_TEMPLATE = """from datasets import load_dataset, Dataset
import verifiers as vf

def load_environment(**kwargs) -> vf.Environment:
    # TODO: Replace with your own dataset loading logic
    # Load your dataset from HuggingFace, S3, or local files
    dataset = Dataset.from_list([
        {"question": "Initial prompt", "answer": "Expected final state"},
    ])
    
    # TODO: Replace with your own parser configuration
    # Define how to parse model responses (e.g., XML, JSON, plain text)
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"Respond in format: {parser.get_format_str()}"
    
    # TODO: Replace with your own reward function logic
    # Define how to score model responses based on your task requirements
    def custom_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip() == answer.strip() else 0.0
    
    # TODO: Adjust rubric weights and functions for your use case
    rubric = vf.Rubric(
        parser=parser,
        funcs=[custom_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )
    
    # TODO: Adjust max_turns based on your multi-turn task requirements
    #TODO: Replace vf.MultiTurnEnv with custom env if required
    return vf.MultiTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,
        max_turns=5,
    )
"""

PYPROJECT_TEMPLATE = """[project]
name = "{env_name}"
description = "Custom RL Environment"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "datasets>=2.0.0",
    "amzn-agi-verifiers>=1.0.3"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{module_name}.py"]
"""

README_TEMPLATE = """# {env_name}

Custom RFT environment for Amazon Nova.

## Installation
```bash
pip install -e .
```

## Usage
```python
from amzn_nova_forge_sdk import RFTMultiturnInfrastructure, CustomEnvironment

custom_env = CustomEnvironment(env_id="{env_name}", local_path="{env_path}")
rft_infra = RFTMultiturnInfrastructure(stack_name="my-stack", custom_env=custom_env)
```
"""
