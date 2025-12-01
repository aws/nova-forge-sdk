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
from typing import Any, Dict, Optional

import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smhp as smhp
import amzn_nova_customization_sdk.recipe_config.v_two.rft_config_smtj as smtj
from amzn_nova_customization_sdk.model.model_enums import (
    Model,
    Platform,
    TrainingMethod,
)
from amzn_nova_customization_sdk.recipe_builder.base_recipe_builder import (
    BaseRecipeBuilder,
)
from amzn_nova_customization_sdk.recipe_config.base_recipe_config import (
    BaseRecipeConfig,
    BaseRunConfig,
)
from amzn_nova_customization_sdk.validation.rft_validator import RFTValidator


class RFTRecipeBuilder(BaseRecipeBuilder):
    def __init__(
        self,
        job_name: str,
        platform: Platform,
        model: Model,
        method: TrainingMethod,
        instance_type: str,
        instance_count: int,
        data_s3_path: Optional[str],
        output_s3_path: str,
        rft_lambda_arn: Optional[str],  # RFTValidator later validates this parameter
        overrides: Dict[str, Any],
        infra: Optional[Any] = None,
        model_path: Optional[str] = None,
    ):
        self.platform = platform
        self.model = model
        self.method = method
        self.rft_lambda_arn = rft_lambda_arn
        super().__init__(
            job_name=job_name,
            platform=platform,
            method=method,
            model_type=model.model_type,
            model_path=model_path or model.model_path,
            instance_type=instance_type,
            instance_count=instance_count,
            data_s3_path=data_s3_path or "",
            output_s3_path=output_s3_path,
            overrides=overrides,
            infra=infra,
        )

    def _validate_user_input(
        self, validation_config: Optional[Dict[str, bool]] = None
    ) -> None:
        RFTValidator.validate(
            platform=self.platform,
            model=self.model,
            method=self.method,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            rft_lambda_arn=self.rft_lambda_arn,
            overrides=self.overrides,
            infra=self.infra,
            validation_config=validation_config,
        )

    def _build_recipe_config_smtj(self, run: BaseRunConfig) -> smtj.RFTRecipeConfig:
        assert self.rft_lambda_arn is not None

        rft_run_config = smtj.RFTRunConfig(
            **run.__dict__,
            **{
                k: v for k, v in self.overrides.items() if hasattr(smtj.RFTRunConfig, k)
            },
            reward_lambda_arn=self.rft_lambda_arn,
        )

        # RFT Lora
        if self.method == TrainingMethod.RFT_LORA:
            lora_tuning = smtj.LoraTuning(
                **{
                    k: v
                    for k, v in self.overrides.items()
                    if hasattr(smtj.LoraTuning, k)
                }
            )
            peft = smtj.Peft(peft_scheme=smtj.PeftScheme.LORA, lora_tuning=lora_tuning)
        # RFT
        else:
            peft = smtj.Peft(peft_scheme=smtj.PeftScheme.NULL)

        model = smtj.Model(peft=peft)

        advantage_strategy = smtj.AdvantageStrategy(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(smtj.AdvantageStrategy, k)
            }
        )

        generator = smtj.Generator(
            **{k: v for k, v in self.overrides.items() if hasattr(smtj.Generator, k)}
        )

        api_endpoint = smtj.ApiEndpoint(
            **{k: v for k, v in self.overrides.items() if hasattr(smtj.ApiEndpoint, k)},
            lambda_arn=self.rft_lambda_arn,
        )

        rewards = smtj.Rewards(api_endpoint=api_endpoint)

        rollout = smtj.Rollout(
            advantage_strategy=advantage_strategy,
            generator=generator,
            rewards=rewards,
        )

        optim = smtj.Optim(
            **{k: v for k, v in self.overrides.items() if hasattr(smtj.Optim, k)}
        )

        trainer = smtj.Trainer(
            **{k: v for k, v in self.overrides.items() if hasattr(smtj.Trainer, k)},
            optim=optim,
        )

        training_config = smtj.RFTTrainingConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(smtj.RFTTrainingConfig, k)
            },
            trainer=trainer,
            model=model,
            rollout=rollout,
        )

        return smtj.RFTRecipeConfig(run=rft_run_config, training_config=training_config)

    def _build_recipe_config_smhp(self, run: BaseRunConfig) -> smhp.RFTRecipeConfig:
        assert self.rft_lambda_arn is not None

        rft_run_config = smhp.RFTRunConfig(
            **run.__dict__,
            **{
                k: v for k, v in self.overrides.items() if hasattr(smhp.RFTRunConfig, k)
            },
            reward_lambda_arn=self.rft_lambda_arn,
        )

        # RFT Lora
        if self.method == TrainingMethod.RFT_LORA:
            lora_tuning = smhp.LoraTuning(
                **{
                    k: v
                    for k, v in self.overrides.items()
                    if hasattr(smhp.LoraTuning, k)
                }
            )
            peft = smhp.Peft(peft_scheme=smhp.PeftScheme.LORA, lora_tuning=lora_tuning)
        # RFT
        else:
            peft = None

        data = smhp.Data(
            **{k: v for k, v in self.overrides.items() if hasattr(smhp.Data, k)}
        )

        rollout_strategy = smhp.RolloutStrategy(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(smhp.RolloutStrategy, k)
            }
        )

        advantage_strategy = smhp.AdvantageStrategy(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(smhp.AdvantageStrategy, k)
            }
        )

        generator = smhp.Generator(
            **{k: v for k, v in self.overrides.items() if hasattr(smhp.Generator, k)}
        )

        api_endpoint = smhp.ApiEndpoint(
            **{k: v for k, v in self.overrides.items() if hasattr(smhp.ApiEndpoint, k)},
            lambda_arn=self.rft_lambda_arn,
        )

        rewards = smhp.Rewards(api_endpoint=api_endpoint)

        rollout = smhp.Rollout(
            rollout_strategy=rollout_strategy,
            advantage_strategy=advantage_strategy,
            generator=generator,
            rewards=rewards,
        )

        optim_config = smhp.OptimConfig(
            **{k: v for k, v in self.overrides.items() if hasattr(smhp.OptimConfig, k)}
        )

        trainer = smhp.Trainer(
            **{k: v for k, v in self.overrides.items() if hasattr(smhp.Trainer, k)},
            optim_config=optim_config,
            peft=peft,
        )

        training_config = smhp.RFTTrainingConfig(
            **{
                k: v
                for k, v in self.overrides.items()
                if hasattr(smhp.RFTTrainingConfig, k)
            },
            data=data,
            rollout=rollout,
            trainer=trainer,
        )

        return smhp.RFTRecipeConfig(run=rft_run_config, training_config=training_config)

    def _build_recipe_config(self) -> BaseRecipeConfig:
        run = self._create_base_run_config()
        for k, v in self.overrides.items():
            if hasattr(run, k):
                setattr(run, k, v)

        if self.platform is Platform.SMTJ:
            return self._build_recipe_config_smtj(run=run)
        if self.platform is Platform.SMHP:
            return self._build_recipe_config_smhp(run=run)

        raise ValueError(
            f"Invalid platform provided for RFT training: '{self.platform.value}'"
        )
