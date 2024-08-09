# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
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

import os
import math

# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    array_id = os.getenv('SLURM_ARRAY_TASK_ID', '')
    assert array_id
    array_id = int(array_id)
    print(f'\nSLURM_ARRAY_TASK_ID = {array_id}\n')

    # set LR based on array_id
    LR_VALS = [7.62939453e-06, 1.52587891e-05, 3.05175781e-05, 6.10351562e-05,
                1.22070312e-04, 2.44140625e-04, 4.88281250e-04, 9.76562500e-04,
                1.95312500e-03] # np.logspace(-17, -9, 9, base=2)
    assert array_id < len(LR_VALS)
    cfg.model.optim.lr = LR_VALS[array_id]

    # set init_method_std based on base model width adn specified scale
    _init_method_scale = 1
    _base_model_width = int(os.getenv('BASE_WIDTH', None))
    assert _base_model_width
    cfg.model.init_method_std = float(_init_method_scale / math.sqrt(_base_model_width))

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
