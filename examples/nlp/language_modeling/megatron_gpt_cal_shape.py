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


# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.mup.shape import append_base_head_widths, make_base_shapes
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    assert cfg.model.get('make_mup', False), \
        'please configure `model.make_mup` to be `True` to calculate base shapes and make use of μP.'
    assert cfg.model.get('shape_file', None), (
        'please configure `model.shape_file` to point to a path in order to '
        'save and later load the file containing base shapes.'
    )
    scalable_widths = cfg.model.get('mup_scalable_widths', [])
    assert scalable_widths, (
        'no `model.mup_scalable_widths` specified; need to specify config values to vary to create base shapes.'
    )
    shrink_factor = cfg.model.get('mup_delta_shrink_factor', 1)
    assert (
        # Do all scalable widths have a specified value?
        all(isinstance(elem, (list, tuple)) and len(elem) == 2 for elem in scalable_widths)
        or shrink_factor != 1
    ), (
        '`model.mup_delta_shrink_factor` must be ≠1 if any scalable width does not have a specified value.'
    )

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    with open_dict(cfg):
        cfg.base_model = cfg.model.copy()
        del cfg.base_model.shape_file
        cfg.delta_model = cfg.model.copy()
        del cfg.delta_model.shape_file
    # Just to make sure the configs were actually deep-copied.
    assert cfg.model.get('shape_file', None), \
        'configs were not deep-copied; the OmegaConf copying code needs an update.'

    # Vary delta model config
    for elem in scalable_widths:
        need_delta_value = True
        # Get config key to set in `delta_model` config and optionally a specified value.
        if isinstance(elem, str):
            cfg_key = elem
        else:
            assert isinstance(elem, (list, tuple)) and 1 <= len(elem) <= 2
            cfg_key = elem[0]
            if len(elem) > 1:
                delta_value = elem[1]
                need_delta_value = False

        base_value = OmegaConf.select(cfg.delta_model, cfg_key)

        # If we don't have a specified delta value, calculate it automatically.
        if need_delta_value:
            delta_value = base_value // shrink_factor
            assert delta_value > 0, 'value became ≤0 after shrinking'
        assert isinstance(base_value, int) and isinstance(delta_value, int), \
            'scalable width value needs to be an integer'
        assert delta_value != base_value, 'scalable width delta value needs to be different from base value'

        OmegaConf.update(cfg.delta_model, cfg_key, delta_value)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.base_model.precision = cfg.trainer.precision
        cfg.delta_model.precision = cfg.trainer.precision

    base_model = MegatronGPTModel(cfg.base_model, trainer)
    delta_model = MegatronGPTModel(cfg.delta_model, trainer)
    make_base_shapes(base_model, delta_model, savefile=cfg.model.shape_file)

    append_base_head_widths(
        cfg.model.shape_file,
        base_model,
        ['.self_attention', '.inter_attention', '.cross_attention', '.core_attention'],
    )


if __name__ == '__main__':
    main()
