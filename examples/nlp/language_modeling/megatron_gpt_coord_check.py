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

# To suppress BF16 compile related issue in the CI runs with turing/V100
import pandas as pd
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.mup.coord_check import plot_coord_data, record_coords
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


class CoordCheckMegatronGPTModel(MegatronGPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nsteps = 4
        self._df = []

    def get_df(self):
        return self._df

    def on_train_batch_start(self, batch, batch_idx):
        model = self
        width = self.cfg.hidden_size

        def filter_module_by_name(name: str):
            if self.mcore_gpt and name.endswith('cross_attn_bda') or name.endswith('mlp_bda'):
                return False
            return True

        output_fdict = None
        input_fdict = None
        param_fdict = None

        if batch_idx == self._nsteps:
            return -1

        self._remove_hooks = []
        # add hooks
        for name, module in model.named_modules():
            if filter_module_by_name and not filter_module_by_name(name):
                continue
            self._remove_hooks.append(module.register_forward_hook(
                record_coords(self._df, width, name, batch_idx + 1,
                    output_fdict=output_fdict,
                    input_fdict=input_fdict,
                    param_fdict=param_fdict)))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # loss_mean = outputs

        # remove hooks
        for handle in self._remove_hooks:
            handle.remove()


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    # This import is just to make sure that `seaborn` is installed (it's
    # JIT-imported later).
    import seaborn

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    scalable_widths = cfg.model.get('mup_scalable_widths', [])
    assert scalable_widths, (
        'no `model.mup_scalable_widths` specified; need to specify config values to vary to do coordinate check.'
    )

    base_widths = {}
    for elem in scalable_widths:
        # Get config key to query from `model` config.
        if isinstance(scalable_widths, dict) or isinstance(elem, str):
            cfg_key = elem
        else:
            assert isinstance(elem, (list, tuple)) and 1 <= len(elem) <= 2
            cfg_key = elem[0]

        base_value = OmegaConf.select(cfg.model, cfg_key)
        base_widths[cfg_key] = base_value

    shrink_factors = [1]
    for shrink_factor in [2, 4, 8, 16, 32]:
        if any(base_value // shrink_factor <= 0 for base_value in base_widths.values()):
            break
        shrink_factors.append(shrink_factor)
    df = []
    assert len(shrink_factors) > 1, (
        'cannot perform coordinate check with just one width. '
        'Please increase base scalable widths to make sure they can be divided by powers of two > 1.'
    )

    for shrink_factor in shrink_factors:
        # `set_base_shapes` returns the model
        new_cfg = cfg.copy()
        if hasattr(new_cfg.model.optim, 'sched'):
            del new_cfg.model.optim.sched

        for (cfg_key, base_value) in base_widths.items():
            delta_value = base_value // shrink_factor
            OmegaConf.update(new_cfg.model, cfg_key, delta_value)
        trainer = MegatronTrainerBuilder(new_cfg).create_trainer()

        model = CoordCheckMegatronGPTModel(new_cfg.model, trainer)
        trainer.fit(model)
        assert len(model.get_df()) > 0
        df = model.get_df() + df

        del model
        del trainer

        # Make space for next model allocation.
        torch.cuda.empty_cache()

    df = pd.DataFrame(df)
    # df.to_pickle(f'coord_check_{torch.distributed.get_rank()}.pkl')
    plot_coord_data(df, save_to=f'coord_check_{torch.distributed.get_rank()}.svg', subplot_width=50, subplot_height=40)


if __name__ == '__main__':
    main()
