# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Most of the code here has been copied from:
# https://github.com/microsoft/mup

from typing import Optional

import torch

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import parallel_lm_logits
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.tensor_parallel import ColumnParallelLinear
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    ShardedStateDict = None
    ColumnParallelLinear = torch.Linear
    MCoreGPTModel = None
    make_tp_sharded_tensor_for_checkpoint = None

    HAVE_MEGATRON_CORE = False


class MuReadout(MegatronModule):
    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output, bias=True, sequence_parallel=False, gradient_accumulation_fusion=False):
        super(MuReadout, self).__init__()
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1
        else:
            self.bias = None
        self.parallel_output = parallel_output
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.warn_once = False

    def forward(self, hidden_states, word_embeddings_weight):
        if hasattr(self, 'weight_infshape'):
            width_mult = self.weight_infshape.width_mult()
        elif hasattr(word_embeddings_weight, 'infshape'):
            width_mult = word_embeddings_weight.infshape.width_mult()
        else:
            width_mult = 1.0
            if not self.warn_once:
                logging.warning("need to set_shape before use mu-Transfer readout layer")
            self.warn_once = True
        async_tensor_model_parallel_allreduce = parallel_state.get_tensor_model_parallel_world_size() > 1
        output = parallel_lm_logits(
            hidden_states / width_mult,
            word_embeddings_weight,
            self.parallel_output,
            bias=self.bias,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
        )
        return output


class MCoreMuReadout(ColumnParallelLinear):
    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    Arguments:
        original_linear: ColumnParallelLinear
    """

    def __init__(self, original_linear):
        assert HAVE_MEGATRON_CORE
        torch.nn.Module.__init__(self)
        self._original_linear = original_linear
        self.warn_once = False
        self._register_load_state_dict_pre_hook(self._patch_load_state_dict)

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if hasattr(self, 'weight_infshape'):
            width_mult = self.weight_infshape.width_mult()
        elif hasattr(weight, 'infshape'):
            width_mult = weight.infshape.width_mult()
        else:
            width_mult = 1.0
            if not self.warn_once:
                logging.warning("need to set_shape before use mu-Transfer readout layer")
            self.warn_once = True

        return self._original_linear(input_ / width_mult, weight)

    def _patch_load_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(f'{prefix}_extra_state', None)
        state_dict.setdefault(f'{prefix}_original_linear._extra_state', None)

    def __getattr__(self, name: str):
        if name.startswith('__') or name in ['_original_linear', 'forward', 'weight_infshape', 'warn_once']:
            return super().__getattr__(name)
        original_linear = super().__getattr__('_original_linear')
        return getattr(original_linear, name)


# This is copied almost verbatim from
# `megatron/core/models/gpt/gpt_model.py`, but we modify the
# `output_layer_key` string.
def _patched_sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = ()) -> ShardedStateDict:
    assert not sharded_offsets, "Unexpected sharded offsets"
    sharded_state_dict = {}

    if self.pre_process:
        embedding_prefix = f'{prefix}embedding.'
        embedding_sharded_state_dict = self.embedding.sharded_state_dict(
            prefix=embedding_prefix
        )
        sharded_state_dict.update(embedding_sharded_state_dict)

    decoder_prefix = f'{prefix}decoder.'
    decoder_sharded_state_dict = self.decoder.sharded_state_dict(prefix=decoder_prefix)
    sharded_state_dict.update(decoder_sharded_state_dict)

    if self.post_process:
        output_layer_prefix = f'{prefix}output_layer.'
        output_layer_key = f'{output_layer_prefix}_original_linear.weight'
        if self.share_embeddings_and_output_weights:
            if not self.pre_process:
                # when sharing embeddings with last stage, we need to use the weights from the first stage
                # on pipeline first rank, word embeddings are saved to {prefix}embedding.word_embeddings.weight
                tensor = self.shared_embedding_or_output_weight()
                first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
                last_stage_word_emb_replica_id = (
                    1,  # copy of first stage embedding
                    0,
                    parallel_state.get_data_parallel_rank(with_context_parallel=True),
                )

                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=tensor,
                    key=first_stage_word_emb_key,
                    replica_id=last_stage_word_emb_replica_id,
                    allow_shape_mismatch=True,
                )

                sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

        else:
            output_layer_state_dict = self.output_layer.state_dict(
                prefix=output_layer_prefix, keep_vars=True
            )
            output_layer_tensor = output_layer_state_dict[output_layer_key]
            # independent output layer
            sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                tensor=output_layer_tensor, key=output_layer_key, allow_shape_mismatch=True,
            )

            sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

    return sharded_state_dict


def patch_mcore_gptmodel_for_mup(model):
    # If we don't have Megatron-LM, we have nothing to patch.
    if MCoreGPTModel is None:
        return

    # Do some evil monkey patching in order to fix up state
    # dicts. To make this safe, we explicitly check for whether
    # our code matches and error out if it doesn't.
    import hashlib
    import inspect
    import types

    sharded_state_dict_code = MCoreGPTModel.sharded_state_dict
    sharded_state_dict_func_hash = hashlib.md5(
        inspect.getsource(sharded_state_dict_code).encode(),
    ).hexdigest()
    assert (
        sharded_state_dict_func_hash
        == '192b67d1526c552d03ea830d2374657f'
    ), (
        'cannot patch this version of Megatron-LM for μP. Please '
        'update the state dict patching implementation to support it.'
    )


    model._old_sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = types.MethodType(_patched_sharded_state_dict, model)

    model.output_layer = MCoreMuReadout(model.output_layer)


def rescale_linear_bias(linear):
    '''Rescale bias in nn.Linear layers to convert SP initialization to μP initialization.

    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.
    '''
    if hasattr(linear, '_has_rescaled_params') and linear._has_rescaled_params:
        raise RuntimeError(
            "`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n"
            "If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n"
            "To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call."
        )
    if linear.bias is None:
        return
    fanin_mult = linear.weight.infshape[1].width_mult()
    linear.bias.data *= fanin_mult ** 0.5
    linear._has_rescaled_params = True
