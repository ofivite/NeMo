import collections
import math

import torch

from nemo.collections.nlp.modules.common.megatron.mup.init import normal_
from nemo.collections.nlp.modules.common.megatron.mup.layer import patch_mcore_gptmodel_for_mup
from nemo.collections.nlp.modules.common.megatron.mup.shape import load_base_head_widths, set_base_shapes
from nemo.utils import logging


def maybe_mup_init(module):
    self = module

    assert hasattr(self, 'cfg'), 'Need `cfg` attribute in model.'
    assert hasattr(self, 'trainer'), 'Need `trainer` attribute in model.'
    assert hasattr(self, 'model'), 'Need `model` attribute in model.'
    assert isinstance(self, torch.nn.Module), 'Can only convert `torch.nn.Module` to μP.'

    mcore_gpt = hasattr(self, 'mcore_gpt') and self.mcore_gpt

    if (
            self.cfg.get('make_mup', False)
            and self.cfg.get('fsdp', False)
            and not self.cfg.get('fsdp_use_orig_params', False)
    ):
        raise ValueError('To use μP, it is required to set `model.fsdp_use_orig_params=True`.')

    if self.cfg.get('make_mup', False) and mcore_gpt:
        patch_mcore_gptmodel_for_mup(self.model)

    if self.cfg.get('make_mup', False) and hasattr(self.cfg, "shape_file"):
        set_base_shapes(self, self.register_artifact("shape_file", self.cfg.shape_file), rescale_params=False)

        # here manually initialize all the named parameters with the muTranfer normal initializer
        for name, tensor in self.named_parameters():
            # MLP output, no NVTE: .dense_4h_to_h.weight
            # Attention output, no NVTE: .dense.weight
            # MLP output, NVTE: .fc2_weight
            # Attention output, NVTE: .proj.weight
            # MLP output, MCore: .linear_fc2.weight
            # Attention output, MCore: .linear_proj.weight
            if (
                    name.endswith('.dense_4h_to_h.weight')
                    or name.endswith('.dense.weight')
                    or name.endswith('.fc2_weight')
                    or name.endswith('.proj.weight')
                    or name.endswith('.linear_fc2.weight')
                    or name.endswith('.linear_proj.weight')
            ):
                # initialize all the output dense matrix weight
                # match the megatron lm model
                std = self.cfg.init_method_std
                if self.cfg.get('use_scaled_init_method', True):
                    std = std / math.sqrt(2.0 * self.cfg.num_layers)
                # Previous version
                # std = self.cfg.init_method_std / math.sqrt(2.0 * 12.0)
                normal_(tensor, 0, std)
            # LayerNorm weight: layernorm.weight
            # (Switch)MLP LayerNorm weight, no NVTE: .normalization.weight
            # NormFormer LayerNorm weight, no NVTE: normformer_norm.weight
            # QKV/MLP LayerNorm weight, NVTE: .layer_norm_weight
            # LayerNorm weight, MCore: layernorm.weight
            elif (
                    name.endswith('layernorm.weight')
                    or name.endswith('.normalization.weight')
                    or name.endswith('normformer_norm.weight')
                    or name.endswith('.layer_norm_weight')
            ):
                # initialize all the layer norm weight
                if tensor.std() != 0 and tensor.mean() != 1:
                    raise ValueError(f'need to check {name} init')
                normal_(tensor, 1, 0)
            # Linear weight: .weight
            # MLP weight, NVTE: .fc1_weight
            # QKV weights, NVTE: .query_weight, .key_weight, .value_weight
            # MLP weight, MCore: .linear_fc1.weight
            # QKV weight, MCore: .linear_qkv.weight
            elif (
                    name.endswith('.weight')
                    or name.endswith('.fc1_weight')
                    or name.endswith('.query_weight')
                    or name.endswith('.key_weight')
                    or name.endswith('.value_weight')
                    or name.endswith('.linear_fc1.weight')
                    or name.endswith('.linear_qkv.weight')
            ):
                # initialize all the other dense matrix weight
                normal_(tensor, 0, self.cfg.init_method_std)
                if self.cfg.get('mup_query_zero_init', False):
                    kv_channels = self.cfg.get('kv_channels', None)
                    hidden_size = self.cfg.hidden_size
                    num_attention_heads = self.cfg.num_attention_heads
                    if kv_channels is None:
                        assert (
                            hidden_size % num_attention_heads == 0
                        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
                        kv_channels = hidden_size // num_attention_heads

                    query_projection_size = kv_channels * num_attention_heads

                    if name.endswith('.query_key_value.weight') or name.endswith('.linear_qkv.weight'):
                        tensor.data[:query_projection_size, :] = 0
                    elif name.endswith('.query.weight') or name.endswith('.query_weight') or name.endswith('.query_layer.weight'):
                        tensor.data.zero_()
                if self.cfg.get('mup_readout_zero_init', False):
                    # We do not zero shared embeddings.
                    if mcore_gpt and name.endswith('.output_layer.weight'):
                        tensor.data.zero_()
                    elif name.endswith('.language_model.output_layer.weight'):
                        tensor.data.zero_()
            # TODO .head_scale_tensor anywhere?
            else:
                if tensor.std() != 0 and tensor.mean() != 0:
                    raise ValueError(f'need to check {name} init')

        # here manually overwrite the norm factor
        # Previous version
        # note, has to turn off the model.apply_query_key_layer_scaling
        # assert not self.cfg.apply_query_key_layer_scaling
        apply_query_key_layer_scaling = self.cfg.get('apply_query_key_layer_scaling', False)
        fp16_enabled = self.trainer.precision in [16, '16', '16-mixed']
        if mcore_gpt and not fp16_enabled:
            # The option is automatically turned off in this case, so we should not handle it here.
            apply_query_key_layer_scaling = False

        base_head_widths = load_base_head_widths(self.cfg.shape_file)
        # The user specified a custom value.
        if self.cfg.get('mup_base_model_head_width', None) is not None:
            base_head_width = self.cfg.mup_base_model_head_width
            assert isinstance(base_head_width, int), 'manually specified base model head width can only be single integer'
            attn_norm_head_divisor = math.sqrt(base_head_width)
            attn_norm_head_divisors = collections.defaultdict(lambda: attn_norm_head_divisor)
        elif not base_head_widths:
            # Use 8 as default, meaning a base head width of 64 is assumed.
            attn_norm_head_divisor = 8.0
            attn_norm_head_divisors = collections.defaultdict(lambda: attn_norm_head_divisor)
        else:
            # Here we don't use a `defaultdict` so that we get errors for missing values.
            attn_norm_head_divisors = {name: math.sqrt(head_width) for (name, head_width) in base_head_widths.items()}

        for name, layer in self.named_modules():
            if (
                name.endswith('.self_attention')
                or name.endswith('.inter_attention')
                or name.endswith('.cross_attention')
                or name.endswith('.core_attention')
            ):
                if hasattr(layer, 'norm_factor') and hasattr(layer, 'hidden_size_per_attention_head'):
                    if apply_query_key_layer_scaling and hasattr(layer, 'layer_number'):
                        extra_factor = layer.layer_number
                    else:
                        extra_factor = 1.0
                    layer.norm_factor = layer.hidden_size_per_attention_head / attn_norm_head_divisors[name] * extra_factor
                    # Previous version
                    # layer.norm_factor = (
                    #     layer.hidden_size_per_attention_head / 8.0
                    # )  # divide 8 to make it consist with ADLR setting
                elif hasattr(layer, 'hidden_size_per_attention_head'):
                    for sublayer_name in ['flash_attention', 'fused_attention', 'unfused_attention']:
                        if hasattr(layer, sublayer_name):
                            sublayer = getattr(layer, sublayer_name)
                            if hasattr(sublayer, 'norm_factor'):
                                if apply_query_key_layer_scaling and hasattr(sublayer, 'layer_number'):
                                    extra_factor = sublayer.layer_number
                                else:
                                    extra_factor = 1.0
                                sublayer.norm_factor = (
                                    layer.hidden_size_per_attention_head
                                    / attn_norm_head_divisors[name]
                                    * extra_factor
                                )
                                # Previous version
                                # sublayer.norm_factor = (
                                #     layer.hidden_size_per_attention_head / 8.0
                                # )  # divide 8 to make it consist with ADLR setting
            elif (
                name.endswith('.flash_attention')
                or name.endswith('.fused_attention')
                or name.endswith('.unfused_attention')
            ):
                # These are handled in the else-block above.
                pass
            else:
                if hasattr(layer, 'norm_factor') or hasattr(layer, 'hidden_size_per_attention_head'):
                    logging.error(
                        f'module {name} has norm factor but its name is not ending with attention, need to double check'
                    )

        # Manually set `MuReadout` infshape for FSDP support.
        if mcore_gpt:
            self.model.output_layer.weight_infshape = (
                self.model.output_layer.weight
                if not self.model.share_embeddings_and_output_weights
                else self.model.shared_embedding_or_output_weight()
            ).infshape
        else:
            self.model.tokens_head.weight_infshape = (
                self.model.language_model.output_layer.weight
                if not self.model.share_embeddings_and_output_weights
                else self.model.word_embeddings_weight()
            ).infshape
