import copy
import logging
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention

from grag.postfusion.data_utils import SPECIAL_TOKENS
from grag.postfusion.dialogue_modules.custom_models import CustomT5DecoderLayerWithStyle
from grag.postfusion.dialogue_modules.model_utils import MaskedMean
from grag.postfusion.dialogue_modules.model_utils import get_last_token_index
from grag.postfusion.dialogue_modules.model_utils import process_padding
from grag.postfusion.dialogue_modules.style_controller import AttentionStyleController
from grag.postfusion.dialogue_modules.style_controller import T5AttentionStyleController

logger = logging.getLogger(__name__)


class T5WithRescale(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.softmax = nn.Softmax(dim=-1)

    def get_t5_sequence_output(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None):
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(input_ids)
        if decoder_attention_mask is None:
            decoder_attention_mask = self._shift_right(attention_mask)
        sequence_output = super().forward(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          decoder_input_ids=decoder_input_ids,
                                          decoder_attention_mask=decoder_attention_mask,
                                          output_hidden_states=True)["decoder_hidden_states"][0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        sequence_output = self.softmax(sequence_output)
        return sequence_output

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AttentionStyleController):
            module.apply(module._init_weights)


class T5PostFusionModelGRAG(T5WithRescale):
    _custom_modules = [
        'lm_head',
        'style_controller',
        'style_self_attn',
        'masked_mean'
    ]

    def __init__(self, config):
        super().__init__(config)
        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.style_controller = T5AttentionStyleController(
            embed_dim=config.d_model,
            num_heads=1,
            dropout=config.dropout_rate,
            config=config
        )
        self.style_self_attn = T5Attention(config,
                                           has_relative_attention_bias=False)

        self.masked_mean = MaskedMean()
        self.replace_decoder_layers_with_custom()
        self.init_weights()

    def replace_decoder_layers_with_custom(self):
        old_layers = self.decoder.block
        new_layers = []

        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers

        for i, old_layer in enumerate(old_layers):
            # 1. Create new layer
            new_layer = CustomT5DecoderLayerWithStyle(decoder_config, has_relative_attention_bias=bool(i == 0),
                                                      layer_idx=i)
            # 2. Load pretrained weights
            new_layer.load_state_dict(old_layer.state_dict())

            new_layers.append(new_layer)

        # 3. Replace ModuleList
        self.decoder.block = torch.nn.ModuleList(new_layers)

    def encode(self,
               input_ids: Optional[torch.LongTensor] = None,
               attention_mask: Optional[torch.Tensor] = None,
               head_mask: Optional[torch.Tensor] = None,
               inputs_embeds: Optional[torch.FloatTensor] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None,
               ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs.last_hidden_state * (self.model_dim ** -0.5)
        return sequence_output

    def decode(self,
               decoder_input_ids: Optional[torch.LongTensor] = None,
               decoder_attention_mask: Optional[torch.LongTensor] = None,
               decoder_head_mask: Optional[torch.Tensor] = None,
               cross_attn_head_mask: Optional[torch.Tensor] = None,
               encoder_hidden_states: Optional[torch.FloatTensor] = None,
               encoder_attention_mask: Optional[torch.Tensor] = None,
               past_key_values: Optional[List[torch.FloatTensor]] = None,
               decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
               use_cache: Optional[bool] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None
               ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return decoder_outputs.last_hidden_state

    def create_style_embeds(self, hidden_states, attention_mask):
        # input_shape = hidden_states.size()[:-1]
        # inputs_embeds = hidden_states
        # past_key_values_length = 0
        # attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa(
        #     attention_mask,
        #     input_shape,
        #     inputs_embeds,
        #     past_key_values_length,
        # )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        extended_mask = (attention_mask == 0).view(attention_mask.shape[0], 1, 1, -1)
        enc_style = self.style_self_attn(hidden_states,
                                         mask=extended_mask,
                                         cache_position=cache_position)[0]
        return self.masked_mean(enc_style, attention_mask, keepdim=True)

    def set_style_embeds_to_decoder_layers(self, style_embeds, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.decoder
        for layer in decoder.block:
            setattr(layer, "_style_embeds", style_embeds)

    def set_style_embeds_enabled(self, style_embeds_enabled: bool, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.decoder
        for layer in decoder.block:
            setattr(layer, "_style_embeds_enabled", style_embeds_enabled)

    def forward(self,
                input_ids=None,
                decoder_input=None,
                lm_labels=None,
                history=None,
                persona=None,
                response=None,
                generic_response=None,
                training=None,
                padding=None,
                tokenizer=None,
                **kwargs):
        assert tokenizer is not None
        agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id

        outputs = {}

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        batch, length = decoder_input.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        history, history_attention_mask = process_padding(padding, history)
        persona, persona_attention_mask = process_padding(padding, persona)
        generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        response, response_attention_mask = process_padding(padding, response)
        decoder_input, decoder_attention_mask = process_padding(padding, decoder_input)
        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = decoder_input.size(1)
            # +1 here because decoder_input/lm_labels are prepended with unnecessary leading <pad> token
            lm_labels = lm_labels[:, 1:new_length + 1]

        # note:
        # with BOS/EOS: input_ids
        # without BOS/EOS: history, persona, response, generic_response
        enc_input_ids = self.encode(input_ids=input_ids, attention_mask=input_ids_attention_mask)
        enc_history = self.encode(input_ids=history, attention_mask=history_attention_mask)
        enc_persona = self.encode(input_ids=persona, attention_mask=persona_attention_mask)
        enc_generic_response = self.encode(input_ids=generic_response, attention_mask=generic_response_attention_mask)
        enc_response = None

        # forward pass
        # input = history + generic_response + persona
        # target = response
        fw_enc_style = self.style_controller.add_style(query=enc_persona,
                                                       key=enc_history,
                                                       value=enc_generic_response,
                                                       query_mask=persona_attention_mask,
                                                       key_mask=history_attention_mask,
                                                       value_mask=generic_response_attention_mask)
        fw_enc_style = self.create_style_embeds(hidden_states=fw_enc_style,
                                                attention_mask=persona_attention_mask)
        self.set_style_embeds_to_decoder_layers(fw_enc_style)

        fw_hidden_state = self.decode(decoder_input_ids=decoder_input,
                                      decoder_attention_mask=decoder_attention_mask,
                                      encoder_hidden_states=enc_input_ids,
                                      encoder_attention_mask=input_ids_attention_mask)
        outputs["fw_lm_logits"] = self.lm_head(fw_hidden_state)

        if training is True:
            # LM loss forward pass
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs["fw_lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["fw_lm_loss"] = lm_loss
            outputs["lm_loss"] = lm_loss

            # backward pass
            backward_weight = kwargs.get("backward_weight", 1.0)
            if backward_weight > 0.0:
                # input = history + response - persona
                # target = generic_response
                enc_response = self.encode(input_ids=response,
                                           attention_mask=response_attention_mask) if enc_response is None else enc_response

                bw_decoder_input = generic_response
                bw_decoder_attention_mask = torch.ne(bw_decoder_input, padding) if padding is not None else None
                bw_enc_style = self.style_controller.remove_style(query=enc_persona,
                                                                  key=enc_history,
                                                                  value=enc_response,
                                                                  query_mask=persona_attention_mask,
                                                                  key_mask=history_attention_mask,
                                                                  value_mask=response_attention_mask)
                bw_enc_style = self.create_style_embeds(hidden_states=bw_enc_style,
                                                        attention_mask=persona_attention_mask)
                self.set_style_embeds_to_decoder_layers(bw_enc_style)
                bw_hidden_state = self.decode(decoder_input_ids=bw_decoder_input,
                                              decoder_attention_mask=bw_decoder_attention_mask,
                                              encoder_hidden_states=enc_response,
                                              encoder_attention_mask=response_attention_mask)
                outputs["bw_lm_logits"] = self.lm_head(bw_hidden_state)
                bw_lm_labels = torch.where(generic_response != tokenizer.pad_token_id, generic_response, -100)

                # backward pass
                shift_logits = outputs["bw_lm_logits"][..., :-1, :].contiguous()
                shift_labels = bw_lm_labels[..., 1:].contiguous()
                bw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                outputs["bw_lm_loss"] = bw_lm_loss
                outputs["lm_loss"] = outputs["lm_loss"] + backward_weight * bw_lm_loss

            # JEPA loss
            jepa_weight = kwargs.get("jepa_weight", 0.0)
            if jepa_weight > 0.0:
                context_embedding = enc_input_ids
                enc_response = self.encode(input_ids=response,
                                           attention_mask=response_attention_mask) if enc_response is None else enc_response
                response_embedding = enc_response
                generic_embedding = enc_generic_response
                context_last_token_index = get_last_token_index(input_ids_attention_mask)
                response_last_token_index = get_last_token_index(response_attention_mask)
                generic_last_token_index = get_last_token_index(generic_response_attention_mask)
                context_embedding = context_embedding[range(context_embedding.shape[0]), context_last_token_index]
                response_embedding = response_embedding[
                    range(response_embedding.shape[0]), response_last_token_index]
                generic_embedding = generic_embedding[range(generic_embedding.shape[0]), generic_last_token_index]

                sim_response = F.cosine_similarity(context_embedding, response_embedding, dim=-1)
                sim_generic = F.cosine_similarity(context_embedding, generic_embedding, dim=-1)

                outputs["jepa_loss"] = torch.mean(sim_generic) - torch.mean(sim_response)
                outputs["lm_loss"] = outputs["lm_loss"] + jepa_weight * outputs["jepa_loss"]

        else:
            outputs["lm_logits"] = outputs["fw_lm_logits"]
            outputs["lm_labels"] = lm_labels

        return outputs
