import logging
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration
from transformers import BartModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.bart.modeling_bart import BartSdpaAttention

from grag.postfusion.data_utils import SPECIAL_TOKENS
from grag.postfusion.dialogue_modules.custom_models import CustomBartDecoderLayerWithStyle
from grag.postfusion.dialogue_modules.model_utils import MaskedMean
from grag.postfusion.dialogue_modules.model_utils import get_last_token_index
from grag.postfusion.dialogue_modules.model_utils import process_padding
from grag.postfusion.dialogue_modules.model_utils import process_super_batch
from grag.postfusion.dialogue_modules.style_controller import AttentionStyleController

logger = logging.getLogger(__name__)


class BARTPostFusionModelGRAG(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head',
        'style_controller',
        'style_self_attn',
        'masked_mean'
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_position = config.max_position_embeddings
        self.style_controller = AttentionStyleController(
            embed_dim=config.d_model,
            num_heads=1,
            dropout=config.attention_dropout,
            config=config
        )
        self.style_self_attn = BartSdpaAttention(embed_dim=config.d_model,
                                                 num_heads=1,
                                                 dropout=config.attention_dropout,
                                                 config=config)
        self.masked_mean = MaskedMean()
        self.replace_decoder_layers_with_custom()

        self.init_weights()

    def replace_decoder_layers_with_custom(self):
        old_layers = self.model.decoder.layers
        new_layers = []

        for old_layer in old_layers:
            # 1. Create new layer
            new_layer = CustomBartDecoderLayerWithStyle(self.config)
            # 2. Load pretrained weights
            new_layer.load_state_dict(old_layer.state_dict())

            new_layers.append(new_layer)

        # 3. Replace ModuleList
        self.model.decoder.layers = torch.nn.ModuleList(new_layers)

    def encode(self,
               encoder: Optional[nn.Module] = None,
               input_ids: Optional[torch.LongTensor] = None,
               attention_mask: Optional[torch.Tensor] = None,
               head_mask: Optional[torch.Tensor] = None,
               inputs_embeds: Optional[torch.FloatTensor] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None,
               ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if encoder is None:
            encoder = self.model.encoder

        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs.last_hidden_state

    def decode(self,
               decoder: Optional[nn.Module] = None,
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
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if decoder is None:
            decoder = self.model.decoder

        decoder_outputs = decoder(
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
            return_dict=return_dict
        )
        return decoder_outputs.last_hidden_state

    def create_style_embeds(self, hidden_states, attention_mask):
        input_shape = hidden_states.size()[:-1]
        inputs_embeds = hidden_states
        past_key_values_length = 0
        attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )

        enc_style = self.style_self_attn(hidden_states,
                                         attention_mask=attention_mask_for_sdpa)[0]
        return self.masked_mean(enc_style, attention_mask, keepdim=True)

    def set_style_embeds_to_decoder_layers(self, style_embeds, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.model.decoder
        for layer in decoder.layers:
            setattr(layer, "_style_embeds", style_embeds)

    def set_style_embeds_enabled(self, style_embeds_enabled: bool, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.model.decoder
        for layer in decoder.layers:
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

        # Optimization: Super-batch encoding
        use_super_batch = kwargs.get("use_super_batch", True)
        backward_weight = kwargs.get("backward_weight", 1.0)
        jepa_weight = kwargs.get("jepa_weight", 0.0)
        need_response_encoding = training is True and (backward_weight > 0.0 or jepa_weight > 0.0)

        if use_super_batch:
            # Prepare inputs for super-batch
            encoder_inputs = [input_ids, history, persona, generic_response]
            encoder_masks = [input_ids_attention_mask, history_attention_mask, persona_attention_mask,
                             generic_response_attention_mask]
            if need_response_encoding:
                encoder_inputs.append(response)
                encoder_masks.append(response_attention_mask)

            # Process super-batch using utility function
            super_input_ids, super_attention_mask, split_fn = process_super_batch(
                encoder_inputs, encoder_masks, tokenizer.pad_token_id, batch
            )

            # Encode everything in one call
            super_enc_outputs = self.encode(input_ids=super_input_ids, attention_mask=super_attention_mask)

            # Split results back
            enc_outputs_split = split_fn(super_enc_outputs)
            enc_input_ids = enc_outputs_split[0]
            enc_history = enc_outputs_split[1]
            enc_persona = enc_outputs_split[2]
            enc_generic_response = enc_outputs_split[3]
            enc_response = enc_outputs_split[4] if need_response_encoding else None
        else:
            # Original processing: separate self.encode() calls
            # note:
            # with BOS/EOS: input_ids
            # without BOS/EOS: history, persona, response, generic_response
            enc_input_ids = self.encode(input_ids=input_ids, attention_mask=input_ids_attention_mask)
            enc_history = self.encode(input_ids=history, attention_mask=history_attention_mask)
            enc_persona = self.encode(input_ids=persona, attention_mask=persona_attention_mask)
            enc_generic_response = self.encode(input_ids=generic_response,
                                               attention_mask=generic_response_attention_mask)
            enc_response = None  # only needed for training with regularizations

        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = decoder_input.size(1)
            lm_labels = lm_labels[:, :new_length]

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
            # calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # forward pass
            shift_logits = outputs["fw_lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            fw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["fw_lm_loss"] = fw_lm_loss
            outputs["lm_loss"] = fw_lm_loss

            # backward pass
            backward_weight = kwargs.get("backward_weight", 1.0)
            if backward_weight > 0.0:
                # input = history + response - persona
                # target = generic_response
                # enc_response is already computed in super-batch if needed
                if enc_response is None:
                    enc_response = self.encode(input_ids=response, attention_mask=response_attention_mask)

                bw_decoder_input = torch.cat([
                    torch.tensor([[bos]] * batch, device=generic_response.device),
                    generic_response
                ], dim=1)
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
                bw_lm_labels = torch.cat([
                    torch.tensor([[-100]] * batch, device=generic_response.device),
                    torch.where(generic_response != tokenizer.pad_token_id, generic_response, -100)
                ], dim=1)

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
                if enc_response is None:
                    enc_response = self.encode(input_ids=response, attention_mask=response_attention_mask)
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


class BARTPostFusionModelSFT(BARTPostFusionModelGRAG):
    """
    BARTPostFusionModelSFT is a post-fusion model without generic responses.
    It is created from BARTPostFusionModelGRAG for the Scaffold Controls experiments
    """

    def forward(self,
                input_ids=None,
                decoder_input=None,
                lm_labels=None,
                history=None,
                persona=None,
                response=None,
                # generic_response=None,
                training=None,
                padding=None,
                tokenizer=None,
                **kwargs):
        assert tokenizer is not None
        agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id

        outputs = {}

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        batch, length = decoder_input.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        history, history_attention_mask = process_padding(padding, history)
        persona, persona_attention_mask = process_padding(padding, persona)
        # generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        response, response_attention_mask = process_padding(padding, response)
        decoder_input, decoder_attention_mask = process_padding(padding, decoder_input)

        # Optimization: Super-batch encoding
        use_super_batch = kwargs.get("use_super_batch", True)

        if use_super_batch:
            # Prepare inputs for super-batch
            encoder_inputs = [input_ids, history, persona]
            encoder_masks = [input_ids_attention_mask, history_attention_mask, persona_attention_mask]

            # Process super-batch using utility function
            super_input_ids, super_attention_mask, split_fn = process_super_batch(
                encoder_inputs, encoder_masks, tokenizer.pad_token_id, batch
            )

            # Encode everything in one call
            super_enc_outputs = self.encode(input_ids=super_input_ids, attention_mask=super_attention_mask)

            # Split results back
            enc_outputs_split = split_fn(super_enc_outputs)
            enc_input_ids = enc_outputs_split[0]
            enc_history = enc_outputs_split[1]
            enc_persona = enc_outputs_split[2]
        else:
            # Original processing: separate self.encode() calls
            # note:
            # with BOS/EOS: input_ids
            # without BOS/EOS: history, persona, response, generic_response
            enc_input_ids = self.encode(input_ids=input_ids, attention_mask=input_ids_attention_mask)
            enc_history = self.encode(input_ids=history, attention_mask=history_attention_mask)
            enc_persona = self.encode(input_ids=persona, attention_mask=persona_attention_mask)

        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = decoder_input.size(1)
            lm_labels = lm_labels[:, :new_length]

        # forward pass
        # input = history + generic_response + persona
        # target = response
        fw_enc_style = self.style_controller.add_style(query=enc_persona,
                                                       key=enc_history,
                                                       value=None,
                                                       query_mask=persona_attention_mask,
                                                       key_mask=history_attention_mask,
                                                       value_mask=None)
        fw_enc_style = self.create_style_embeds(hidden_states=fw_enc_style,
                                                attention_mask=persona_attention_mask)
        self.set_style_embeds_to_decoder_layers(fw_enc_style)

        fw_hidden_state = self.decode(decoder_input_ids=decoder_input,
                                      decoder_attention_mask=decoder_attention_mask,
                                      encoder_hidden_states=enc_input_ids,
                                      encoder_attention_mask=input_ids_attention_mask)
        outputs["fw_lm_logits"] = self.lm_head(fw_hidden_state)

        if training is True:
            # calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # forward pass
            shift_logits = outputs["fw_lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            fw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["fw_lm_loss"] = fw_lm_loss
            outputs["lm_loss"] = fw_lm_loss

        else:
            outputs["lm_logits"] = outputs["fw_lm_logits"]
            outputs["lm_labels"] = lm_labels

        return outputs
