import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from transformers import GPT2Model
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.bart.modeling_bart import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.models.bart.modeling_bart import BartSdpaAttention

from grag.postfusion.data_utils import SPECIAL_TOKENS
from grag.postfusion.dialogue_modules.custom_models import CustomGPT2DecoderLayerWithStyle
from grag.postfusion.dialogue_modules.model_utils import GumbelSigmoidGate
from grag.postfusion.dialogue_modules.model_utils import MaskedMean
from grag.postfusion.dialogue_modules.model_utils import process_padding
from grag.postfusion.dialogue_modules.style_controller import AttentionStyleController

logger = logging.getLogger(__name__)


class GPT2PostFusionModelGRAG(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = BartEncoder(self.load_encoder_config())
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.max_position = config.n_positions
        self.style_controller = AttentionStyleController(
            embed_dim=config.n_embd,
            num_heads=1,
            dropout=config.embd_pdrop,
            config=config
        )

        self.style_self_attn = BartSdpaAttention(embed_dim=config.n_embd,
                                                 num_heads=1,
                                                 dropout=config.embd_pdrop,
                                                 config=config)
        self.masked_mean = MaskedMean()
        self.replace_decoder_layers_with_custom()

        self.init_weights()

    def load_encoder_config(self):
        return BartConfig(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=1024,
            encoder_layers=3,
            encoder_ffn_dim=4096,
            encoder_attention_heads=16,
            activation_function="gelu",
            d_model=self.config.n_embd,
            dropout=self.config.embd_pdrop,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=False,
            use_cache=True,
            num_labels=3,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
            eos_token_id=self.config.eos_token_id,
            is_encoder_decoder=False
        )

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None,
                                pad_to_multiple_of: Optional[int] = None,
                                mean_resizing: bool = True):
        self.encoder.resize_token_embeddings(new_num_tokens, mean_resizing=mean_resizing)
        super().resize_token_embeddings(new_num_tokens)

    def replace_decoder_layers_with_custom(self):
        old_layers = self.transformer.h
        new_layers = []

        for i, old_layer in enumerate(old_layers):
            # 1. Create new layer
            new_layer = CustomGPT2DecoderLayerWithStyle(self.config, layer_idx=i)
            # 2. Load pretrained weights
            new_layer.load_state_dict(old_layer.state_dict())
            new_layers.append(new_layer)

        # 3. Replace ModuleList
        self.transformer.h = torch.nn.ModuleList(new_layers)

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

        enc_style = self.style_self_attn(hidden_states=hidden_states,
                                         attention_mask=attention_mask_for_sdpa)[0]
        return self.masked_mean(enc_style, attention_mask, keepdim=True)

    def set_style_embeds_to_decoder_layers(self, style_embeds):
        for layer in self.transformer.h:
            setattr(layer, "_style_embeds", style_embeds)

    def forward(self,
                input_ids=None,
                input_tti=None,
                lm_labels=None,
                history=None,
                history_tti=None,
                persona=None,
                persona_tti=None,
                response=None,
                response_tti=None,
                generic_response=None,
                generic_response_tti=None,
                training=None,
                padding=None,
                tokenizer=None,
                **kwargs):

        assert tokenizer is not None
        agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id

        outputs = {}

        batch, length = input_ids.size()

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        history, history_attention_mask = process_padding(padding, history)
        persona, persona_attention_mask = process_padding(padding, persona)
        response, response_attention_mask = process_padding(padding, response)
        generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        input_tti, _ = process_padding(padding, input_tti)
        history_tti, _ = process_padding(padding, history_tti)
        persona_tti, _ = process_padding(padding, persona_tti)
        response_tti, _ = process_padding(padding, response_tti)
        generic_response_tti, _ = process_padding(padding, generic_response_tti)

        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = input_ids.size(1)
            lm_labels = lm_labels[:, :new_length]

        # note:
        # with BOS/EOS: input_ids
        # without BOS/EOS: history, persona, response, generic_response
        enc_persona = self.encoder(input_ids=persona, attention_mask=persona_attention_mask)["last_hidden_state"]
        enc_generic_response = self.encoder(input_ids=generic_response, attention_mask=generic_response_attention_mask)[
            "last_hidden_state"]
        enc_history = self.encoder(input_ids=history, attention_mask=history_attention_mask)["last_hidden_state"]

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

        fw_decoder_hs = self.transformer(input_ids=input_ids, token_type_ids=input_tti,
                                         attention_mask=input_ids_attention_mask)["last_hidden_state"]
        outputs["fw_lm_logits"] = self.lm_head(fw_decoder_hs)

        if training is True:
            # calculate loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # forward pass
            shift_logits = outputs["fw_lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            fw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["fw_lm_loss"] = fw_lm_loss

            # backward pass
            backward_weight = kwargs.get("backward_weight", 1.0)
            if backward_weight > 0.0:
                # input = history + response - persona
                # target = generic_response
                enc_response = self.encoder(input_ids=response, attention_mask=response_attention_mask)[
                    "last_hidden_state"]
                bw_input_ids = torch.cat([
                    torch.tensor([[bos]] * batch, device=generic_response.device),
                    generic_response
                ], dim=1)
                bw_input_tti = torch.cat([
                    torch.tensor([[generic_st]] * batch, device=generic_response.device),
                    generic_response_tti
                ], dim=1)
                bw_attention_mask = torch.ne(bw_input_ids, padding) if padding is not None else None
                bw_enc_style = self.style_controller.remove_style(query=enc_persona,
                                                                  key=enc_history,
                                                                  value=enc_response,
                                                                  query_mask=persona_attention_mask,
                                                                  key_mask=history_attention_mask,
                                                                  value_mask=response_attention_mask)
                bw_enc_style = self.create_style_embeds(hidden_states=bw_enc_style,
                                                        attention_mask=persona_attention_mask)
                self.set_style_embeds_to_decoder_layers(bw_enc_style)
                bw_decoder_hs = self.transformer(input_ids=bw_input_ids, token_type_ids=bw_input_tti,
                                                 attention_mask=bw_attention_mask)["last_hidden_state"]
                outputs["bw_lm_logits"] = self.lm_head(bw_decoder_hs)
                bw_lm_labels = torch.cat([
                    torch.tensor([[-100]] * batch, device=generic_response.device),
                    torch.where(generic_response != tokenizer.pad_token_id, generic_response, -100)
                ], dim=1)

                # backward pass
                shift_logits = outputs["bw_lm_logits"][..., :-1, :].contiguous()
                shift_labels = bw_lm_labels[..., 1:].contiguous()
                bw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                outputs["bw_lm_loss"] = bw_lm_loss
                outputs["lm_loss"] = fw_lm_loss + backward_weight * bw_lm_loss
            else:
                outputs["lm_loss"] = fw_lm_loss


        else:
            outputs["lm_logits"] = outputs["fw_lm_logits"]
            outputs["lm_labels"] = lm_labels

        return outputs

