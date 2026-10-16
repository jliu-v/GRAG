import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration

from grag.prefusion.dialogue_modules.model_utils import get_last_token_index
from grag.prefusion.dialogue_modules.model_utils import process_padding

logger = logging.getLogger(__name__)


class T5PreFusionModelSFT(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()

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

        outputs = {}

        # Process padding and create attention masks
        batch, length = decoder_input.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        # history, history_attention_mask = process_padding(padding, history)
        # persona, persona_attention_mask = process_padding(padding, persona)
        # response, response_attention_mask = process_padding(padding, response)
        # generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        decoder_input, decoder_attention_mask = process_padding(padding, decoder_input)
        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = decoder_input.size(1)
            # +1 here because decoder_input/lm_labels are prepended with unnecessary leading <pad> token
            lm_labels = lm_labels[:, 1:new_length + 1]

        # # truncate paddings to maximum position of non-padding token in the batch
        # # and create attention mask
        # if padding is not None:
        #     batch, length = input_ids.size()
        #     mask_enc = torch.all(torch.ne(input_ids, padding), dim=0, keepdim=True)
        #     input_ids = torch.masked_select(input_ids, mask_enc).view(batch, -1)
        #     mask_dec = torch.all(torch.ne(decoder_input, padding), dim=0, keepdim=True)
        #     decoder_input = torch.masked_select(decoder_input, mask_dec).view(batch, -1)
        #     if lm_labels is not None:
        #         lm_labels = torch.masked_select(lm_labels, mask_dec).view(batch, -1)
        #     attention_mask = torch.ne(input_ids, padding)
        #     decoder_attention_mask = torch.ne(decoder_input, padding)
        #     assert torch.sum(mask_enc, dim=1)[0] == input_ids.size(1) # make sure sizes match
        #     assert torch.sum(mask_dec, dim=1)[0] == decoder_input.size(1)  # make sure sizes match
        # else:
        #     attention_mask, decoder_attention_mask = None, None

        forward_output = super().forward(input_ids=input_ids,
                                         decoder_input_ids=decoder_input,
                                         attention_mask=input_ids_attention_mask,
                                         decoder_attention_mask=decoder_attention_mask,
                                         output_hidden_states=True)
        # lm_hidden_states = forward_output["decoder_hidden_states"]

        outputs["lm_logits"] = forward_output["logits"]

        if training:
            # LM loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs["lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["lm_loss"] = lm_loss
        else:
            outputs["lm_labels"] = lm_labels

        return outputs


class T5PreFusionModelGRAG(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()

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
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id

        outputs = {}

        batch, length = decoder_input.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        # history, history_attention_mask = process_padding(padding, history)
        # persona, persona_attention_mask = process_padding(padding, persona)
        response, response_attention_mask = process_padding(padding, response)
        generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        decoder_input, decoder_attention_mask = process_padding(padding, decoder_input)
        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = decoder_input.size(1)
            # +1 here because decoder_input/lm_labels are prepended with unnecessary leading <pad> token
            lm_labels = lm_labels[:, 1:new_length + 1]

        forward_output = super().forward(input_ids=input_ids,
                                         decoder_input_ids=decoder_input,
                                         attention_mask=input_ids_attention_mask,
                                         decoder_attention_mask=decoder_attention_mask,
                                         output_hidden_states=True)
        # lm_hidden_states = forward_output["decoder_hidden_states"]

        outputs["lm_logits"] = forward_output["logits"]

        if training:
            # LM loss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = outputs["lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["lm_loss"] = lm_loss

            # JEPA loss
            jepa_weight = kwargs.get("jepa_weight", 0.0)
            if jepa_weight > 0.0:
                context_embedding = forward_output['encoder_last_hidden_state']
                response_embedding = self.encoder(input_ids=response, attention_mask=response_attention_mask)[
                    'last_hidden_state']
                generic_embedding = \
                    self.encoder(input_ids=generic_response, attention_mask=generic_response_attention_mask)[
                        'last_hidden_state']
                context_last_token_index = get_last_token_index(input_ids_attention_mask)
                response_last_token_index = get_last_token_index(response_attention_mask)
                generic_last_token_index = get_last_token_index(generic_response_attention_mask)
                context_embedding = context_embedding[range(context_embedding.shape[0]), context_last_token_index]
                response_embedding = response_embedding[range(response_embedding.shape[0]), response_last_token_index]
                generic_embedding = generic_embedding[range(generic_embedding.shape[0]), generic_last_token_index]

                sim_response = F.cosine_similarity(context_embedding, response_embedding, dim=-1)
                sim_generic = F.cosine_similarity(context_embedding, generic_embedding, dim=-1)

                outputs["jepa_loss"] = torch.mean(sim_generic) - torch.mean(sim_response)

                outputs["lm_loss"] = outputs["lm_loss"] + jepa_weight * outputs["jepa_loss"]

        else:
            outputs["lm_labels"] = lm_labels

        return outputs
