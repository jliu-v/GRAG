import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM
from transformers import LlamaModel

from grag.prefusion.dialogue_modules.model_utils import get_last_token_index
from grag.prefusion.dialogue_modules.model_utils import process_padding

logger = logging.getLogger(__name__)


class Llama3PreFusionModelSFT(LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head'
    ]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self,
                input_ids=None,
                input_tti=None,
                lm_labels=None,
                persona=None,
                persona_tti=None,
                response=None,
                response_tti=None,
                generic_response=None,
                generic_response_tti=None,
                training=None,
                padding=None,
                **kwargs):

        outputs = {}

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        if padding is not None:
            batch, length = input_ids.size()
            mask = torch.all(torch.ne(input_ids, padding), dim=0, keepdim=True)
            input_ids = torch.masked_select(input_ids, mask).view(batch, -1)
            if lm_labels is not None:
                lm_labels = torch.masked_select(lm_labels, mask).view(batch, -1)
            attention_mask = torch.ne(input_ids, padding)
            assert torch.sum(mask, dim=1)[0] == input_ids.size(1)  # make sure sizes match
        else:
            attention_mask = None

        lm_hidden_states = self.model(input_ids=input_ids,
                                      attention_mask=attention_mask)['last_hidden_state']

        if lm_hidden_states is not None:
            outputs["lm_logits"] = self.lm_head(lm_hidden_states)

        if not training:
            outputs["lm_labels"] = lm_labels
            lm_labels = None

        if lm_labels is not None:
            shift_logits = outputs["lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["lm_loss"] = lm_loss

        return outputs


class Llama3PreFusionModelGRAG(LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self,
                input_ids=None,
                input_tti=None,
                lm_labels=None,
                persona=None,
                persona_tti=None,
                response=None,
                response_tti=None,
                generic_response=None,
                generic_response_tti=None,
                training=None,
                padding=None,
                **kwargs):

        outputs = {}

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        batch, length = input_ids.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        input_tti, input_tti_attention_mask = process_padding(padding, input_tti)
        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = input_ids.size(1)
            lm_labels = lm_labels[:, :new_length]

        lm_hidden_states = self.model(input_ids=input_ids,
                                      attention_mask=input_ids_attention_mask)['last_hidden_state']

        outputs["lm_logits"] = self.lm_head(lm_hidden_states)

        if training:
            # LM loss
            shift_logits = outputs["lm_logits"][..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs["lm_loss"] = lm_loss

            # JEPA loss
            jepa_weight = kwargs.get("jepa_weight", 0.0)
            if jepa_weight > 0.0:
                context_embedding = lm_hidden_states

                response, response_attention_mask = process_padding(padding, response)
                generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
                response_tti, _ = process_padding(padding, response_tti)
                generic_response_tti, _ = process_padding(padding, generic_response_tti)

                response_embedding = self.model(input_ids=response,
                                                attention_mask=response_attention_mask)['last_hidden_state']
                generic_embedding = self.model(input_ids=generic_response,
                                               attention_mask=generic_response_attention_mask)['last_hidden_state']
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
