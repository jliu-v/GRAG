import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from grag.postfusion.data_utils import SPECIAL_TOKENS
from grag.postfusion.dialogue_modules.custom_models import CustomQwenDecoderLayerWithStyle
from grag.postfusion.dialogue_modules.model_utils import MaskedMean
from grag.postfusion.dialogue_modules.model_utils import get_last_token_index
from grag.postfusion.dialogue_modules.model_utils import process_padding
from grag.postfusion.dialogue_modules.style_controller import QwenAttentionStyleController

logger = logging.getLogger(__name__)


class Qwen3PostFusionModelGRAG(Qwen3ForCausalLM):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
    _custom_modules = [
        'lm_head',
        'style_controller',
        'style_self_attn',
        'masked_mean'
    ]

    def __init__(self, config):
        super().__init__(config)
        # self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.style_controller = QwenAttentionStyleController(
            embed_dim=config.hidden_size,
            num_heads=1,
            dropout=config.attention_dropout,
            config=config
        )

        self.style_self_attn = Qwen3Attention(config=config, layer_idx=0)
        self.masked_mean = MaskedMean()
        self.replace_decoder_layers_with_custom()

        self.init_weights()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, QwenAttentionStyleController):
            module.apply(module._init_weights)
        self._init_custom_modules()

    def _init_custom_modules(self):
        """
        Re-apply Qwen3's weight initialization to custom modules added
        after super().__init__(). This matches the scaled normal distribution
        used by the rest of the pretrained model.
        """
        for module in [self.lm_head, self.self_attn]:
            self._init_weights(module)
            # _init_weights on the parent only handles the top-level module,
            # so we must recurse into children explicitly
            for submodule in module.modules():
                self._init_weights(submodule)

    def _init_weights(self, module):
        std = self.config.initializer_range  # typically 0.02 for Qwen3
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        else:
            # Catch Qwen3RMSNorm and any other custom norm that isn't
            # a subclass of nn.LayerNorm but has a 'weight' parameter
            if hasattr(module, 'weight') and module.weight is not None:
                # Only handle if it looks like a norm (weight is 1D, not a projection)
                if module.weight.dim() == 1:
                    module.weight.data.fill_(1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()

    def replace_decoder_layers_with_custom(self):
        old_layers = self.model.layers
        new_layers = []

        for i, old_layer in enumerate(old_layers):
            # 1. Create new layer
            new_layer = CustomQwenDecoderLayerWithStyle(self.config, layer_idx=i)
            # 2. Load pretrained weights
            new_layer.load_state_dict(old_layer.state_dict())
            new_layers.append(new_layer)

        # 3. Replace ModuleList
        self.model.layers = torch.nn.ModuleList(new_layers)

    def create_style_embeds(self, hidden_states, attention_mask):

        B, seq_len, dim = hidden_states.shape
        device = hidden_states.device

        # 1. Generate position IDs required for Llama's Rotary Embeddings (RoPE)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        # position_ids = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # 2. Convert 2D boolean mask to 4D float bias mask for LlamaAttention
        # Shape transition: [B, Seq] -> [B, 1, 1, Seq]
        # True becomes 0.0 (attend), False becomes -inf (mask)
        hf_attention_mask = attention_mask[:, None, None, :]
        hf_attention_mask = (1.0 - hf_attention_mask.to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min

        # 3. Execute the LlamaAttention forward pass
        # Output shape: [B, Seq, dim]
        hs = self.style_self_attn(
            hidden_states,
            attention_mask=hf_attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False
        )[0]

        # 4. Compute the masked mean pooling along the sequence dimension (dim=1)
        # Output shape: [B, 1, dim]
        return self.masked_mean(hs, attention_mask, keepdim=True)

    def set_style_embeds_to_decoder_layers(self, style_embeds, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.model
        for layer in decoder.layers:
            setattr(layer, "_style_embeds", style_embeds)

    def set_style_embeds_enabled(self, style_embeds_enabled: bool, decoder: Optional[nn.Module] = None):
        if decoder is None:
            decoder = self.model
        for layer in decoder.layers:
            setattr(layer, "_style_embeds_enabled", style_embeds_enabled)

    def construct_backward_inputs(self, method, tokenizer, history, response, generic_response, pad_token_id=None):
        bos = tokenizer.bos_token_id
        # Check if BOS token exists to determine offset shifting dynamically
        has_bos = bos is not None
        bos_offset = 1 if has_bos else 0

        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id

        if method in ["GRAG", "Hist"]:
            main_inputs = "history-only"
        elif method == "TST":
            main_inputs = "persona-tst"
        else:
            raise ValueError(f"Unknown method for post-fusion models: {method}")

        if main_inputs == "persona-tst":
            main_input_ids = history
        elif main_inputs == "history-only":
            main_input_ids = response
        else:
            raise ValueError(f"Invalid main_inputs: {main_inputs}")

        device = main_input_ids.device
        B = main_input_ids.size(0)

        # 1. Create boolean masks for valid tokens (True means NOT a padding token)
        mask_a = (main_input_ids != pad_token_id)
        mask_b = (generic_response != pad_token_id)

        # 2. Calculate the true unpadded sequence lengths for every individual row
        lens_a = mask_a.sum(dim=1)
        lens_b = mask_b.sum(dim=1)

        # 3. Determine maximum combined length (accounting for the conditional +BOS token slot)
        combined_lens = lens_a + lens_b + bos_offset
        max_combined_len = combined_lens.max().item()

        # 4. Allocate combined arrays (defaulting labels to -100)
        combined_inputs = torch.full((B, max_combined_len), pad_token_id, dtype=main_input_ids.dtype, device=device)
        lm_labels = torch.full((B, max_combined_len), -100, dtype=main_input_ids.dtype, device=device)

        # 5. Conditionally prepend the BOS token to the leading column (index 0) if it exists
        if has_bos:
            combined_inputs[:, 0] = bos

        # 6. Build a continuous index grid to map remaining contents
        grid = torch.arange(max_combined_len, device=device).unsqueeze(0).expand(B, -1)

        # 7. Generate the final combined sequence validity mask
        combined_mask = grid < combined_lens.unsqueeze(1)

        # 8. Place main_input_ids immediately after the BOS token (if present)
        mask_dest_a = (grid >= bos_offset) & (grid < (lens_a + bos_offset).unsqueeze(1))
        combined_inputs[mask_dest_a] = main_input_ids[mask_a]

        # 9. Place input_b immediately following the valid elements of main_input_ids
        mask_dest_b = combined_mask & ~mask_dest_a
        if has_bos:
            mask_dest_b[:, 0] = False  # Explicitly preserve the BOS column boundary if active

        valid_tokens_b = generic_response[mask_b]
        combined_inputs[mask_dest_b] = valid_tokens_b
        lm_labels[mask_dest_b] = valid_tokens_b

        return combined_inputs, combined_mask, lm_labels

    def forward(self,
                input_ids=None,
                lm_labels=None,
                history=None,
                persona=None,
                response=None,
                generic_response=None,
                training=None,
                padding=None,
                tokenizer=None,
                **kwargs):
        method = kwargs.get("method", "history-only")
        agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id

        outputs = {}

        # truncate paddings to maximum position of non-padding token in the batch
        # and create attention mask
        batch, length = input_ids.size()
        input_ids, input_ids_attention_mask = process_padding(padding, input_ids)
        history, history_attention_mask = process_padding(padding, history)
        persona, persona_attention_mask = process_padding(padding, persona)
        generic_response, generic_response_attention_mask = process_padding(padding, generic_response)
        response, response_attention_mask = process_padding(padding, response)
        if lm_labels is not None:
            assert lm_labels.size() == (batch, length)
            new_length = input_ids.size(1)
            # +1 here because input/lm_labels are prepended with unnecessary leading <pad> token
            lm_labels = lm_labels[:, 1:new_length + 1]

        # note:
        # with BOS/EOS: input_ids
        # without BOS/EOS: history, persona, response, generic_response
        self.set_style_embeds_enabled(style_embeds_enabled=False)
        enc_persona = self.model(input_ids=persona, attention_mask=persona_attention_mask)["last_hidden_state"]
        enc_generic_response = self.model(input_ids=generic_response, attention_mask=generic_response_attention_mask)[
            "last_hidden_state"]
        enc_history = self.model(input_ids=history, attention_mask=history_attention_mask)["last_hidden_state"]
        enc_response = None  # only needed for training with regularizations

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
        self.set_style_embeds_enabled(style_embeds_enabled=True)

        fw_decoder_hs = self.model(input_ids=input_ids, attention_mask=input_ids_attention_mask)["last_hidden_state"]
        outputs["fw_lm_logits"] = self.lm_head(fw_decoder_hs)

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
                self.set_style_embeds_enabled(style_embeds_enabled=False)
                enc_response = self.model(input_ids=response,
                                          attention_mask=response_attention_mask)["last_hidden_state"]
                bw_enc_style = self.style_controller.remove_style(query=enc_persona,
                                                                  key=enc_history,
                                                                  value=enc_response,
                                                                  query_mask=persona_attention_mask,
                                                                  key_mask=history_attention_mask,
                                                                  value_mask=response_attention_mask)
                bw_enc_style = self.create_style_embeds(hidden_states=bw_enc_style,
                                                        attention_mask=persona_attention_mask)
                self.set_style_embeds_to_decoder_layers(bw_enc_style)
                self.set_style_embeds_enabled(style_embeds_enabled=True)
                bw_input_ids, bw_attention_mask, bw_lm_labels = self.construct_backward_inputs(method, tokenizer,
                                                                                               history, response,
                                                                                               generic_response,
                                                                                               pad_token_id=padding)
                bw_decoder_hs = self.model(input_ids=bw_input_ids,
                                           attention_mask=bw_attention_mask)["last_hidden_state"]
                outputs["bw_lm_logits"] = self.lm_head(bw_decoder_hs)

                # backward pass
                shift_logits = outputs["bw_lm_logits"][..., :-1, :].contiguous()
                shift_labels = bw_lm_labels[..., 1:].contiguous()
                bw_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                outputs["bw_lm_loss"] = bw_lm_loss
                outputs["lm_loss"] = fw_lm_loss + backward_weight * bw_lm_loss

            # JEPA loss
            jepa_weight = kwargs.get("jepa_weight", 0.0)
            if jepa_weight > 0.0:
                context_embedding = fw_decoder_hs
                self.set_style_embeds_enabled(style_embeds_enabled=False)
                enc_response = self.model(input_ids=response,
                                          attention_mask=response_attention_mask)[
                    "last_hidden_state"] if enc_response is None else enc_response
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
