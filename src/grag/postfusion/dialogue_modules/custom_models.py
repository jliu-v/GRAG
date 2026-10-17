from typing import Optional
from typing import Tuple

import torch
from transformers.models.bart.modeling_bart import BartDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.t5.modeling_t5 import T5Block


class CustomBartDecoderLayerWithStyle(BartDecoderLayer):
    def __init__(self, config, **kwargs):
        super(CustomBartDecoderLayerWithStyle, self).__init__(config)

        conditioning_method = kwargs.get("conditioning_method", None)
        self.conditioning_method = conditioning_method
        self.hidden_dim = config.d_model

        if conditioning_method is not None:
            if conditioning_method == "film":
                self.film_gamma = kwargs.get("film_gamma", None)
                self.film_beta = kwargs.get("film_beta", None)
            elif conditioning_method == "adain":
                self.adain_norm = kwargs.get("adain_norm", None)
                self.adain_gamma = kwargs.get("adain_gamma", None)
                self.adain_beta = kwargs.get("adain_beta", None)
            elif conditioning_method == "gated":
                self.gate_transform = kwargs.get("gate_transform", None)
                self.persona_transform = kwargs.get("persona_transform", None)

    def apply_persona_conditioning(self, hidden_states, style_embeds):
        """Apply persona conditioning to generic response."""
        if self.conditioning_method is None:
            return hidden_states + style_embeds
        elif self.conditioning_method == "film":
            gamma = self.film_gamma(style_embeds)
            beta = self.film_beta(style_embeds)
            return gamma * hidden_states + beta
        elif self.conditioning_method == "adain":
            normalized = self.adain_norm(hidden_states)
            gamma = self.adain_gamma(style_embeds)
            beta = self.adain_beta(style_embeds)
            return gamma * normalized + beta
        elif self.conditioning_method == "gated":
            persona_features = self.persona_transform(style_embeds)
            persona_features = persona_features.expand(-1, hidden_states.size(1), -1)
            combined = torch.cat([hidden_states, persona_features], dim=-1)
            gate = torch.sigmoid(self.gate_transform(combined))
            return gate * hidden_states + (1 - gate) * persona_features

    def forward(
            self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        style_embeds = getattr(self, "_style_embeds", None)
        style_embeds_enabled = getattr(self, "_style_embeds_enabled", True)

        if style_embeds is None and style_embeds_enabled:
            raise ValueError("Attribute _style_embeds is not set when _style_embeds_enabled = True")

        output = super().forward(*args, **kwargs)

        if style_embeds_enabled:
            hidden_states: torch.Tensor = output[0]
            hidden_states = self.apply_persona_conditioning(hidden_states, style_embeds)
            output = (hidden_states,) + output[1:]

        return output


class CustomBartDecoderLayerWithStyleAndCopy(CustomBartDecoderLayerWithStyle):
    def __init__(self, config, **kwargs):
        super(CustomBartDecoderLayerWithStyleAndCopy, self).__init__(config, **kwargs)

        self.use_copy_mechanism = kwargs.get("use_copy_mechanism", False)
        self.copy_cross_attention = None
        self.copy_layer_norm = None
        self._copy_layer_ready = False

        # make copy_weights an attribute so that it can be accessed by caller
        self._copy_weights = None

    def set_copy_layer(self, copy_cross_attention, copy_layer_norm):
        if self.use_copy_mechanism:
            self.copy_cross_attention = copy_cross_attention
            self.copy_layer_norm = copy_layer_norm
            self._copy_layer_ready = True

    def set_copy_weights(self, copy_weights):
        self._copy_weights = copy_weights

    def clear_copy_weights(self):
        self._copy_weights = None

    def get_copy_weights(self):
        return self._copy_weights

    def forward(self, *args, **kwargs) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        secondary_hidden_states = getattr(self, "_secondary_hidden_states", None)
        secondary_attention_mask = getattr(self, "_secondary_attention_mask", None)

        if secondary_hidden_states is None and self.use_copy_mechanism:
            raise ValueError("Attribute _secondary_hidden_states is not set when self.use_copy_mechanism = True")

        output = super().forward(*args, **kwargs)
        if self.use_copy_mechanism and self._copy_layer_ready:
            hidden_states: torch.Tensor = output[0]
            residual = hidden_states
            decoder_hidden_norm = self.copy_layer_norm(hidden_states)

            copy_attn_output, copy_weights = self.copy_cross_attention(
                query=decoder_hidden_norm,
                key=secondary_hidden_states,
                value=secondary_hidden_states,
                key_padding_mask=~secondary_attention_mask.bool() if secondary_attention_mask is not None else None
            )
            hidden_states = residual + copy_attn_output
            output = (hidden_states,) + output[1:]
            self.set_copy_weights(copy_weights)

        return output


class CustomGPT2DecoderLayerWithStyle(GPT2Block):
    def __init__(self, *args, **kwargs):
        super(CustomGPT2DecoderLayerWithStyle, self).__init__(*args, **kwargs)

    def forward(
            self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        style_embeds = getattr(self, "_style_embeds", None)
        if style_embeds is None:
            raise ValueError("Attribute _style_embeds is not set")

        output = super().forward(*args, **kwargs)
        hidden_states: torch.Tensor = output[0]
        hidden_states = hidden_states + style_embeds

        return (hidden_states,) + output[1:]


class CustomT5DecoderLayerWithStyle(T5Block):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        conditioning_method = kwargs.get("conditioning_method", None)
        self.conditioning_method = conditioning_method
        self.hidden_dim = config.d_model

        if conditioning_method is not None:
            if conditioning_method == "film":
                self.film_gamma = kwargs.get("film_gamma", None)
                self.film_beta = kwargs.get("film_beta", None)
            elif conditioning_method == "adain":
                self.adain_norm = kwargs.get("adain_norm", None)
                self.adain_gamma = kwargs.get("adain_gamma", None)
                self.adain_beta = kwargs.get("adain_beta", None)
            elif conditioning_method == "gated":
                self.gate_transform = kwargs.get("gate_transform", None)
                self.persona_transform = kwargs.get("persona_transform", None)

    def apply_persona_conditioning(self, hidden_states, style_embeds):
        """Apply persona conditioning to generic response."""
        if self.conditioning_method is None:
            return hidden_states + style_embeds
        elif self.conditioning_method == "film":
            gamma = self.film_gamma(style_embeds)
            beta = self.film_beta(style_embeds)
            return gamma * hidden_states + beta
        elif self.conditioning_method == "adain":
            normalized = self.adain_norm(hidden_states)
            gamma = self.adain_gamma(style_embeds)
            beta = self.adain_beta(style_embeds)
            return gamma * normalized + beta
        elif self.conditioning_method == "gated":
            persona_features = self.persona_transform(style_embeds)
            persona_features = persona_features.expand(-1, hidden_states.size(1), -1)
            combined = torch.cat([hidden_states, persona_features], dim=-1)
            gate = torch.sigmoid(self.gate_transform(combined))
            return gate * hidden_states + (1 - gate) * persona_features

    def forward(
            self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        style_embeds = getattr(self, "_style_embeds", None)
        style_embeds_enabled = getattr(self, "_style_embeds_enabled", True)

        if style_embeds is None and style_embeds_enabled:
            raise ValueError("Attribute _style_embeds is not set when _style_embeds_enabled = True")

        output = super().forward(*args, **kwargs)

        if style_embeds_enabled:
            hidden_states: torch.Tensor = output[0]
            hidden_states = self.apply_persona_conditioning(hidden_states, style_embeds)
            output = (hidden_states,) + output[1:]

        return output


class CustomLlamaDecoderLayerWithStyle(LlamaDecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        conditioning_method = kwargs.get("conditioning_method", None)
        self.conditioning_method = conditioning_method
        self.hidden_dim = config.hidden_size

        if conditioning_method is not None:
            if conditioning_method == "film":
                self.film_gamma = kwargs.get("film_gamma", None)
                self.film_beta = kwargs.get("film_beta", None)
            elif conditioning_method == "adain":
                self.adain_norm = kwargs.get("adain_norm", None)
                self.adain_gamma = kwargs.get("adain_gamma", None)
                self.adain_beta = kwargs.get("adain_beta", None)
            elif conditioning_method == "gated":
                self.gate_transform = kwargs.get("gate_transform", None)
                self.persona_transform = kwargs.get("persona_transform", None)

    def apply_persona_conditioning(self, hidden_states, style_embeds):
        """Apply persona conditioning to generic response."""
        if self.conditioning_method is None:
            return hidden_states + style_embeds
        elif self.conditioning_method == "film":
            gamma = self.film_gamma(style_embeds)
            beta = self.film_beta(style_embeds)
            return gamma * hidden_states + beta
        elif self.conditioning_method == "adain":
            normalized = self.adain_norm(hidden_states)
            gamma = self.adain_gamma(style_embeds)
            beta = self.adain_beta(style_embeds)
            return gamma * normalized + beta
        elif self.conditioning_method == "gated":
            persona_features = self.persona_transform(style_embeds)
            persona_features = persona_features.expand(-1, hidden_states.size(1), -1)
            combined = torch.cat([hidden_states, persona_features], dim=-1)
            gate = torch.sigmoid(self.gate_transform(combined))
            return gate * hidden_states + (1 - gate) * persona_features

    def forward(
            self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        style_embeds = getattr(self, "_style_embeds", None)
        style_embeds_enabled = getattr(self, "_style_embeds_enabled", True)

        if style_embeds is None and style_embeds_enabled:
            raise ValueError("Attribute _style_embeds is not set when _style_embeds_enabled = True")

        output = super().forward(*args, **kwargs)

        if style_embeds_enabled:
            hidden_states: torch.Tensor = output[0]
            hidden_states = self.apply_persona_conditioning(hidden_states, style_embeds)
            output = (hidden_states,) + output[1:]

        return output


class CustomQwenDecoderLayerWithStyle(Qwen3DecoderLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        conditioning_method = kwargs.get("conditioning_method", None)
        self.conditioning_method = conditioning_method
        self.hidden_dim = config.hidden_size

        if conditioning_method is not None:
            if conditioning_method == "film":
                self.film_gamma = kwargs.get("film_gamma", None)
                self.film_beta = kwargs.get("film_beta", None)
            elif conditioning_method == "adain":
                self.adain_norm = kwargs.get("adain_norm", None)
                self.adain_gamma = kwargs.get("adain_gamma", None)
                self.adain_beta = kwargs.get("adain_beta", None)
            elif conditioning_method == "gated":
                self.gate_transform = kwargs.get("gate_transform", None)
                self.persona_transform = kwargs.get("persona_transform", None)

    def apply_persona_conditioning(self, hidden_states, style_embeds):
        """Apply persona conditioning to generic response."""
        if self.conditioning_method is None:
            return hidden_states + style_embeds
        elif self.conditioning_method == "film":
            gamma = self.film_gamma(style_embeds)
            beta = self.film_beta(style_embeds)
            return gamma * hidden_states + beta
        elif self.conditioning_method == "adain":
            normalized = self.adain_norm(hidden_states)
            gamma = self.adain_gamma(style_embeds)
            beta = self.adain_beta(style_embeds)
            return gamma * normalized + beta
        elif self.conditioning_method == "gated":
            persona_features = self.persona_transform(style_embeds)
            persona_features = persona_features.expand(-1, hidden_states.size(1), -1)
            combined = torch.cat([hidden_states, persona_features], dim=-1)
            gate = torch.sigmoid(self.gate_transform(combined))
            return gate * hidden_states + (1 - gate) * persona_features

    def forward(
            self, *args, **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        style_embeds = getattr(self, "_style_embeds", None)
        style_embeds_enabled = getattr(self, "_style_embeds_enabled", True)

        if style_embeds is None and style_embeds_enabled:
            raise ValueError("Attribute _style_embeds is not set when _style_embeds_enabled = True")

        output = super().forward(*args, **kwargs)

        if style_embeds_enabled:
            hidden_states: torch.Tensor = output[0]
            hidden_states = self.apply_persona_conditioning(hidden_states, style_embeds)
            output = (hidden_states,) + output[1:]

        return output
