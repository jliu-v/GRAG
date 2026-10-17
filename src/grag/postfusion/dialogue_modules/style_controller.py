from typing import Any
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.t5.modeling_t5 import T5Attention


class AttentionStyleController(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 config: Optional[Any] = None
                 ):
        super().__init__()
        self.style_adder = BartAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            config=config
        )
        self.style_remover = BartAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            config=config
        )

    def _init_weights(self, module):
        # Apply custom initialization logic here
        if isinstance(module, torch.nn.Linear):
            # torch.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    @staticmethod
    def create_attention_inputs(query, key, value, query_mask=None, key_mask=None, value_mask=None):
        dtype = query.dtype
        kv = [x for x in [key, value] if x is not None]
        kv = torch.cat(kv, dim=1)
        if query_mask is not None and key_mask is not None and key_mask is not None:
            q_mask = query_mask.unsqueeze(-1).to(dtype)
            kv_mask = [x for x in [key_mask, value_mask] if x is not None]
            kv_mask = torch.cat(kv_mask, dim=1).unsqueeze(-2).to(dtype)
            qkv_mask = torch.bmm(q_mask, kv_mask).unsqueeze(1)
        elif query_mask is None and key_mask is None and value_mask is None:
            qkv_mask = None
        else:
            raise ValueError("query_mask, key_mask, value_mask should be either all None or all provided.")

        return query, kv, qkv_mask

    def add_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, qkv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        hs = self.style_adder(
            hidden_states=query,
            key_value_states=kv,
            attention_mask=qkv_mask
        )[0]
        return hs

    def remove_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, qkv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        hs = self.style_remover(
            hidden_states=query,
            key_value_states=kv,
            attention_mask=qkv_mask
        )[0]
        return hs


class T5AttentionStyleController(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 config: Optional[Any] = None
                 ):
        super().__init__()
        self.style_adder = T5Attention(
            config,
            has_relative_attention_bias=False
        )
        self.style_remover = T5Attention(
            config,
            has_relative_attention_bias=False
        )

    @staticmethod
    def create_attention_inputs(query, key, value, query_mask=None, key_mask=None, value_mask=None):
        dtype = query.dtype
        kv = torch.cat([key, value], dim=1)
        if query_mask is not None and key_mask is not None and key_mask is not None:
            q_mask = query_mask.to(dtype)
            kv_mask = torch.cat([key_mask, value_mask], dim=1).to(dtype)
        elif query_mask is None and key_mask is None and value_mask is None:
            q_mask = None
            kv_mask = None
        else:
            raise ValueError("query_mask, key_mask, value_mask should be either all None or all provided.")

        return query, kv, q_mask, kv_mask

    def add_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        B, T_q, _ = query.shape
        B, T_k, _ = kv.shape

        # Convert your binary kv_mask (1=valid, 0=pad) to T5 additive format
        attn_mask = (kv_mask == 0)  # (B, T_k), True=padding
        attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, T_k)
        attn_mask = attn_mask.expand(B, 1, T_q, T_k)  # (B, 1, T_q, T_k)
        attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(
            attn_mask, torch.finfo(query.dtype).min  # use dtype-safe large negative
        )
        cache_position = torch.arange(query.shape[1], device=query.device)

        hs = self.style_adder(
            hidden_states=query,
            key_value_states=kv,  # ← triggers cross-attention mode
            mask=attn_mask,
            cache_position=cache_position
        )[0]  # returns (attn_output, present_kv, position_bias, ...)

        hs = hs * q_mask.unsqueeze(-1)
        return hs

    def remove_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        B, T_q, _ = query.shape
        B, T_k, _ = kv.shape

        # Convert your binary kv_mask (1=valid, 0=pad) to T5 additive format
        attn_mask = (kv_mask == 0)  # (B, T_k), True=padding
        attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, T_k)
        attn_mask = attn_mask.expand(B, 1, T_q, T_k)  # (B, 1, T_q, T_k)
        attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).masked_fill(
            attn_mask, torch.finfo(query.dtype).min  # use dtype-safe large negative
        )
        cache_position = torch.arange(query.shape[1], device=query.device)

        hs = self.style_remover(
            hidden_states=query,
            key_value_states=kv,  # ← triggers cross-attention mode
            mask=attn_mask,
            cache_position=cache_position
        )[0]  # returns (attn_output, present_kv, position_bias, ...)

        hs = hs * q_mask.unsqueeze(-1)
        return hs


class LlamaAttentionStyleController(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 config: Optional[Any] = None
                 ):
        super().__init__()
        self.style_adder = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.style_remover = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def _init_weights(self, module):

        if isinstance(module, nn.MultiheadAttention):
            # Zero out all underlying parameter tensors explicitly
            nn.init.zeros_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

            nn.init.zeros_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

        # 3. Process remaining regular standalone child blocks normally
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def create_attention_inputs(query, key, value, query_mask=None, key_mask=None, value_mask=None):
        dtype = query.dtype
        kv = torch.cat([key, value], dim=1)
        if query_mask is not None and key_mask is not None and key_mask is not None:
            q_mask = query_mask
            kv_mask = torch.cat([key_mask, value_mask], dim=1)
        elif query_mask is None and key_mask is None and value_mask is None:
            q_mask = None
            kv_mask = None
        else:
            raise ValueError("query_mask, key_mask, value_mask should be either all None or all provided.")

        return query, kv, q_mask, kv_mask

    def add_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        # 1. PyTorch's MultiheadAttention uses an inverted boolean key mask convention:
        # True means "IGNORE/MASK OUT this token slot"
        # False means "KEEP/ATTEND to this token slot"
        key_padding_mask = ~kv_mask  # Inverts your True/False values

        # 2. PyTorch MHA expects batch_first=True to process [B, Seq, dim] inputs natively.
        # We pass query as Q, and kv as both K and V.
        # It returns a tuple: (attn_output, attn_output_weights)
        attn_output, _ = self.style_adder(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False  # Optimization: skips computing/returning weight matrices
        )

        # 3. Apply your query mask to zero out any trailing padding rows in the final output
        # This prevents noise from propagating down your network architecture.
        attn_output = attn_output * q_mask.unsqueeze(-1).to(attn_output.dtype)

        return attn_output

    def remove_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        # 1. PyTorch's MultiheadAttention uses an inverted boolean key mask convention:
        # True means "IGNORE/MASK OUT this token slot"
        # False means "KEEP/ATTEND to this token slot"
        key_padding_mask = ~kv_mask  # Inverts your True/False values

        # 2. PyTorch MHA expects batch_first=True to process [B, Seq, dim] inputs natively.
        # We pass query as Q, and kv as both K and V.
        # It returns a tuple: (attn_output, attn_output_weights)
        attn_output, _ = self.style_remover(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False  # Optimization: skips computing/returning weight matrices
        )

        # 3. Apply your query mask to zero out any trailing padding rows in the final output
        # This prevents noise from propagating down your network architecture.
        attn_output = attn_output * q_mask.unsqueeze(-1).to(attn_output.dtype)

        return attn_output


class QwenAttentionStyleController(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 config: Optional[Any] = None
                 ):
        super().__init__()
        self.config = config
        self.style_adder = Qwen3Attention(config=config, layer_idx=0)
        self.style_remover = Qwen3Attention(config=config, layer_idx=0)
        self._static_cache = DynamicCache()
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def _init_weights(self, module):

        classname = module.__class__.__name__

        # Force bypass HuggingFace's internal tracking skip flag if present
        if hasattr(module, "_is_hf_initialized"):
            module._is_hf_initialized = False

        # 1. Target all Linear Layers (Projections) using Xavier initialization
        if isinstance(module, nn.Linear) or "Linear" in classname:
            # Xavier uniform scales variance based on input and output dimension sums
            nn.init.zeros_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

        # 2. Target RMSNorm/LayerNorm configurations
        elif "RMSNorm" in classname or "LayerNorm" in classname:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def create_attention_inputs(query, key, value, query_mask=None, key_mask=None, value_mask=None):
        dtype = query.dtype
        kv = torch.cat([key, value], dim=1)
        if query_mask is not None and key_mask is not None and key_mask is not None:
            q_mask = query_mask
            kv_mask = torch.cat([key_mask, value_mask], dim=1)
        elif query_mask is None and key_mask is None and value_mask is None:
            q_mask = None
            kv_mask = None
        else:
            raise ValueError("query_mask, key_mask, value_mask should be either all None or all provided.")

        return query, kv, q_mask, kv_mask

    def add_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        B, Seq1, dim = query.shape
        _, Seq2, _ = kv.shape
        device = query.device
        dtype = query.dtype

        # 1. Clear static cache states to release reference tokens
        self._static_cache.key_cache.clear()
        self._static_cache.value_cache.clear()
        if hasattr(self._static_cache, "_seen_tokens"):
            self._static_cache._seen_tokens = 0

        # 2. Extract configuration features and project KV
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.style_adder.head_dim
        kv_shape = (B, Seq2, num_kv_heads, head_dim)

        k_states = self.style_adder.k_norm(self.style_adder.k_proj(kv).view(kv_shape)).transpose(1, 2)
        v_states = self.style_adder.v_proj(kv).view(kv_shape).transpose(1, 2)

        # 3. Inject new states into the purged persistent cache
        self._static_cache.update(k_states, v_states, layer_idx=0)

        # 4. Generate Position Embeddings (RoPE) exclusively for Query Sequence Length
        q_positions = torch.arange(Seq1, device=device).unsqueeze(0).repeat(B, 1)
        position_embeddings = self.rotary_emb(query, position_ids=q_positions)

        # 5. Construct 4D Mask Configuration matching the true internal dimension size
        # Total internal sequence length = Seq2 (from past_kv) + Seq1 (from query auto-append)
        total_kv_len = Seq2 + Seq1  # e.g., 45 + 2 = 47

        # Base cross attention boolean boundary block
        cross_bool_mask = q_mask.unsqueeze(2) & kv_mask.unsqueeze(1)  # [B, Seq1, Seq2]

        # Create a full 4D mask tracking total internal dimension matrix requirements
        # Base filled with min value penalty (-inf)
        attention_mask = torch.full((B, 1, Seq1, total_kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)

        # Map the valid cross-attention block to the first Seq2 columns
        # Elements corresponding to the query self-append space (the final Seq1 columns) remain masked out
        attention_mask[:, :, :, :Seq2] = torch.where(
            cross_bool_mask.unsqueeze(1),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype)
        )

        # 6. Execute forward calculation loop on target engine
        cache_position = torch.arange(Seq1, device=device, dtype=torch.long)

        hs = self.style_adder(
            hidden_states=query,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=self._static_cache,
            cache_position=cache_position
        )

        return hs[0]

    def remove_style(self, query, key, value, query_mask=None, key_mask=None, value_mask=None):
        query, kv, q_mask, kv_mask = self.create_attention_inputs(query, key, value, query_mask, key_mask, value_mask)

        B, Seq1, dim = query.shape
        _, Seq2, _ = kv.shape
        device = query.device
        dtype = query.dtype

        # 1. Clear static cache states to release reference tokens
        self._static_cache.key_cache.clear()
        self._static_cache.value_cache.clear()
        if hasattr(self._static_cache, "_seen_tokens"):
            self._static_cache._seen_tokens = 0

        # 2. Extract configuration features and project KV
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.style_remover.head_dim
        kv_shape = (B, Seq2, num_kv_heads, head_dim)

        k_states = self.style_remover.k_norm(self.style_remover.k_proj(kv).view(kv_shape)).transpose(1, 2)
        v_states = self.style_remover.v_proj(kv).view(kv_shape).transpose(1, 2)

        # 3. Inject new states into the purged persistent cache
        self._static_cache.update(k_states, v_states, layer_idx=0)

        # 4. Generate Position Embeddings (RoPE) exclusively for Query Sequence Length
        q_positions = torch.arange(Seq1, device=device).unsqueeze(0).repeat(B, 1)
        position_embeddings = self.rotary_emb(query, position_ids=q_positions)

        # 5. Construct 4D Mask Configuration matching the true internal dimension size
        # Total internal sequence length = Seq2 (from past_kv) + Seq1 (from query auto-append)
        total_kv_len = Seq2 + Seq1  # e.g., 45 + 2 = 47

        # Base cross attention boolean boundary block
        cross_bool_mask = q_mask.unsqueeze(2) & kv_mask.unsqueeze(1)  # [B, Seq1, Seq2]

        # Create a full 4D mask tracking total internal dimension matrix requirements
        # Base filled with min value penalty (-inf)
        attention_mask = torch.full((B, 1, Seq1, total_kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)

        # Map the valid cross-attention block to the first Seq2 columns
        # Elements corresponding to the query self-append space (the final Seq1 columns) remain masked out
        attention_mask[:, :, :, :Seq2] = torch.where(
            cross_bool_mask.unsqueeze(1),
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype)
        )

        # 6. Execute forward calculation loop on target engine
        cache_position = torch.arange(Seq1, device=device, dtype=torch.long)

        hs = self.style_remover(
            hidden_states=query,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=self._static_cache,
            cache_position=cache_position
        )

        return hs[0]