import importlib
import json
import os
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from peft import LoraConfig  # for saving adapter weights; for loading adapter weights
from peft import PeftModel
from peft import get_peft_model
from peft import get_peft_model_state_dict
from peft import set_peft_model_state_dict
from transformers import CONFIG_NAME
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.models.bart.modeling_bart import BartSdpaAttention


class GPUUtilizationTracker(threading.Thread):
    def __init__(self, gpu_idx, interval=0.1):
        super().__init__()
        # Map PyTorch local rank to physical GPU ID if CUDA_VISIBLE_DEVICES is set
        self.physical_gpu_id = str(gpu_idx)
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            try:
                ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
                if gpu_idx < len(ids):
                    self.physical_gpu_id = ids[gpu_idx]
            except Exception:
                pass

        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start paused

    def run(self):
        while not self.stop_event.is_set():
            if not self.pause_event.is_set():
                try:
                    util_out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                         "--id=" + self.physical_gpu_id],
                        encoding='utf-8'
                    )
                    self.samples.append(float(util_out.strip()))
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def get_avg_utilization(self):
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


def get_model_gflops(model, seq_len=1):
    # Estimate of GFLOPs per token (forward pass)
    params = sum(p.numel() for p in model.parameters())
    config = getattr(model, "config", None)
    d_model = getattr(config, "hidden_size", getattr(config, "d_model", 768))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "decoder_layers", 12))
    gflops = (2 * params + 4 * num_layers * seq_len * d_model) / 1e9
    return gflops


def get_last_token_index(attention_mask):
    assert len(attention_mask.shape) == 2
    # return attention_mask.sum(dim=-1, keepdim=False) - 1
    indices = torch.arange(attention_mask.size(-1), device=attention_mask.device)

    # Multiply indices by the mask to zero out the '0' positions
    # then take the argmax to find the highest index that was a '1'
    last_index = (indices * attention_mask).argmax(dim=-1)
    return last_index


def process_padding(padding, input_tensor, keep_first=False):
    if padding is not None:
        batch, length = input_tensor.size()

        # Identify columns that contain real data
        col_is_not_all_padding = ~torch.all(input_tensor == padding, dim=0, keepdim=True)
        cumsum = torch.cumsum(col_is_not_all_padding.to(torch.int), dim=1)
        valid_cols_mask = torch.logical_and(col_is_not_all_padding, cumsum > 0)

        # FIX: Force the mask to keep the first column for T5
        if keep_first:
            valid_cols_mask[:, 0] = True

        # Now input_tensor will actually retain that first <pad> token
        input_tensor = torch.masked_select(input_tensor, valid_cols_mask).view(batch, -1)

        # Create attention mask
        attention_mask = torch.ne(input_tensor, padding)

        # FIX: Ensure the attention mask also doesn't block that first token
        if keep_first:
            attention_mask[:, 0] = True

        assert valid_cols_mask.sum().item() == input_tensor.size(1)
    else:
        attention_mask = None

    return input_tensor, attention_mask


def process_super_batch(encoder_inputs, encoder_masks, pad_token_id, batch_size):
    """
    Pads all inputs to the same maximum length, concatenates them into a super-batch,
    and provides a way to split the encoded results back.

    Args:
        encoder_inputs (list[torch.Tensor]): List of input_ids tensors [batch, seq_len_i]
        encoder_masks (list[torch.Tensor]): List of attention mask tensors [batch, seq_len_i]
        pad_token_id (int): Padding token ID for the tokenizer
        batch_size (int): The original batch size

    Returns:
        super_input_ids (torch.Tensor): Concatenated input_ids [batch * num_inputs, max_seq_len]
        super_attention_mask (torch.Tensor): Concatenated attention masks [batch * num_inputs, max_seq_len]
        split_fn (callable): Function that takes the super-encoded output and returns a list of individual encoded tensors
    """
    # Pad all inputs to the same maximum length
    max_len = max(x.size(1) for x in encoder_inputs)
    padded_inputs = []
    padded_masks = []
    original_lengths = [x.size(1) for x in encoder_inputs]

    for x, mask in zip(encoder_inputs, encoder_masks):
        if x.size(1) < max_len:
            pad_len = max_len - x.size(1)
            padded_x = F.pad(x, (0, pad_len), value=pad_token_id)
            padded_m = F.pad(mask, (0, pad_len), value=0)
            padded_inputs.append(padded_x)
            padded_masks.append(padded_m)
        else:
            padded_inputs.append(x)
            padded_masks.append(mask)

    # Concatenate into super-batch
    super_input_ids = torch.cat(padded_inputs, dim=0)
    super_attention_mask = torch.cat(padded_masks, dim=0)

    def split_fn(super_enc_outputs):
        enc_outputs_split = torch.split(super_enc_outputs, batch_size, dim=0)
        return [enc[:, :orig_len, :] for enc, orig_len in zip(enc_outputs_split, original_lengths)]

    return super_input_ids, super_attention_mask, split_fn


def get_model_type(model_name):
    if model_name in ["gpt2", "llama3", "qwen3"]:
        return "decoder-only"
    elif model_name in ["bart", "t5"]:
        return "encoder-decoder"
    else:
        raise ValueError(f"Model name is not supported: {model_name}.")


def load_model_class(args):
    args.model_name = args.model_name.lower()
    if args.model_name == "gpt2":
        module_name = "grag.postfusion.dialogue_modules.gpt2_modules"
        model_class_prefix = "GPT2"
    elif args.model_name == "bart":
        module_name = "grag.postfusion.dialogue_modules.bart_modules"
        model_class_prefix = "BART"
    elif args.model_name == "llama3":
        module_name = "grag.postfusion.dialogue_modules.llama3_modules"
        model_class_prefix = "Llama3"
    elif args.model_name == "t5":
        module_name = "grag.postfusion.dialogue_modules.t5_modules"
        model_class_prefix = "T5"
    elif args.model_name == "qwen3":
        module_name = "grag.postfusion.dialogue_modules.qwen3_modules"
        model_class_prefix = "Qwen3"
    else:
        raise NotImplementedError

    # Step 1: Equivalent to 'import os'
    module_obj = importlib.import_module(module_name)

    if args.method == "GRAG":
        method = "GRAG"
    elif args.method in ["Hist", "TST"]:
        method = "SFT"
    else:
        raise NotImplementedError

    # Step 2: Equivalent to 'from os import path as attribute_alias'
    attribute_name = f"{model_class_prefix}PostFusionModel{method}"

    try:
        model_class = getattr(module_obj, attribute_name)
        print(f"Loading model class {model_class.__name__}")
        return model_class
    except AttributeError:
        raise ValueError(f"Model class cannot be loaded {attribute_name}")


class MaskedMean(nn.Module):
    """
    Computes masked mean over dimension 1.

    Inputs:
      A : (batch, seq_len, dim)
      B : (batch, seq_len), binary mask (1=keep, 0=ignore)

    Output:
      out : (batch, dim)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor, B: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        # ensure mask is float for multiplication
        B = B.to(dtype=A.dtype)

        # expand mask to match A's shape: (batch, seq_len, 1)
        mask_expanded = B.unsqueeze(-1)

        # apply mask
        masked_A = A * mask_expanded

        # sum over sequence
        summed = masked_A.sum(dim=1, keepdim=keepdim)  # (batch, 1, dim)

        # count of valid entries per batch
        counts = mask_expanded.sum(dim=1, keepdim=keepdim)  # (batch, 1, 1)

        # avoid divide-by-zero
        mean = summed / (counts + self.eps)

        return mean


class GumbelSigmoidGate(nn.Module):
    """Differentiable binary gate with straight-through estimator."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperature: float = 1.0, training: bool = True):
        if not training:
            # Inference → deterministic
            return (torch.sigmoid(logits) > 0.5).to(logits.dtype)

        # Sample Gumbel noise
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-9) + 1e-9)

        # Continuous relaxation
        y_soft = torch.sigmoid((logits + gumbel) / temperature)

        # Straight-through: hard 0/1 in forward, soft gradient in backward
        y_hard = (y_soft > 0.5).to(logits.dtype)
        gate = y_hard + (y_soft - y_soft.detach())
        return gate


class GumbelSoftmaxTopK(nn.Module):
    """Gumbel Softmax gate with TopK estimator."""

    def forward(self, logits: torch.Tensor, temperature: float = 1.0, topk: int = 500,
                hard: bool = True, training: bool = True):
        samples = []
        for i in range(10):  # take gumbel_softmax average over 10 times
            y_s = F.gumbel_softmax(logits, temperature, False)
            samples.append(y_s.unsqueeze(-1))
        y = torch.mean(torch.cat(samples, dim=-1), dim=-1)
        vals, idx = torch.topk(y, topk, dim=-1, sorted=False)
        y_ = torch.zeros_like(y, dtype=y.dtype).scatter(-1, idx, vals)
        y_[y_ == 0] = -float("inf")
        y__ = F.softmax(y_, dim=-1)

        if hard:
            shape = y__.size()
            _, ind = y__.max(dim=-1)
            y_hard = torch.zeros_like(y__, dtype=y.dtype).view(-1, shape[-1]).scatter(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            return y_hard - (y__).detach() + y__
        else:
            return y__


class SequenceReducer(nn.Module):
    """Self-attention based sequence reduction: [B, L, D] → [B, D]."""

    def __init__(self, emb_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=1, batch_first=True)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # Self-attend over sequence
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        # Weighted mean pooling via learned attention projection
        weights = torch.softmax(self.proj(attn_out).mean(dim=-1), dim=-1).unsqueeze(-1)
        x_reduced = (attn_out * weights).sum(dim=1)  # [B, D]
        return x_reduced


class BartSdpaSequenceReducer(nn.Module):
    """Self-attention based sequence reduction base on BartSdpaAttention: [B, L, D] → [B, D]."""

    def __init__(self, d_model, dropout=0.1, eps=1e-12):
        super().__init__()
        self.self_attn = BartSdpaAttention(embed_dim=d_model,
                                           num_heads=1,
                                           dropout=dropout)
        self.masked_mean = MaskedMean()

    def forward(self, hidden_states, attention_mask):
        input_shape = hidden_states.size()[:-1]
        inputs_embeds = hidden_states
        attention_mask_for_sdpa = _prepare_4d_attention_mask_for_sdpa(
            attention_mask,
            hidden_states.dtype,
            attention_mask.size(-1)
        )

        output = self.self_attn(hidden_states=hidden_states,
                                attention_mask=attention_mask_for_sdpa)[0]
        return self.masked_mean(output, attention_mask, keepdim=True)


def load_model_for_inference(model_class, args, bnb_config=None, **extra_model_kwargs):
    """
    args.model_checkpoint: Path to your saved checkpoint directory.
                           Works for BOTH regular and LoRA checkpoints.
    """
    checkpoint_dir = args.model_checkpoint

    # ── 1. Detect mode & resolve base model name ──────────────────────────────
    mode_file = os.path.join(checkpoint_dir, 'training_mode.json')
    adapter_config_file = os.path.join(checkpoint_dir, 'adapter_config.json')

    if os.path.exists(mode_file):
        with open(mode_file) as f:
            use_lora = json.load(f)['use_lora']
    else:
        use_lora = os.path.exists(adapter_config_file)

    if use_lora:
        with open(adapter_config_file) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config['base_model_name_or_path']
    else:
        # training_args = torch.load(os.path.join(checkpoint_dir, 'model_training_args.bin'))
        # base_model_name = training_args.model_name_or_path
        model = model_class.from_pretrained(checkpoint_dir, device_map="auto", quantization_config=bnb_config,
                                            torch_dtype=torch.bfloat16, **extra_model_kwargs)
        model.eval()
        return model

    print(f"[load_model] use_lora={use_lora}, base_model={base_model_name}")

    # ── 2. Load fine-tuned weights to inspect vocab size BEFORE building model ─
    weight_file = _resolve_weight_file(args, checkpoint_dir)
    assert weight_file, f"No checkpoint .pt file found in {checkpoint_dir}"
    print(f"[load_model] Loading weights from: {weight_file}")
    state_dict = torch.load(weight_file, map_location="cpu")

    # ── 3. Load base model on CPU first, resize embeddings, then move to device -
    # device_map="auto" causes newly resized embedding rows to land on CPU
    # while the rest of the model is on GPU, leading to device mismatch errors.
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        **extra_model_kwargs,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        # Quantized models must use device_map — handle separately below
        load_kwargs["device_map"] = "auto"

    model = model_class.from_pretrained(base_model_name, **load_kwargs)

    # Infer and apply vocab resize before any weight loading or device movement
    vocab_size = _infer_vocab_size(state_dict, model=model)
    # print(f"[load_model] Inferred vocab size from checkpoint: {vocab_size}")

    if vocab_size and model.get_input_embeddings().weight.shape[0] != vocab_size:
        # print(f"[load_model] Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)

    # Move the fully resized model to GPU in one shot (skip if quantized, already on device)
    if bnb_config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # ── 4. Apply LoRA structure then load weights ──────────────────────────────
    if use_lora:
        lora_cfg = LoraConfig.from_pretrained(checkpoint_dir)
        model = get_peft_model(model, lora_cfg)
        model.eval()

        result = set_peft_model_state_dict(model, state_dict)
        # print(f"[load_model] PEFT load result: {result}")
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_model] Missing keys : {missing}")
        if unexpected:
            print(f"[load_model] Unexpected keys: {unexpected}")

    model.eval()
    return model


def _infer_vocab_size(state_dict, model=None):
    """
    Infers vocab size from the state dict.

    If `model` is provided, uses get_input_embeddings() to find the exact
    parameter name — fully generalizes across all HF model architectures.

    Falls back to scanning for embedding-shaped tensors if model is not provided.
    """
    if model is not None:
        # get_input_embeddings() is the same HF interface used by resize_token_embeddings,
        # so this generalizes to any architecture without hardcoding parameter names.
        embedding_layer = model.get_input_embeddings()
        for name, module in model.named_modules():
            if module is embedding_layer:
                # Check both regular and LoRA modules_to_save prefixed versions
                candidates = [
                    f"{name}.weight",
                    f"modules_to_save.default.{name}.weight",
                ]
                for key in candidates:
                    if key in state_dict:
                        print(f"[_infer_vocab_size] Found vocab size via key: '{key}'")
                        return state_dict[key].shape[0]
                break  # Found the module but not in state_dict — fall through to scan

    # Fallback: scan for any 2D tensor whose name suggests an embedding or output projection.
    # This handles cases where the model is not available (e.g. called before model init).
    for key, tensor in state_dict.items():
        if tensor.dim() == 2 and any(k in key for k in ('embed', 'shared', 'lm_head')):
            print(f"[_infer_vocab_size] Found vocab size via fallback scan, key: '{key}'")
            return tensor.shape[0]

    return None


def _resolve_weight_file(args, checkpoint_dir):
    """
    Returns the .pt file to load, in this priority order:
      1. args.weight_checkpoint  (explicit override)
      2. Latest checkpoint_mymodel_<N>.pt in checkpoint_dir
      3. None  (caller decides what to do)
    """
    if hasattr(args, 'weight_checkpoint') and args.weight_checkpoint:
        return args.weight_checkpoint

    import glob
    pattern = os.path.join(checkpoint_dir, 'checkpoint_mymodel_*.pt')
    files = sorted(glob.glob(pattern))  # ModelCheckpoint names are sortable by epoch
    return files[-1] if files else None


def convert_to_lora_model(model):
    # 1. Apply LoRA to the base model part of your custom class
    custom_modules = getattr(model, "_custom_modules", [])
    for name, module in model.named_modules():
        if any(custom_name in name for custom_name in custom_modules):
            # Force the custom modules and all their sub-layers to bfloat16
            module.to(torch.bfloat16)

            # Ensure any parameters they hold require gradients
            for param in module.parameters():
                param.requires_grad = True

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        modules_to_save=custom_modules
    )

    # 2. Wrap your entire custom model
    # modules_to_save will ensure these layers are trained fully (not as LoRA)
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def save_checkpoint(trainer, model, tokenizer, args):
    log_dir = args.log_dir
    raw_model = getattr(model, 'module', model)
    is_lora = isinstance(raw_model, PeftModel)

    # 1. ModelCheckpoint saves the right state dict
    # For LoRA: save only adapter weights (small). For regular: save full weights.
    def get_state_dict(m):
        if isinstance(m, PeftModel):
            # Only save trainable adapter weights + modules_to_save layers
            return get_peft_model_state_dict(m)
        return m.state_dict()

    class SmartCheckpointHandler:
        """Wraps model to expose the correct state_dict to ModelCheckpoint."""

        def __init__(self, m):
            self.m = m

        def state_dict(self):
            return get_state_dict(self.m)

        def load_state_dict(self, sd):
            if isinstance(self.m, PeftModel):
                set_peft_model_state_dict(self.m, sd)
            else:
                self.m.load_state_dict(sd)

    smart_model = SmartCheckpointHandler(raw_model)
    checkpoint_handler = ModelCheckpoint(
        log_dir, 'checkpoint', n_saved=3, require_empty=False
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': smart_model}
    )

    # 2. Save metadata (config, tokenizer, args) — no large weights here
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_metadata(engine):
        if is_lora:
            # Saves adapter_config.json (no base weights)
            raw_model.save_pretrained(log_dir)
        else:
            raw_model.config.to_json_file(os.path.join(log_dir, CONFIG_NAME))

        tokenizer.save_pretrained(log_dir)
        torch.save(args, os.path.join(log_dir, 'model_training_args.bin'))

        # Save a flag so inference knows what mode was used
        mode = {'use_lora': is_lora}
        with open(os.path.join(log_dir, 'training_mode.json'), 'w') as f:
            json.dump(mode, f)

    return checkpoint_handler
