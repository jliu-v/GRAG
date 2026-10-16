import importlib

from grag.postfusion.dialogue_modules.model_utils import GPUUtilizationTracker
from grag.postfusion.dialogue_modules.model_utils import convert_to_lora_model
from grag.postfusion.dialogue_modules.model_utils import get_last_token_index
from grag.postfusion.dialogue_modules.model_utils import get_model_gflops
from grag.postfusion.dialogue_modules.model_utils import get_model_type
from grag.postfusion.dialogue_modules.model_utils import load_model_for_inference
from grag.postfusion.dialogue_modules.model_utils import process_padding
from grag.postfusion.dialogue_modules.model_utils import save_checkpoint

_ = get_last_token_index, process_padding, save_checkpoint, convert_to_lora_model, load_model_for_inference, get_model_type, get_model_gflops, GPUUtilizationTracker


def load_model_class(args):
    args.model_name = args.model_name.lower()
    if args.model_name == "gpt2":
        module_name = "grag.prefusion.dialogue_modules.gpt2_modules"
        model_class_prefix = "GPT2"
    elif args.model_name == "bart":
        module_name = "grag.prefusion.dialogue_modules.bart_modules"
        model_class_prefix = "BART"
    elif args.model_name == "llama3":
        module_name = "grag.prefusion.dialogue_modules.llama3_modules"
        model_class_prefix = "Llama3"
    elif args.model_name == "t5":
        module_name = "grag.prefusion.dialogue_modules.t5_modules"
        model_class_prefix = "T5"
    elif args.model_name == "qwen3":
        module_name = "grag.prefusion.dialogue_modules.qwen3_modules"
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
    attribute_name = f"{model_class_prefix}PreFusionModel{method}"

    try:
        model_class = getattr(module_obj, attribute_name)
        print(f"Loading model class {model_class.__name__}")
        return model_class
    except AttributeError:
        raise ValueError(f"Model class cannot be loaded {attribute_name}")
