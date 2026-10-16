import logging
import os
import random
import re
from argparse import ArgumentParser

import numpy as np
import psutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import set_seed

from dataset_parser import DatasetCreatorConvAI2
from dataset_parser import DatasetCreatorEmpathetic
from dataset_parser import DatasetCreatorPFG
from dataset_parser import DatasetCreatorSPC
from grag.common_evaluations.utils import log_inference_outputs
from grag.common_evaluations.utils import maybe_write_inference_outputs
from grag.common_evaluations.utils import save_individual_metrics
from grag.postfusion.dialogue_modules.model_utils import GPUUtilizationTracker
from grag.postfusion.dialogue_modules.model_utils import get_model_gflops

logger = logging.getLogger(__file__)


def process_generated_text(generated_text):
    match = re.search(pattern="\[Agent\]", string=generated_text)
    msg = None
    if match:
        processed_text = generated_text[match.start():]
    else:
        msg = "Didn't find pattern [Agent]"
        processed_text = generated_text
        if "\n" in processed_text:
            processed_text = processed_text.split("\n")[-1]
        processed_text = f"[Agent] {processed_text}"

    # should_regenerate = msg is not None
    should_regenerate = False
    return processed_text, should_regenerate, msg


def set_global_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="name of pre-trained language model from HuggingFace")
    parser.add_argument("--task", type=str, required=True, choices=["pfg", "empathetic", "spc", "convai2"],
                        help="{pfg, empathetic, spc, convai2}")
    parser.add_argument("--context_mode", type=str, default="persona-history",
                        help="{persona-history, persona-tst}")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path or url of the dataset.")
    parser.add_argument("--test_dataset_path", type=str, required=True,
                        help="Path or url of the dataset.")
    parser.add_argument("--flag", type=str, default="", help="Assign the name of the folder")
    parser.add_argument("--n_shot", type=int, default=1, help="Number of examples for in-context reference.")
    parser.add_argument("--output_file", type=str, required=True, help="File path for generated output.")
    parser.add_argument("--seed", type=int, required=False, default=None, help="global seed")
    parser.add_argument("--inference_output_size", type=int, required=False, default=None,
                        help="Number of examples to log in generated output.")
    parser.add_argument("--metrics_output_file", type=str, required=False, default=None,
                        help="File path for individual metrics output.")

    args = parser.parse_args()

    task = args.task
    model_name = args.model_name
    context_mode = args.context_mode
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    flag = args.flag
    n_shot = args.n_shot
    output_file = args.output_file
    seed = args.seed

    if seed is not None:
        print("Setting global seed: ", seed)
        set_global_seed(seed)
    else:
        print("Seed is None")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with torch.no_grad():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)

    # Load dataset_parser
    if task == "pfg":
        dataset_parser_cls = DatasetCreatorPFG
    elif task == "empathetic":
        dataset_parser_cls = DatasetCreatorEmpathetic
    elif task == "spc":
        dataset_parser_cls = DatasetCreatorSPC
    elif task == "convai2":
        dataset_parser_cls = DatasetCreatorConvAI2
    else:
        raise ValueError(f"Task not recognized: {task}")

    dataset_parser = dataset_parser_cls(train_dataset_path=train_dataset_path,
                                        test_dataset_path=test_dataset_path,
                                        context_mode=context_mode)
    dataset_parser.create_dataset()
    train_dataset, test_dataset = dataset_parser.processed_dataset["train"], dataset_parser.processed_dataset["test"]

    global _inference_outputs

    with torch.no_grad():
        total_test = 0
        individual_metrics = []

        # Memory and latency tracking
        process = psutil.Process(os.getpid())
        peak_ram = 0
        total_ram = 0
        peak_vram = 0
        total_vram = 0
        peak_vram_reserved = 0
        total_vram_reserved = 0
        total_duration = 0
        total_gpu_duration = 0
        total_tokens = 0
        total_gpu_util = 0
        total_gflops = 0
        mem_samples = 0
        latency_samples = 0

        for j, conversation in enumerate(tqdm(test_dataset, miniters=5, mininterval=20)):
            if args.inference_output_size is not None and len(individual_metrics) >= args.inference_output_size:
                print(f"Early stopping as inference_output_size {args.inference_output_size} reached.")
                break

            for i, ref_ut in enumerate(conversation["utterances"]):
                if args.inference_output_size is not None and len(individual_metrics) >= args.inference_output_size:
                    break

                if ref_ut.startswith("[User]") or i == 0:
                    continue
                ref_ut = ref_ut.replace("[Agent] ", "")

                if n_shot > 0:
                    shots = random.choices(train_dataset, k=n_shot)
                else:
                    shots = None

                # generate prompt
                prompt_messages = dataset_parser.generate_prompt_messages(i, conversation, shots)

                # Tokenize the input
                input_ids = tokenizer.apply_chat_template(prompt_messages, return_dict=True, tokenize=True,
                                                          add_generation_prompt=True,
                                                          return_tensors="pt").to(device)
                prompt_length = input_ids.data["input_ids"].shape[1]

                run_generation = True
                retries = 0

                while run_generation:
                    retries += 1

                    import time
                    start_time = time.time()

                    # GPU latency and utilization measurement
                    start_event = None
                    end_event = None
                    gpu_tracker = None
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        gpu_idx = torch.cuda.current_device()
                        gpu_tracker = GPUUtilizationTracker(gpu_idx)
                        gpu_tracker.start()
                        gpu_tracker.resume()

                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()

                    # Generate text
                    output = model.generate(**input_ids.data,
                                            num_return_sequences=1,
                                            pad_token_id=tokenizer.pad_token_type_id,
                                            max_new_tokens=2000,
                                            repetition_penalty=1.1)

                    if torch.cuda.is_available():
                        end_event.record()
                        gpu_tracker.pause()
                        gpu_tracker.stop()
                        gpu_tracker.join()

                    end_time = time.time()

                    num_tokens = output.shape[1] - prompt_length
                    # Average sequence length during generation
                    avg_seq_len = prompt_length + (num_tokens / 2)
                    model_gflops_per_token = get_model_gflops(model, seq_len=avg_seq_len)

                    # Memory sampling during/immediately after inference
                    cur_ram = process.memory_info().rss / (1024 * 1024)  # MB
                    peak_ram = max(peak_ram, cur_ram)
                    total_ram += cur_ram

                    cur_vram = 0
                    cur_vram_reserved = 0
                    if torch.cuda.is_available():
                        cur_vram = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / (1024 * 1024))
                        cur_vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                        peak_vram_reserved = max(peak_vram_reserved, cur_vram_reserved)
                    total_vram += cur_vram
                    total_vram_reserved += cur_vram_reserved

                    if torch.cuda.is_available() and gpu_tracker:
                        cur_gpu_util = gpu_tracker.get_avg_utilization()
                        total_gpu_util += cur_gpu_util

                    mem_samples += 1

                    # Latency sampling
                    duration = end_time - start_time
                    num_tokens = output.shape[1] - prompt_length
                    if num_tokens > 0:
                        total_duration += duration
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            gpu_duration = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                            total_gpu_duration += gpu_duration
                        total_tokens += num_tokens
                        total_gflops += model_gflops_per_token * num_tokens
                        latency_samples += 1

                    # Decode the output
                    generated_response = tokenizer.decode(output[0][prompt_length:], attention_mask=None,
                                                          skip_special_tokens=True)
                    # Process generated text
                    if tokenizer.eos_token_id in output[0][prompt_length:]:
                        pred_response = generated_response.strip().replace("[Agent] ", "").replace("[Agent]", "")
                        should_regenerate = False
                    else:
                        should_regenerate = True

                    if not should_regenerate:
                        total_test += 1

                        persona = convert_persona(conversation["context"])
                        history = convert_history(conversation["utterances"][:i])
                        generic_response = conversation["generic_utterances"][i].replace("[Agent] ", "")
                        pred_reply = pred_response
                        gold_reply = ref_ut

                        log_inference_outputs(gold_reply, pred_reply, persona, history, generic_response, args)

                        # Individual metrics
                        individual_metrics.append({
                            "index": len(individual_metrics),
                            "ram_usage_mb": cur_ram,
                            "vram_usage_mb": cur_vram,
                            "latency_per_token": duration / num_tokens if num_tokens > 0 else 0
                        })

                    run_generation = should_regenerate and retries <= 5

        # if inference output is not written, then write it
        maybe_write_inference_outputs(args)

        print("=" * 60)
        print("INFERENCE PERFORMANCE RESULTS")
        print("=" * 60)
        # Final memory and latency metrics
        avg_ram = total_ram / mem_samples if mem_samples > 0 else 0
        avg_vram = total_vram / mem_samples if mem_samples > 0 else 0
        avg_vram_reserved = total_vram_reserved / mem_samples if mem_samples > 0 else 0
        avg_latency = (total_duration / total_tokens * 1000) if total_tokens > 0 else 0
        avg_gpu_latency = (total_gpu_duration / total_tokens * 1000) if total_tokens > 0 else 0
        avg_gpu_util = total_gpu_util / mem_samples if mem_samples > 0 else 0
        avg_gflops = total_gflops / total_tokens if total_tokens > 0 else 0
        avg_tokens = total_tokens / latency_samples if latency_samples > 0 else 0
        print(f"Peak RAM Usage: {peak_ram:.2f} MB")
        print(f"Average RAM Usage: {avg_ram:.2f} MB")
        if torch.cuda.is_available():
            print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
            print(f"Average VRAM Usage: {avg_vram:.2f} MB")
            print(f"Peak VRAM Reserved: {peak_vram_reserved:.2f} MB")
            print(f"Average VRAM Reserved: {avg_vram_reserved:.2f} MB")
            print(f"Average GPU Utilization: {avg_gpu_util:.2f} %")
        print(f"Average Latency per Token: {avg_latency:.4f} ms")
        if torch.cuda.is_available():
            print(f"Average GPU Latency per Token: {avg_gpu_latency:.4f} ms")
        print(f"Average GFLOPs per Token: {avg_gflops:.4f}")
        print(f"Average Tokens per Response: {avg_tokens:.2f}")
        print("=" * 60)

        if args.metrics_output_file is not None:
            save_individual_metrics(args, individual_metrics)


def convert_history(history):
    # Convert history - merge history and user_prompt
    history_parts = []

    # Process existing history
    for h in history:
        # Replace [User] with <user> and [Agent] with <agent>
        h = re.sub(r'\[User\]', '<user>', h, flags=re.IGNORECASE)
        h = re.sub(r'\[Agent\]', '<agent>', h, flags=re.IGNORECASE)
        history_parts.append(h.strip())

    return "".join(history_parts)


def convert_persona(persona):
    """Convert persona to the target format based on its type"""
    persona_parts = []

    if persona is None:
        return ""

    # Case 1: Original format - dict with "User" and "Agent" lists
    if isinstance(persona, dict) and ("User" in persona or "Agent" in persona):
        # Check if values are lists (original format)
        if isinstance(persona.get("User"), list) or isinstance(persona.get("Agent"), list):
            # Add user personas if they exist
            user_personas = persona.get("User", [])
            for p in user_personas:
                persona_parts.append(f"<user_persona>{p}")

            # Add agent personas if they exist
            agent_personas = persona.get("Agent", [])
            for p in agent_personas:
                persona_parts.append(f"<agent_persona>{p}")

        # Check if values are dicts (File 1 format)
        elif isinstance(persona.get("User"), dict) or isinstance(persona.get("Agent"), dict):
            # Add agent persona if it exists
            if "Agent" in persona and persona["Agent"]:
                agent_dict = persona["Agent"]
                dict_str = "{" + ", ".join(f"{k}: {v}" for k, v in agent_dict.items()) + "}"
                persona_parts.append(f"<agent_persona> {dict_str}")

            # Add user persona if it exists
            if "User" in persona and persona["User"]:
                user_dict = persona["User"]
                dict_str = "{" + ", ".join(f"{k}: {v}" for k, v in user_dict.items()) + "}"
                persona_parts.append(f"<user_persona> {dict_str}")

    # Case 2: File 2 format - simple string
    elif isinstance(persona, str):
        persona_parts.append(f"<agent_persona> {persona}")

    return "".join(persona_parts)


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print("Total time needed:", end - start)
