import logging
import os
import random
from argparse import ArgumentParser
from pprint import pformat

import psutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

from data_utils import BART_MODEL_INPUTS
from data_utils import DatasetCreatorConvAI2
from data_utils import DatasetCreatorEmpathetic
from data_utils import DatasetCreatorPFG
from data_utils import DatasetCreatorSPC
from data_utils import GPT2_MODEL_INPUTS
from data_utils import SPECIAL_TOKENS
from grag.common_evaluations.utils import log_inference_outputs
from grag.common_evaluations.utils import maybe_write_inference_outputs
from grag.common_evaluations.utils import save_individual_metrics
from grag.postfusion.dialogue_modules.model_utils import GPUUtilizationTracker
from grag.postfusion.dialogue_modules.model_utils import get_model_gflops
from grag.postfusion.dialogue_modules.model_utils import get_model_type
from grag.postfusion.dialogue_modules.model_utils import load_model_class
from grag.postfusion.dialogue_modules.model_utils import load_model_for_inference
from inference_test import sample_sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__file__)


def run():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--task", type=str, required=True, choices=["pfg", "empathetic", "spc", "convai2"],
                        help="{pfg, empathetic, spc, convai2}")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="{gpt2, bart, llama3, t5, qwen3}")
    parser.add_argument("--test_dataset_path", type=str, default="./data/persuasionforgood/sample_test.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--test_dataset_cache", type=str, default='./cache/persuasionforgood_sample_test.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--use_quantization", action='store_true',
                        help="Load model is 4-bit quantization to save GPU memory.")
    parser.add_argument("--method", type=str, default="GRAG",
                        help="Method of model architecture and implementation. Options: ['GRAG', 'Hist', 'TST']")
    parser.add_argument("--inference_output_file", type=str, required=False, default=None,
                        help="File path for generated output.")
    parser.add_argument("--metrics_output_file", type=str, required=False, default=None,
                        help="File path for individual metrics output.")
    parser.add_argument("--inference_output_size", type=int, required=False, default=None,
                        help="Number of examples to log in generated output.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    assert args.test_batch_size == 1

    if args.seed is not None:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Load dataset_creator
    if args.task == "pfg":
        dataset_creator_cls = DatasetCreatorPFG
    elif args.task == "empathetic":
        dataset_creator_cls = DatasetCreatorEmpathetic
    elif args.task == "spc":
        dataset_creator_cls = DatasetCreatorSPC
    elif args.task == "convai2":
        dataset_creator_cls = DatasetCreatorConvAI2
    else:
        raise ValueError(f"Task not recognized: {args.task}")

    model_name = args.model_name.lower()

    logger.info("Get model and tokenizer")

    model_class = load_model_class(args)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=["MultiheadAttention"] + getattr(model_class, "_custom_modules", []),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.use_quantization and "cuda" in args.device else None

    extra_model_kwargs = {}

    model = load_model_for_inference(model_class, args, bnb_config=bnb_config, **extra_model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    logger.info("Prepare datasets")

    dataset_creator = dataset_creator_cls(tokenizer=tokenizer,
                                          test_dataset_path=args.test_dataset_path,
                                          test_dataset_cache=args.test_dataset_cache)
    dataset_creator.create_data_loader(args, generation_mode=True, selected_partitions=["test"])
    dataloaders = dataset_creator.dataset_loader

    model_type = get_model_type(model_name)
    if model_type == "decoder-only":
        model_input_names = GPT2_MODEL_INPUTS
    elif model_type == "encoder-decoder":
        model_input_names = BART_MODEL_INPUTS
    else:
        raise ValueError(f"Model model_type is not supported: {model_type}.")

    test_loader = dataloaders["test"]

    with torch.no_grad():
        pass_num = 0
        test_pass_list = []
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

        for test_data_index, batch in enumerate(tqdm(test_loader, miniters=5, mininterval=20)):
            if args.inference_output_size is not None and len(individual_metrics) >= args.inference_output_size:
                logger.info(f"Early stopping as inference_output_size {args.inference_output_size} reached.")
                break

            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            model_inputs = dict(zip(model_input_names, batch))
            model_inputs.update({
                "method": args.method
            })
            model_inputs["lm_labels"] = None
            response = model_inputs["response"]
            mask = (response != tokenizer.pad_token_id)
            response = response[mask]

            if len(response.tolist()) == 0:  # empty response, skip
                pass_num += 1
                test_pass_list.append(test_data_index)
                continue

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

            out_ids = sample_sequence(model=model,
                                      model_inputs=model_inputs,
                                      tokenizer=tokenizer,
                                      args=args,
                                      current_output=None)

            if torch.cuda.is_available():
                end_event.record()
                gpu_tracker.pause()
                gpu_tracker.stop()
                gpu_tracker.join()

            end_time = time.time()

            # Input length (number of tokens the first generated token attends to)
            # For simplicity, we use the length of the main input
            input_ids = model_inputs.get("history", model_inputs.get("input_ids"))
            input_len = input_ids.shape[1] if input_ids is not None else 0
            num_tokens = len(out_ids)

            # Average sequence length during generation
            avg_seq_len = input_len + (num_tokens / 2)
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
            num_tokens = len(out_ids)
            if num_tokens > 0:
                total_duration += duration
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_duration = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                    total_gpu_duration += gpu_duration
                total_tokens += num_tokens
                total_gflops += model_gflops_per_token * num_tokens
                latency_samples += 1

            special_tokens_list = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
            special_tokens_list += [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
            special_tokens = set(special_tokens_list)
            role_tokens = set(special_tokens_list[:2])
            role_sp_tokens = set(special_tokens_list[2:4])

            gold_reply = response

            for special_token in special_tokens_list:
                out_ids = [value for value in out_ids if value != special_token]

            history = [x for x in model_inputs["history"][0].tolist() if
                       x not in (special_tokens - role_tokens)]
            persona = [x for x in model_inputs["persona"][0].tolist() if x not in (special_tokens - role_sp_tokens)]
            generic_response = [x for x in model_inputs["generic_response"][0].tolist() if
                                x not in (special_tokens - role_sp_tokens)]

            # Decode from ids to text
            gold_reply = tokenizer.decode(gold_reply.tolist(), skip_special_tokens=True)
            pred_reply = tokenizer.decode(out_ids, skip_special_tokens=True)
            persona = tokenizer.decode(persona, skip_special_tokens=False)
            history = tokenizer.decode(history, skip_special_tokens=False)
            generic_response = tokenizer.decode(generic_response, skip_special_tokens=False)

            log_inference_outputs(gold_reply, pred_reply, persona, history, generic_response, args)

            # Individual metrics
            individual_metrics.append({
                "index": test_data_index,
                "ram_usage_mb": cur_ram,
                "vram_usage_mb": cur_vram,
                "latency_per_token": duration / num_tokens if num_tokens > 0 else 0
            })

        # if inference output is not written, then write it
        maybe_write_inference_outputs(args)

        print('test pass list: ', test_pass_list, 'len: ', pass_num)
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


if __name__ == "__main__":
    import time

    start = time.time()
    run()
    end = time.time()
    print("Total time needed:", end - start)
