import random
import re
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import set_seed

from dataset_parser import DatasetCreatorConvAI2
from dataset_parser import DatasetCreatorEmpathetic
from dataset_parser import DatasetCreatorPFG
from dataset_parser import DatasetCreatorSPC
from grag.common_evaluations.metrics import calculate_sample_bpc_bpb
from grag.common_evaluations.metrics import print_ppl_evaluation_results
from grag.common_evaluations.utils import save_individual_metrics


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


def run():
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
    parser.add_argument("--seed", type=int, required=False, default=None, help="global seed")
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
    seed = args.seed

    if seed is not None:
        print("Setting global seed: ", seed)
        set_global_seed(seed)
    else:
        print("Seed is None")

    # Load the Llama model and tokenizer
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your chosen Llama model
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("Device: ", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with torch.no_grad():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)

    model.to(device)
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
    output_data = []

    with torch.no_grad():
        ppl_avg = []
        all_bpc = []
        all_bpb = []
        total_tokens = 0
        total_chars = 0
        total_bytes = 0
        individual_metrics = []
        total_loss = 0.0

        for j, conversation in enumerate(tqdm(test_dataset, miniters=5, mininterval=20)):
            for i, ref_ut in enumerate(conversation["utterances"]):
                if ref_ut.startswith("[User]") or i == 0:
                    continue
                ref_ut = ref_ut.replace("[Agent] ", "")
                shots = random.choices(train_dataset, k=n_shot)

                # generate and tokenize prompt
                prompt_messages = dataset_parser.generate_prompt_messages(i, conversation, shots)
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                agent_prefix = "[Agent] "

                prompt_enc = tokenizer(prompt_text, return_tensors="pt")
                agent_enc = tokenizer(agent_prefix, return_tensors="pt", add_special_tokens=False)
                reply_enc = tokenizer(ref_ut, return_tensors="pt", add_special_tokens=False)

                input_ids = torch.cat(
                    [
                        prompt_enc.input_ids,
                        agent_enc.input_ids,
                        reply_enc.input_ids,
                    ],
                    dim=1
                ).to(device)

                prompt_len = prompt_enc.input_ids.shape[1]
                agent_len = agent_enc.input_ids.shape[1]
                prompt_length = prompt_len + agent_len

                # get logits
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                lm_logits = model(input_ids).logits
                lm_labels = input_ids.clone()
                lm_labels[0, :prompt_length] = -100
                lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
                lm_loss = loss_fct(lm_logits_flat_shifted, lm_labels_flat_shifted)

                # ppl
                ppl = torch.exp(lm_loss).item()
                ppl_avg.append(ppl)

                # Calculate BPC/BPB
                batch_bpc, batch_bpb, stats = calculate_sample_bpc_bpb(
                    lm_logits,
                    lm_labels,
                    tokenizer
                )

                all_bpc.extend(batch_bpc)
                all_bpb.extend(batch_bpb)
                total_tokens += stats['total_tokens']
                total_chars += stats['total_chars']
                total_bytes += stats['total_bytes']

                # Individual metrics
                for i in range(len(batch_bpc)):
                    individual_metrics.append({
                        "index": len(individual_metrics),
                        "ppl": ppl,
                        "bpc": batch_bpc[i],
                        "bpb": batch_bpb[i]
                    })

        # Print results
        print_ppl_evaluation_results(ppl_avg, all_bpc, all_bpb, total_tokens, total_chars, total_bytes)

        if args.metrics_output_file is not None:
            save_individual_metrics(args, individual_metrics)


if __name__ == "__main__":
    import time

    start = time.time()
    run()
    end = time.time()
    print("Total time needed:", end - start)
