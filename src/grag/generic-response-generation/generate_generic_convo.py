import datetime
import json
import os
import re
from argparse import ArgumentParser

import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset_parser import DatasetParserConvAI2
from dataset_parser import DatasetParserEmpatheticDialog
from dataset_parser import DatasetParserPFG
from dataset_parser import DatasetParserSPC
from transformers import set_seed

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
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="name of output directory")
    parser.add_argument("--seed", type=int, required=False, default=None, help="global seed")
    parser.add_argument("--sample_mode", action="store_true", default=False)


    args = parser.parse_args()

    task = args.task
    sample_mode = args.sample_mode
    model_name = args.model_name
    output_dir = args.output_dir
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
        dataset_parser_cls = DatasetParserPFG
    elif task == "empathetic":
        dataset_parser_cls = DatasetParserEmpatheticDialog
    elif task == "spc":
        dataset_parser_cls = DatasetParserSPC
    elif task == "convai2":
        dataset_parser_cls = DatasetParserConvAI2
    else:
        raise ValueError(f"Task not recognized: {task}")

    dataset_parser = dataset_parser_cls(sample_mode=sample_mode)
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = dataset_parser.process_dataset()

    for dataset_type, sub_dataset in dataset.items():

        if sample_mode:
            sub_dataset = sub_dataset[:min(4, len(sub_dataset))]

        output_data_partition_name = dataset_parser.output_data_partition_name[dataset_type]
        output_file = f'{output_dir}/{output_data_partition_name}.json'
        job_split = os.environ.get('JOB_SPLIT')

        if job_split is not None:
            split_idx = int(len(sub_dataset) / 2)
            if job_split == "0":
                sub_dataset = sub_dataset[:split_idx]
            elif job_split == "1":
                sub_dataset = sub_dataset[split_idx:]
            else:
                raise ValueError("Job split must be '0' or '1'")
            output_file = f'{output_dir}/{output_data_partition_name}_{job_split}.json'

        if os.path.exists(output_file):
            print(f"Skipping generating generic dataset {dataset_type}. Output file already exists: {output_file}")
            continue

        print(f"Generating generic dataset: {dataset_type} with {len(sub_dataset)} conversations.")
        output_dataset = []

        for i, conversation in enumerate(sub_dataset):

            if (i + 1) % 50 == 0:
                with open(output_file, 'w') as f:
                    json.dump(output_dataset, f, indent=2)
                print(f"Processing progress: {i + 1}/{len(sub_dataset)} at {datetime.datetime.now()}")

            new_utterances = []

            for i, ref_ut in enumerate(conversation["utterances"]):
                if ref_ut.startswith("[User]") or i == 0:
                    new_utterances.append(conversation["utterances"][i])
                    continue

                # generate prompt
                prompt_messages = dataset_parser.generate_prompt(conversation, i)

                # Tokenize the input
                # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                input_ids = tokenizer.apply_chat_template(prompt_messages, return_dict=True, tokenize=True,
                                                          add_generation_prompt=True,
                                                          return_tensors="pt").to(device)
                prompt_length = input_ids.data["input_ids"].shape[1]

                run_generation = True
                retries = 0
                generated_text = ""
                utterances = []
                msg = None
                while run_generation:
                    retries += 1

                    # Generate text
                    output = model.generate(**input_ids.data, pad_token_id=tokenizer.pad_token_type_id,
                                            max_new_tokens=2000, repetition_penalty=1.1)

                    # Decode the output
                    generated_text = tokenizer.decode(output[0][prompt_length:], attention_mask=None,
                                                      skip_special_tokens=True)
                    # Process generated text
                    if not generated_text.startswith("[Agent]"):
                        generated_text, should_regenerate, msg = process_generated_text(generated_text)
                        if msg is not None:
                            msg += f": verify turn {i}"
                            conversation["warn"] = msg
                    else:
                        should_regenerate, msg = False, None

                    run_generation = should_regenerate and retries <= 5

                new_utterances.append(generated_text)

            conversation["generic_utterances"] = new_utterances
            output_dataset.append(conversation)

        with open(output_file, 'w') as f:
            json.dump(output_dataset, f, indent=2)


if __name__ == "__main__":
    main()
