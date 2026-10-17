import json
import logging
import os
import random
import warnings
from argparse import ArgumentParser
from pprint import pformat

import torch
import torch.nn.functional as F
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
from data_utils import add_special_tokens_
from grag.postfusion.dialogue_modules.model_utils import get_model_type
from grag.postfusion.dialogue_modules.model_utils import load_model_class

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__file__)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(model, model_inputs, tokenizer, args, current_output=None):
    model_type = get_model_type(args.model_name)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    agent = [special_tokens_ids[0]]
    model_inputs["generation_mode"] = True
    model_inputs["max_length"] = args.max_length

    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        if model_type == "decoder-only":
            if i == 0:
                model_inputs["initial_input_ids"] = model_inputs["input_ids"]
                model_inputs["initial_input_tti"] = model_inputs["input_tti"]

            initial_input_ids = model_inputs["initial_input_ids"]
            initial_input_tti = model_inputs["initial_input_tti"]

            current_input_ids = torch.cat(
                [initial_input_ids[0], torch.tensor(current_output).type_as(initial_input_ids)],
                dim=-1).unsqueeze(0)
            current_input_tti = torch.cat(
                [initial_input_tti[0], torch.tensor(agent * len(current_output)).type_as(initial_input_tti)],
                dim=-1).unsqueeze(0)
            model_inputs["input_ids"] = current_input_ids
            model_inputs["input_tti"] = current_input_tti

        elif model_type == "encoder-decoder":
            if i == 0:
                model_inputs["initial_decoder_input"] = model_inputs["decoder_input"]

            initial_decoder_input = model_inputs["initial_decoder_input"]
            current_decoder_input = torch.cat([initial_decoder_input[0],
                                               torch.tensor(current_output).type_as(initial_decoder_input)],
                                              dim=-1).unsqueeze(0)
            model_inputs["decoder_input"] = current_decoder_input

        else:
            raise ValueError(f"Model model_type is not supported: {model_type}.")

        output = model(training=False,
                       padding=tokenizer.pad_token_id,
                       tokenizer=tokenizer,
                       **model_inputs)
        logits = output["lm_logits"]

        logits = logits[0, -1, :] / args.temperature  # size [50262]
        logits1 = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits1, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)

        if i < args.min_length and prev.item() in tokenizer.all_special_ids:
            n_try = 0
            while prev.item() in tokenizer.all_special_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                if n_try > 50:
                    warnings.warn(f"Warning: model generating special token for {n_try} retries.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)
                n_try += 1
        if prev.item() in tokenizer.all_special_ids:
            break
        current_output.append(prev.item())

    return current_output


def run():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--task", type=str, required=True, choices=["pfg", "empathetic", "spc", "convai2"],
                        help="{pfg, empathetic, spc, convai2}")
    parser.add_argument("--output_file", type=str, required=True, help="File path for generated output.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="{gpt2, bart, llama3, t5}")
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
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--use_quantization", action='store_true',
                        help="Load model is 4-bit quantization to save GPU memory.")
    parser.add_argument("--method", type=str, default="GRAG",
                        help="Method of model architecture and implementation. Options: ['GRAG', 'Hist', 'TST']")
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=["MultiheadAttention"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.use_quantization and "cuda" in args.device else None

    extra_model_kwargs = {}

    model_class = load_model_class(args)
    model = model_class.from_pretrained(args.model_checkpoint, device_map="auto", quantization_config=bnb_config,
                                        torch_dtype=torch.bfloat16, **extra_model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    add_special_tokens_(model, tokenizer)

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

    # with open(args.test_dataset_path, 'r') as original_file:
    #     data = json.load(original_file)

    if args.output_file is None:
        raise Exception('Please specify file name to save the generated outputs.')

    with torch.no_grad():
        all_output = list()

        for data_index, batch in enumerate(tqdm(test_loader, miniters=5, mininterval=20)):
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
                continue

            out_ids = sample_sequence(model=model, model_inputs=model_inputs,
                                      tokenizer=tokenizer, args=args, current_output=None)

            # ["<agent>", "<user>", "<agent_persona>", "<user_persona>", "<generic_response>"]
            special_tokens_list = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

            role_tokens = set(special_tokens_list[:2])
            role_sp_tokens = set(special_tokens_list[2:4])
            special_tokens = set(
                special_tokens_list + [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id])

            # ["persona_history_input", "history_only_input", "persona", "response", "generic_response"]
            history = [x for x in model_inputs["history"][0].tolist() if
                       x not in (special_tokens - role_tokens)]
            persona = [x for x in model_inputs["persona"][0].tolist() if x not in (special_tokens - role_sp_tokens)]
            outputdict = {
                "persona": tokenizer.decode(persona, skip_special_tokens=False),
                "history": tokenizer.decode(history, skip_special_tokens=False),
                "gold_response": tokenizer.decode(response, skip_special_tokens=True),
                "pred_response": tokenizer.decode(out_ids, skip_special_tokens=True),
            }
            if args.method == "TST":
                generic_response = model_inputs["generic_response"]
                mask = (generic_response != tokenizer.pad_token_id)
                generic_response = generic_response[mask]
                outputdict["generic_response"] = tokenizer.decode(generic_response, skip_special_tokens=True),
            all_output.append(outputdict)
            if data_index >= 100:
                break

    with open(args.output_file, 'w') as fp:
        json.dump(all_output, fp, indent=2)


if __name__ == "__main__":
    run()
