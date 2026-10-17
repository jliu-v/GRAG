import logging
import os
import random
from argparse import ArgumentParser
from pprint import pformat

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

from data_utils import BART_MODEL_INPUTS
from data_utils import DatasetCreatorConvAI2
from data_utils import DatasetCreatorEmpathetic
from data_utils import DatasetCreatorPFG
from data_utils import DatasetCreatorSPC
from data_utils import GPT2_MODEL_INPUTS
from grag.common_evaluations.metrics import calculate_sample_bpc_bpb
from grag.common_evaluations.metrics import print_ppl_evaluation_results
from grag.common_evaluations.utils import save_individual_metrics
from grag.postfusion.dialogue_modules.model_utils import get_model_type
from grag.postfusion.dialogue_modules.model_utils import load_model_class
from grag.postfusion.dialogue_modules.model_utils import load_model_for_inference

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__file__)


def run():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--task", type=str, required=True, choices=["pfg", "empathetic", "spc", "convai2"],
                        help="{pfg, empathetic, spc, convai2}")
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
    parser.add_argument("--use_quantization", action='store_true',
                        help="Load model is 4-bit quantization to save GPU memory.")
    parser.add_argument("--metrics_output_file", type=str, required=False, default=None,
                        help="File path for individual metrics output.")
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
    model = load_model_for_inference(model_class, args, bnb_config=bnb_config, **extra_model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    logger.info("Prepare datasets")

    dataset_creator = dataset_creator_cls(tokenizer=tokenizer,
                                          test_dataset_path=args.test_dataset_path,
                                          test_dataset_cache=args.test_dataset_cache)
    dataset_creator.create_data_loader(args, generation_mode=False, selected_partitions=["test"])
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
        ppl_avg = []
        all_bpc = []
        all_bpb = []
        total_tokens = 0
        total_chars = 0
        total_bytes = 0
        total_loss = 0.0
        individual_metrics = []
        for test_data_index, batch in enumerate(tqdm(test_loader, miniters=5, mininterval=20)):
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            model_inputs = dict(zip(model_input_names, batch))
            model_inputs.update({
                "method": args.method
            })

            output = model(training=False,
                           padding=tokenizer.pad_token_id,
                           tokenizer=tokenizer,
                           **model_inputs)
            lm_labels, lm_logits = output["lm_labels"], output["lm_logits"]

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
