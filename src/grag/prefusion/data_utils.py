import json
import logging
import os
from collections import defaultdict
from itertools import chain
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from grag.prefusion.dialogue_modules.model_utils import get_model_type

SPECIAL_TOKENS = ["<agent>", "<user>", "<agent_persona>", "<user_persona>", "<generic_response>"]
ATTR_TO_SPECIAL_TOKEN = {
    "additional_special_tokens": ["<agent>", "<user>", "<agent_persona>", "<user_persona>", "<generic_response>"]}
BART_MODEL_INPUTS = ["input_ids", "decoder_input", "lm_labels",
                     "history", "persona", "response", "generic_response"]
GPT2_MODEL_INPUTS = ["input_ids", "input_tti", "lm_labels",
                     "history", "history_tti",
                     "persona", "persona_tti",
                     "response", "response_tti",
                     "generic_response", "generic_response_tti"]

logger = logging.getLogger(__file__)


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. """
    for name in dataset.keys():
        if name == "conv_id":
            continue

        # for all data that are 1-d vector
        max_l = max(len(x) for x in dataset[name])
        pad_val = -100 if "lm_labels" in name else padding
        dataset[name] = [x + [pad_val] * (max_l - len(x)) for x in dataset[name]]

    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.vocab)
    num_added_tokens = 0
    if tokenizer.pad_token is None:
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = '<|pad|>'
        num_added_tokens += 1
        print('<|pad|> token added!')
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    num_added_tokens += len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens)  # 50265, 4
    if num_added_tokens > 0 and model is not None:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens, mean_resizing=False)


def build_bart_instance(conv_id, context, history, response, generic_response,
                        tokenizer, args, generation_mode=False, with_eos=True):
    # SPECIAL_TOKENS = ["<agent>", "<user>", "<agent_persona>", "<user_persona>", "<generic_response>"]
    agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id  # Note: t5 doesn't have bos token. So put pad here, which will be removed later.
    eos = tokenizer.eos_token_id

    # history should already start with either "<agent>" or "<user>"
    response = [agent_st] + response
    generic_response = [generic_st] + generic_response
    if "Agent" in context:
        agent_persona = list(chain(*context["Agent"]))
    else:
        agent_persona, agent_persona_tti = [], []
    if "User" in context:
        user_persona = list(chain(*context["User"]))
    else:
        user_persona = []
    persona = agent_persona + user_persona

    if args.method == "GRAG":
        prefusion_inputs = "persona-history-tst"
    elif args.method == "Hist":
        prefusion_inputs = "persona-history"
    elif args.method == "TST":
        prefusion_inputs = "persona-tst"
    else:
        raise ValueError(f"Unknown method for pre-fusion models: {args.method}")

    if generation_mode:
        if prefusion_inputs == "persona-history":
            input_ids = [[bos], persona, list(chain(*history))]
        elif prefusion_inputs == "persona-tst":
            input_ids = [[bos], persona, generic_response]
        elif prefusion_inputs == "history-only":
            input_ids = [[bos], list(chain(*history))]
        elif prefusion_inputs == "persona-history-tst":
            input_ids = [[bos], persona, list(chain(*history)), generic_response]
        else:
            raise ValueError(f"Unknown pre-fusion inputs: {prefusion_inputs}")
        decoder_input = [bos] + [agent_st]
    else:
        if prefusion_inputs == "persona-history":
            input_ids = [[bos], persona, list(chain(*history)), [eos] if with_eos else []]
        elif prefusion_inputs == "persona-tst":
            input_ids = [[bos], persona, generic_response, [eos] if with_eos else []]
        elif prefusion_inputs == "history-only":
            input_ids = [[bos], list(chain(*history)), [eos] if with_eos else []]
        elif prefusion_inputs == "persona-history-tst":
            input_ids = [[bos], persona, list(chain(*history)), generic_response, [eos] if with_eos else []]
        else:
            raise ValueError(f"Unknown pre-fusion inputs: {prefusion_inputs}")

        decoder_input = [bos] + response + ([eos] if with_eos else [])

    instance = {}
    instance["conv_id"] = conv_id
    instance["history"] = list(chain(*history))
    instance["persona"] = persona
    instance["response"] = response[1:]
    instance["generic_response"] = generic_response[1:]

    instance["input_ids"] = list(chain(*input_ids))
    instance["decoder_input"] = decoder_input
    instance["lm_labels"] = [bos] + decoder_input[1:]

    assert len(instance["decoder_input"]) == len(instance["lm_labels"])

    return instance


def build_gpt2_instance(conv_id, context, history, response, generic_response,
                        tokenizer, args, generation_mode=False, with_eos=True):
    # SPECIAL_TOKENS = ["<agent>", "<user>", "<agent_persona>", "<user_persona>", "<generic_response>"]
    agent_st, user_st, agent_p_st, user_p_st, generic_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id  # Note: Qwen doesn't have bos token. So put pad here, which will be removed later.
    eos = tokenizer.eos_token_id

    # history should already start with either "<agent>" or "<user>"
    history_tti = [[sentence[0]] * len(sentence) for sentence in history]
    response = [agent_st] + response
    response_tti = [agent_st] * len(response)
    generic_response = [generic_st] + generic_response
    generic_response_tti = [generic_st] * len(generic_response)
    if "Agent" in context:
        agent_persona = list(chain(*context["Agent"]))
        agent_persona_tti = [agent_p_st] * len(agent_persona)
    else:
        agent_persona, agent_persona_tti = [], []
    if "User" in context:
        user_persona = list(chain(*context["User"]))
        user_persona_tti = [user_p_st] * len(user_persona)
    else:
        user_persona, user_persona_tti = [], []
    persona = agent_persona + user_persona
    persona_tti = agent_persona_tti + user_persona_tti

    if args.method == "GRAG":
        prefusion_inputs = "persona-history-tst"
    elif args.method == "Hist":
        prefusion_inputs = "persona-history"
    elif args.method == "TST":
        prefusion_inputs = "persona-tst"
    else:
        raise ValueError(f"Unknown method for pre-fusion models: {args.method}")

    if generation_mode:
        if prefusion_inputs == "persona-history":
            input_ids = [[bos], persona, list(chain(*history)), [agent_st]]
            input_tti = [[persona_tti[0]], persona_tti, list(chain(*history_tti)), [agent_st]]
        elif prefusion_inputs == "persona-tst":
            input_ids = [[bos], persona, generic_response, [agent_st]]
            input_tti = [[persona_tti[0]], persona_tti, generic_response_tti, [agent_st]]
        elif prefusion_inputs == "history-only":
            input_ids = [[bos], list(chain(*history)), [agent_st]]
            input_tti = [[history_tti[0][0]], list(chain(*history_tti)), [agent_st]]
        elif prefusion_inputs == "persona-history-tst":
            input_ids = [[bos], persona, list(chain(*history)), generic_response, [agent_st]]
            input_tti = [[persona_tti[0]], persona_tti, list(chain(*history_tti)), generic_response_tti, [agent_st]]
        else:
            raise ValueError(f"Unknown pre-fusion inputs: {prefusion_inputs}")
    else:
        if prefusion_inputs == "persona-history":
            input_ids = [[bos], persona, list(chain(*history)), response + ([eos] if with_eos else [])]
            input_tti = [[persona_tti[0]], persona_tti, list(chain(*history_tti)),
                         response_tti + (response_tti[-1:] if with_eos else [])]
        elif prefusion_inputs == "persona-tst":
            input_ids = [[bos], persona, generic_response, response + ([eos] if with_eos else [])]
            input_tti = [[persona_tti[0]], persona_tti, generic_response_tti,
                         response_tti + (response_tti[-1:] if with_eos else [])]
        elif prefusion_inputs == "history-only":
            input_ids = [[bos], list(chain(*history)), response + ([eos] if with_eos else [])]
            input_tti = [[history_tti[0][0]], list(chain(*history_tti)),
                         response_tti + (response_tti[-1:] if with_eos else [])]
        elif prefusion_inputs == "persona-history-tst":
            input_ids = [[bos], persona, list(chain(*history)), generic_response,
                         response + ([eos] if with_eos else [])]
            input_tti = [[persona_tti[0]], persona_tti, list(chain(*history_tti)), generic_response_tti,
                         response_tti + (response_tti[-1:] if with_eos else [])]
        else:
            raise ValueError(f"Unknown pre-fusion inputs: {prefusion_inputs}")

    instance = {}
    instance["conv_id"] = conv_id
    instance["history"] = list(chain(*history))
    instance["history_tti"] = list(chain(*history_tti))
    instance["persona"] = persona
    instance["persona_tti"] = persona_tti
    instance["response"] = response[1:]
    instance["response_tti"] = response_tti[1:]
    instance["generic_response"] = generic_response[1:]
    instance["generic_response_tti"] = generic_response_tti[1:]

    instance["input_ids"] = list(chain(*input_ids))
    instance["input_tti"] = list(chain(*input_tti))
    instance["lm_labels"] = ([-100] * sum(len(s) for s in input_ids[:-1])) + [-100] + \
                            input_ids[-1][1:]

    assert len(instance["input_ids"]) == len(instance["input_tti"]) == len(instance["lm_labels"])

    return instance


class DatasetCreator:
    def __init__(self, tokenizer, **kwargs):
        train_dataset_path = kwargs.get("train_dataset_path")
        train_dataset_cache = kwargs.get("train_dataset_cache")
        test_dataset_path = kwargs.get("test_dataset_path")
        test_dataset_cache = kwargs.get("test_dataset_cache")
        valid_dataset_path = kwargs.get("valid_dataset_path")
        valid_dataset_cache = kwargs.get("valid_dataset_cache")

        self.dataset_path = {
            "train": train_dataset_path,
            "test": test_dataset_path,
            "valid": valid_dataset_path
        }
        self.dataset_cache = {
            "train": train_dataset_cache,
            "test": test_dataset_cache,
            "valid": valid_dataset_cache
        }
        self.dataset_path = {k: v for k, v in self.dataset_path.items() if v is not None}
        self.dataset_cache = {k: v for k, v in self.dataset_cache.items() if v is not None}

        self.tokenizer = tokenizer
        self.processed_dataset = {}
        self.dataset_loader = {}

    def create_dataset(self):
        self.processed_dataset = {pl: self.process_dataset_partition(partition_label=pl) for pl in
                                  self.dataset_path.keys()}

    def process_context(self, context: dict):
        raise NotImplementedError("Please implement the process_context function in child classes.")

    def process_dataset_partition(self, partition_label: Literal["train", "valid", "test"]):
        tokenizer = self.tokenizer
        dataset_path = self.dataset_path[partition_label]
        dataset_cache = self.dataset_cache[partition_label] + type(tokenizer).__name__

        if dataset_cache and os.path.isfile(dataset_cache):
            logger.info("Load tokenized dataset from cache at %s", dataset_cache)
            dataset = torch.load(dataset_cache)
        else:
            logger.info("Process dataset from %s", dataset_path)
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = list()
                for conversation in dataset:
                    # process shared information on conversation level
                    conv_id = conversation["conv_id"]
                    context = conversation["context"]
                    context = self.process_context(context)
                    context = {k: [slim_tokenize(tokenizer, c) for c in ctxt] for k, ctxt in context.items()}

                    # process turn information on utterance level
                    utterances = conversation["utterances"]
                    generic_utterances = conversation["generic_utterances"]
                    history_enc = list()

                    if len(utterances) != len(generic_utterances):
                        raise ValueError(f"mismatched length: {conv_id}")

                    for i, utt in enumerate(utterances):

                        if utt.startswith("[Agent]"):

                            hist = utt.replace("[Agent]", "<agent>")
                            hist_enc = slim_tokenize(tokenizer, hist)

                            if i > 0:
                                # If the first utterance is agent, don't use it as a data point
                                # Otherwise, in agent turn, create a data point for both original utt and generic utt
                                # remove leading "[Agent] "
                                resp = utt.replace("[Agent] ", "")
                                gresp = generic_utterances[i]
                                gresp = gresp.replace("[Agent] ", "")
                                resp_enc = slim_tokenize(tokenizer, resp)
                                gresp_enc = slim_tokenize(tokenizer, gresp)

                                data_point = dict()
                                data_point["conv_id"] = conv_id
                                data_point["context"] = context
                                data_point["history"] = history_enc.copy()
                                data_point["response"] = resp_enc
                                data_point["generic_response"] = gresp_enc
                                dataset_enc.append(data_point)
                        elif utt.startswith("[User]"):
                            # user turn, do nothing
                            hist = utt.replace("[User]", "<user>")
                            hist_enc = slim_tokenize(tokenizer, hist)
                        else:
                            raise ValueError("Unknown utterance: {}".format(utt))

                        history_enc.append(hist_enc)

            dataset = dataset_enc
            torch.save(dataset, dataset_cache)
        return dataset

    def create_data_loader(self, args, generation_mode=False,
                           selected_partitions: list[Literal["train", "valid", "test"]] | None = None):
        all_partitions = set(self.dataset_path.keys())
        if selected_partitions is None:
            selected_partitions = all_partitions
        else:
            selected_partitions = set(selected_partitions)
            assert selected_partitions.issubset(all_partitions)

        for partition_label in selected_partitions:
            loader = self.create_data_loader_partition(args=args,
                                                       partition_label=partition_label,
                                                       generation_mode=generation_mode)
            self.dataset_loader[partition_label] = loader

    def create_data_loader_partition(self, args,
                                     partition_label: Literal["train", "valid", "test"],
                                     generation_mode: bool = True):

        if partition_label not in self.processed_dataset:
            self.processed_dataset[partition_label] = self.process_dataset_partition(partition_label=partition_label)

        dataset_partition = self.processed_dataset[partition_label]
        logger.info(f"Build dataloaders for partition {partition_label}. Size = {len(dataset_partition)}")

        tokenizer = self.tokenizer
        model_name = args.model_name

        datasets = defaultdict(list)

        model_type = get_model_type(model_name)
        if model_type == "decoder-only":
            build_instance = build_gpt2_instance
            models_input_names = GPT2_MODEL_INPUTS
        elif model_type == "encoder-decoder":
            build_instance = build_bart_instance
            models_input_names = BART_MODEL_INPUTS
        else:
            raise ValueError(f"Model model_type is not supported: {model_type}.")

        for dialog in dataset_partition:
            conv_id = dialog["conv_id"]
            context = dialog["context"]
            if args.max_history is not None:
                history = dialog["history"][-(2 * args.max_history):]
            else:
                history = dialog["history"]
            response = dialog["response"]
            generic_response = dialog["generic_response"]

            instance = build_instance(conv_id, context, history, response, generic_response,
                                      tokenizer, args, generation_mode=generation_mode)

            for input_name, input_array in instance.items():
                datasets[input_name].append(input_array)

        logger.info("Pad inputs and convert to Tensor")
        tensor_datasets = []

        dataset_partition = pad_dataset(datasets, padding=tokenizer.pad_token_id)

        for input_name in models_input_names:
            tensor = torch.tensor(dataset_partition[input_name], device='cpu')
            tensor_datasets.append(tensor)

        tensor_dataset = TensorDataset(*tensor_datasets)
        if partition_label == "train":
            batch_size = args.train_batch_size
        elif partition_label == "valid":
            batch_size = args.valid_batch_size
        elif partition_label == "test":
            batch_size = args.test_batch_size
        else:
            raise ValueError("Unknown partition label: {}".format(partition_label))
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=(partition_label == "train"),
                                 num_workers=4, pin_memory=True)
        logger.info(
            f"Dataset {partition_label} (Batch, Candidates, Seq length): {tensor_dataset.tensors[0].shape}")

        return data_loader


class DatasetCreatorPFG(DatasetCreator):
    def process_context(self, context: dict[list[str]]):
        context = {role: str(vals).replace("\"", "").replace("'", "") for role, vals in context.items()}
        context = {role: [prepend_speaker_role_to_persona(role, vals)] for role, vals in context.items()}
        return context


class DatasetCreatorConvAI2(DatasetCreator):

    def process_context(self, context: dict[list[str]]):
        context = {role: [prepend_speaker_role_to_persona(role, p) for p in persona] for role, persona in
                   context.items()}
        return context


class DatasetCreatorSPC(DatasetCreator):
    def process_context(self, context: dict[list[str]]):
        context = {role: [prepend_speaker_role_to_persona(role, p) for p in persona] for role, persona in
                   context.items()}
        return context


class DatasetCreatorEmpathetic(DatasetCreator):

    def process_context(self, context: str):
        return {"Agent": [prepend_speaker_role_to_persona("Agent", context)]}


def prepend_speaker_role_to_persona(role, persona_str):
    if role == "Agent":
        return "<agent_persona> " + persona_str
    elif role == "User":
        return "<user_persona> " + persona_str
    else:
        raise ValueError("Unknown role: {}".format(role))


def slim_tokenize(tokenizer, sentence):
    """Avoid using tokenizer.encode() so it doesn't include [BOS]/[EOS] tokens."""
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip()))
