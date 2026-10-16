import json
import logging

import random

from typing import Literal

logger = logging.getLogger(__file__)


class DatasetParser:
    def __init__(self, train_dataset_path, test_dataset_path, sample_mode=False):
        self.dataset_path = {
            "train": train_dataset_path,
            "test": test_dataset_path
        }

        self.processed_dataset = {}
        self._file_paths = {}
        self.sample_mode = sample_mode
        self._system_prompt_template_file = None
        self._user_prompt_template_file = None
        self._system_prompt = None
        self._user_prompt_template = None

    def create_dataset(self):
        self.processed_dataset = {pl: self.load_dataset_partition(partition_label=pl) for pl in
                                  self.dataset_path.keys()}

    def load_dataset_partition(self, partition_label: Literal["train", "valid", "test"]):
        dataset_path = self.dataset_path[partition_label]

        logger.info("Loading dataset partition [%s] from %s", partition_label, dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        # remove bad conversations where there is no Agent turn
        dataset = [c for c in dataset if any(u.startswith("[Agent] ") for u in c["utterances"])]
        return dataset

    def get_system_prompt(self):
        if self._system_prompt is None:
            raise ValueError("self._system_prompt is not set.")
        return self._system_prompt

    def get_user_prompt_template(self):
        if self._user_prompt_template is None:
            raise ValueError("self._user_prompt_template is not set.")
        return self._user_prompt_template

    def load_prompt_templates(self):
        if self._system_prompt_template_file is None or self._user_prompt_template_file is None:
            raise ValueError("self._system_prompt_template_file or self._user_prompt_template_file is None")

        with open(self._system_prompt_template_file, 'r', encoding='utf-8') as file:
            self._system_prompt = file.read()

        with open(self._user_prompt_template_file, 'r', encoding='utf-8') as file:
            self._user_prompt_template = file.read()

    def process_context(self, context):
        raise NotImplementedError()

    def process_train_shot(self, idx, shot):
        # first, sample one turn from agent as the example
        utterances = shot["utterances"]
        agent_turns = [i for i, u in enumerate(utterances) if u.startswith("[Agent]")]
        n_turns = len(agent_turns)
        if n_turns > 5:
            sample_agent_turns = agent_turns[max(0, int(0.25 * n_turns)):min(n_turns, int(0.75 * n_turns), 10)]
            i = random.choice(sample_agent_turns)
        else:
            i = agent_turns[-1]

        # prepare context
        context = shot["context"]
        context = self.process_context(context)

        # prepare history
        history = utterances[:i - 1]
        user_prompt = utterances[i - 1]
        formatted_history = "\n".join(history)
        reference_response = utterances[i].replace("[Agent] ", "")
        generic_response = shot["generic_utterances"][i].replace("[Agent] ", "")

        # generate prompt message from template (without reference response)
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt,
                                                              persona=context,
                                                              generic_response=generic_response)
        # append reference response at the end
        user_message = f"# Example {idx} #\n\n" + user_message + f"\n{reference_response}"

        message = {"role": "user", "content": user_message}
        return message

    def process_test_shot(self, i, conversation):
        # prepare context
        context = conversation["context"]
        context = self.process_context(context)

        # prepare history
        history = conversation["utterances"][:i - 1]
        user_prompt = conversation["utterances"][i - 1]
        formatted_history = "\n".join(history)
        generic_response = conversation["generic_utterances"][i].replace("[Agent] ", "")

        # generate prompt message from template (without reference response)
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt,
                                                              persona=context,
                                                              generic_response=generic_response)
        # append reference response at the end
        user_message = f"# Your Turn #\n\n" + user_message

        message = {"role": "user", "content": user_message}
        return message

    def process_test_shot_zero(self, i, conversation):
        # prepare context
        context = conversation["context"]
        context = self.process_context(context)

        # prepare history
        utterances = conversation["utterances"]
        history = conversation["utterances"][:i - 1]
        user_prompt = conversation["utterances"][i - 1]
        formatted_history = "\n".join(history)
        reference_response = utterances[i].replace("[Agent] ", "")
        generic_response = conversation["generic_utterances"][i].replace("[Agent] ", "")

        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt,
                                                              persona=context,
                                                              generic_response=generic_response)

        message = {"role": "user", "content": user_message}
        return message

    def generate_prompt_messages(self, i, conversation, shots):
        assert i >= 1

        system_message = self.get_system_prompt()
        prompt_messages = [{"role": "system", "content": system_message}]
        # few-shot
        if shots is not None:
            # process training shots
            processed_train_messages = [self.process_train_shot(idx + 1, s) for idx, s in enumerate(shots)]

            # process inference prompt
            processed_test_prompt = self.process_test_shot(i, conversation)

            # final prompt
            prompt_messages = prompt_messages + processed_train_messages + [processed_test_prompt]

        # zero-shot
        else:
            # process inference prompt
            processed_test_prompt = self.process_test_shot_zero(i, conversation)
            prompt_messages = prompt_messages + [processed_test_prompt]

        return prompt_messages


class DatasetCreatorPFG(DatasetParser):
    def __init__(self, train_dataset_path, test_dataset_path, context_mode, sample_mode=False):
        super().__init__(train_dataset_path, test_dataset_path, sample_mode=sample_mode)
        if context_mode == "persona-history":
            self._system_prompt_template_file = "templates/fewshot/pfg/history/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/pfg/history/user_prompt.txt"
        elif context_mode == "persona-tst":
            self._system_prompt_template_file = "templates/fewshot/pfg/tst/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/pfg/tst/user_prompt.txt"
        elif context_mode == "history-generic":
            self._system_prompt_template_file = "templates/fewshot/pfg/history_generic/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/pfg/history_generic/user_prompt.txt"
        else:
            raise NotImplementedError(f"Invalid context_mode: {context_mode}")
        self.load_prompt_templates()

    def process_context(self, context: dict[list[str]]):
        # context = {role: str(vals).replace("\"", "").replace("'", "") for role, vals in context.items()}
        context = "\n".join(
            f"{role}:\n" +
            "\n".join(f"- {k}: {v}" for k, v in attrs.items())
            for role, attrs in context.items()
        )
        return context


class DatasetCreatorConvAI2(DatasetParser):
    def __init__(self, train_dataset_path, test_dataset_path, context_mode, sample_mode=False):
        super().__init__(train_dataset_path, test_dataset_path, sample_mode=sample_mode)
        if context_mode == "persona-history":
            self._system_prompt_template_file = "templates/fewshot/convai2/history/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/convai2/history/user_prompt.txt"
        elif context_mode == "persona-tst":
            self._system_prompt_template_file = "templates/fewshot/convai2/tst/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/convai2/tst/user_prompt.txt"
        elif context_mode == "history-generic":
            self._system_prompt_template_file = "templates/fewshot/convai2/history_generic/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/convai2/history_generic/user_prompt.txt"
        else:
            raise NotImplementedError(f"Invalid context_mode: {context_mode}")
        self.load_prompt_templates()

    def process_context(self, context: dict[list[str]]):
        # context = str(context)
        context = "\n".join(
            f"{role}:\n" +
            "\n".join(f"- {entry}" for entry in persona_entries)
            for role, persona_entries in context.items()
        )
        return context


class DatasetCreatorSPC(DatasetCreatorConvAI2):
    def __init__(self, train_dataset_path, test_dataset_path, context_mode, sample_mode=False):
        super().__init__(train_dataset_path, test_dataset_path, context_mode, sample_mode=sample_mode)


class DatasetCreatorEmpathetic(DatasetParser):
    def __init__(self, train_dataset_path, test_dataset_path, context_mode, sample_mode=False):
        super().__init__(train_dataset_path, test_dataset_path, sample_mode=sample_mode)
        if context_mode == "persona-history":
            self._system_prompt_template_file = "templates/fewshot/empathetic/history/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/empathetic/history/user_prompt.txt"
        elif context_mode == "persona-tst":
            self._system_prompt_template_file = "templates/fewshot/empathetic/tst/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/empathetic/tst/user_prompt.txt"
        elif context_mode == "history-generic":
            self._system_prompt_template_file = "templates/fewshot/empathetic/history_generic/system_prompt.txt"
            self._user_prompt_template_file = "templates/fewshot/empathetic/history_generic/user_prompt.txt"
        else:
            raise NotImplementedError(f"Invalid context_mode: {context_mode}")
        self.load_prompt_templates()

    def process_context(self, context: str) -> str:
        return str(context)
