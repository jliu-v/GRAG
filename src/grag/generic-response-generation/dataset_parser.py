import csv
import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset


class DatasetParser:
    def __init__(self, sample_mode=False):
        self._file_paths = {}
        self.sample_mode = sample_mode
        self._system_prompt_template_file = None
        self._user_prompt_template_file = None

        self._system_prompt = None
        self._user_prompt_template = None


    def process_dataset(self, **kwargs):
        raise NotImplementedError()

    def generate_prompt(self, **kwargs):
        raise NotImplementedError()

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



class DatasetParserPFG(DatasetParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._file_paths['full_dialog'] = "input_datasets/persuasionforgood-master-data/data/FullData/full_dialog.csv"
        self._file_paths['full_info'] = "input_datasets/persuasionforgood-master-data/data/FullData/full_info.csv"
        self._system_prompt = None
        self.output_data_partition_name = {
            'full_dialog': 'full_dialog'
        }
        self._system_prompt_template_file = "templates/generic_responses/persuasionforgood/system_prompt.txt"
        self._user_prompt_template_file = "templates/generic_responses/persuasionforgood/user_prompt.txt"
        self.load_prompt_templates()

    def process_dataset(self):
        full_dialog = pd.read_csv(self._file_paths['full_dialog'])
        full_info = pd.read_csv(self._file_paths['full_info'])

        full_dialog = full_dialog.rename(columns={"Unnamed: 0": "turn_index",
                                                  "B2": "conv_id",
                                                  "B4": "speaker_idx",
                                                  "Unit": "utterance"})
        full_info = full_info.rename(columns={"B2": "conv_id",
                                              "B3": "user_id",
                                              "B4": "speaker_idx",
                                              "B6": "donation_amount",
                                              "B7": "num_turns"})
        feature_names = [x for x in full_info.columns.tolist() if x.endswith(".x")]
        full_info = full_info.rename(columns={f: f[:-2] for f in feature_names})  # remove ".x"
        feature_names = [f[:-2] for f in feature_names]
        full_info = full_info.replace({np.nan: None})

        dataset_sorted = full_dialog.sort_values(by=["conv_id", "turn_index"])

        conv_id = None
        conversation = None
        utterances = []
        last_speaker_idx = None
        processed_dataset = []
        skip_current_conv_id = False

        for i, row in dataset_sorted.iterrows():
            if (i + 1) % 2000 == 0:
                print(f"Dataset parsing progress: {i + 1}/{len(dataset_sorted)} at {datetime.datetime.now()}")

            is_new_conv = row["turn_index"] == 0
            speaker_idx = row["speaker_idx"]
            role = "Agent" if speaker_idx == 0 else "User"

            if is_new_conv:

                # if previous conversation id is not null and not skipped, then write it to processed
                if conv_id is not None and not skip_current_conv_id:
                    assert conversation is not None
                    conversation["utterances"] = utterances
                    processed_dataset.append(conversation)

                skip_current_conv_id = False
                conv_id = row["conv_id"]
                info_slice = full_info[full_info["conv_id"] == conv_id]
                agent_info = info_slice[info_slice["speaker_idx"] == 0][feature_names].iloc[0].to_dict()
                user_info = info_slice[info_slice["speaker_idx"] == 1][feature_names].iloc[0].to_dict()
                agent_info = {k: v for k, v in agent_info.items() if v is not None}
                user_info = {k: v for k, v in user_info.items() if v is not None}
                context = {
                    "Agent": agent_info,
                    "User": user_info
                }
                # conversation = [conv_id, context, utterances]
                conversation = {
                    "conv_id": row["conv_id"],
                    "context": context
                }
                utterances = []
                last_speaker_idx = None

            ut = f"[{role}] " + row["utterance"]

            if speaker_idx == last_speaker_idx:
                assert len(utterances) > 0
                utterances[-1] = utterances[-1] + " " + ut
                print(conversation)
            else:
                utterances.append(ut)

            last_speaker_idx = speaker_idx

        conversation["utterances"] = utterances
        processed_dataset.append(conversation)

        processed_dataset = self.remove_bad_convs(processed_dataset)
        return {"full_dialog": processed_dataset}

    def remove_bad_convs(self, processed_dataset):
        output = []
        for conversation in processed_dataset:
            conv_id = conversation["conv_id"]

            # remove conversations annotated bad
            if conv_id.startswith("BAD"):
                continue

            # remove conversations with empty speaker profile data
            agent_info = conversation["context"]["Agent"]
            user_info = conversation["context"]["User"]
            if len(agent_info) == 0 or len(user_info) == 0:
                print(f"Warning: Conversation removed. No speaker data found for conv_id: {conv_id}")
                continue

            # keep the remaining good conversations
            output.append(conversation)
        return output

    def generate_prompt(self, conversation, i):
        assert i >= 1

        context = conversation["context"]
        user_info = context["User"]
        agent_info = context["Agent"]
        demographic = ["age", "sex", "race", "edu", "marital", "employment", "income", "religion", "ideology"]
        user_demo = str({k: v for k, v in user_info.items() if k in demographic}).replace('\'', '')
        user_beha = str({k: v for k, v in user_info.items() if k not in demographic}).replace('\'', '')
        agent_demo = str({k: v for k, v in agent_info.items() if k in demographic}).replace('\'', '')
        agent_beha = str({k: v for k, v in agent_info.items() if k not in demographic}).replace('\'', '')

        history = conversation["utterances"][:i-1]
        user_prompt = conversation["utterances"][i-1]
        formatted_history = "\n".join(history)
        reference_response = conversation["utterances"][i]

        system_message = self.get_system_prompt()
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return messages


class DatasetParserEmpatheticDialog(DatasetParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._file_paths = {
            "validation": "input_datasets/empatheticdialogues/valid.csv",
            "test": "input_datasets/empatheticdialogues/test.csv",
            "train": "input_datasets/empatheticdialogues/train.csv"
        }
        self.output_data_partition_name = {
            "validation": "validation",
            "test": "test",
            "train": "train"
        }
        self._system_prompt_template_file = "templates/generic_responses/empatheticdialogues/system_prompt.txt"
        self._user_prompt_template_file = "templates/generic_responses/empatheticdialogues/user_prompt.txt"
        self.load_prompt_templates()

    def process_dataset(self):
        dataset = {
            "validation": pd.read_csv(self._file_paths["validation"], quoting=csv.QUOTE_NONE),
            "test": pd.read_csv(self._file_paths["test"], quoting=csv.QUOTE_NONE),
            "train": pd.read_csv(self._file_paths["train"], quoting=csv.QUOTE_NONE)
        }


        processed_dataset = {}

        for dataset_type, dataset_sorted in dataset.items():
            dataset_sorted = dataset[dataset_type].sort_values(by=["conv_id", "utterance_idx"])
            print(f"Processing dataset: {dataset_type} at {datetime.datetime.now()}")

            conv_id = None
            conversation = None
            utterances = []
            speaker_map = {}
            last_speaker_idx = None
            processed_dataset[dataset_type] = []

            for i, row in dataset_sorted.iterrows():
                if (i + 1) % 2000 == 0:
                    print(f"Dataset parsing progress: {i + 1}/{len(dataset_sorted)} at {datetime.datetime.now()}")

                is_new_conv = (conv_id != row["conv_id"] or conv_id is None)
                speaker_idx = row["speaker_idx"]

                if is_new_conv:
                    if conv_id is not None:
                        assert conversation is not None
                        conversation["utterances"] = utterances
                        processed_dataset[dataset_type].append(conversation)

                    conv_id = row["conv_id"]
                    # conversation = [conversation_id, context, prompt, utterances]
                    conversation = {
                        "conv_id": row["conv_id"],
                        "context": row["context"],
                        "prompt": row["prompt"].replace("_comma_", ",")
                    }
                    utterances = []
                    speaker_map = {}
                    last_speaker_idx = None

                if speaker_idx not in speaker_map:
                    speaker_map[speaker_idx] = len(speaker_map)

                ut = row["utterance"].replace("_comma_", ",")
                if ut.count("\"") % 2 == 1 and ut[0] == "\"":
                    ut = ut.lstrip("\"")

                if speaker_idx == last_speaker_idx:
                    assert len(utterances) > 0
                    utterances[-1] = utterances[-1] + " " + ut
                    print(conversation)
                else:
                    role = "User" if speaker_map[speaker_idx] == 0 else "Agent"
                    ut = f"[{role}] {ut}"
                    utterances.append(ut)

                last_speaker_idx = speaker_idx

            conversation["utterances"] = utterances
            processed_dataset[dataset_type].append(conversation)

        return processed_dataset

    def generate_prompt(self, conversation, i):
        assert i >= 1

        context = conversation["context"]
        history = conversation["utterances"][:i-1]
        user_prompt = conversation["utterances"][i-1]
        formatted_history = "\n".join(history)
        reference_response = conversation["utterances"][i]

        system_message = self.get_system_prompt()
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return messages


class DatasetParserSPC(DatasetParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._dataset_card = "google/Synthetic-Persona-Chat"
        self.output_data_partition_name = {
            "validation": "validation",
            "test": "test",
            "train": "train"
        }
        # same template as ConvAI2
        self._system_prompt_template_file = "templates/generic_responses/convai2/system_prompt.txt"
        self._user_prompt_template_file = "templates/generic_responses/convai2/user_prompt.txt"
        self.load_prompt_templates()

    def process_dataset(self):
        dataset = {
            "train": load_dataset(path=self._dataset_card, split="train", trust_remote_code=True),
            "validation": load_dataset(path=self._dataset_card, split="validation", trust_remote_code=True),
            "test": load_dataset(path=self._dataset_card, split="test", trust_remote_code=True)
        }

        processed_dataset = {}

        for dataset_type, dataset_sorted in dataset.items():
            dataset_sorted = dataset[dataset_type]
            print(f"Processing dataset: {dataset_type} at {datetime.datetime.now()}")

            processed_dataset[dataset_type] = []

            for i, row in enumerate(dataset_sorted):
                if self.sample_mode and len(processed_dataset[dataset_type]) > 50:
                    break

                if (i + 1) % 2000 == 0:
                    print(f"Dataset parsing progress: {i + 1}/{len(dataset_sorted)} at {datetime.datetime.now()}")

                if row["Best Generated Conversation"] is None:
                    continue

                context = {
                    "User": row["user 1 personas"].split("\n"),
                    "Agent": row["user 2 personas"].split("\n")
                }
                utterances = (row["Best Generated Conversation"] \
                    .replace("User 1: ", "[User] ") \
                    .replace("User 2: ", "[Agent] ") \
                    .split("\n"))
                has_bad_conv = any(not(u.startswith("[User] ") or u.startswith("[Agent] ")) for u in utterances)
                if has_bad_conv:
                    continue

                conversation = {
                    "conv_id": f"{dataset_type}_{i}",
                    "context": context,
                    "utterances": utterances
                }

                processed_dataset[dataset_type].append(conversation)

        return processed_dataset

    def generate_prompt(self, conversation, i):
        assert i >= 1

        user_persona = "\n".join(conversation["context"]["User"])
        agent_persona = "\n".join(conversation["context"]["Agent"])
        history = conversation["utterances"][:i-1]
        user_prompt = conversation["utterances"][i-1]
        formatted_history = "\n".join(history)
        reference_response = conversation["utterances"][i]

        system_message = self.get_system_prompt()
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return messages


class DatasetParserConvAI2(DatasetParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._dataset_card = "convai-challenge/conv_ai_2"
        self.output_data_partition_name = {
            "valid": "valid",
            "train": "train"
        }
        self._system_prompt_template_file = "templates/generic_responses/convai2/system_prompt.txt"
        self._user_prompt_template_file = "templates/generic_responses/convai2/user_prompt.txt"
        self.load_prompt_templates()

    def process_dataset(self):

        datasets = {
            "train": "input_datasets/convai2/train_both_original.txt",
            "valid": "input_datasets/convai2/valid_both_original.txt"
        }
        processed_dataset = {}

        for dataset_type, dataset_path in datasets.items():
            processed_dataset[dataset_type] = []

            utterances = []
            context = {
                "User": [],
                "Agent": []
            }
            with open(dataset_path, mode="r", newline="\n", encoding="utf-8") as fp:
                for i, line in enumerate(fp):

                    if self.sample_mode and len(processed_dataset[dataset_type]) > 20:
                        break

                    line_id = line.split(" ")[0]
                    is_new_conv = line_id == "1"

                    if is_new_conv:
                        if len(utterances) > 0:
                            conversation = {
                                "conv_id": f"{dataset_type}_{len(processed_dataset[dataset_type])}",
                                "context": context,
                                "utterances": utterances
                            }
                            processed_dataset[dataset_type].append(conversation)

                        utterances = []
                        context = {
                            "User": [],
                            "Agent": []
                        }

                    if "your persona: " in line:  # agent persona
                        persona = line.split("your persona: ")[-1].rstrip("\n")
                        context["Agent"].append(persona)
                    elif "partner's persona: " in line:  # agent persona
                        persona = line.split("partner's persona: ")[-1].rstrip("\n")
                        context["User"].append(persona)
                    else:
                        splits = line.replace(f"{line_id} ", "").split("\t")
                        user_uttr = splits[0]
                        agent_uttr = splits[1]
                        if user_uttr != "__SILENCE__":
                            utterances.append(f"[User] {splits[0]}")
                        if agent_uttr != "__SILENCE__":
                            utterances.append(f"[Agent] {splits[1]}")

                conversation = {
                    "conv_id": f"{dataset_type}_{len(processed_dataset[dataset_type])}",
                    "context": context,
                    "utterances": utterances
                }
                processed_dataset[dataset_type].append(conversation)

        return processed_dataset

    def generate_prompt(self, conversation, i):
        assert i >= 1

        user_persona = "\n".join(conversation["context"]["User"])
        agent_persona = "\n".join(conversation["context"]["Agent"])
        history = conversation["utterances"][:i-1]
        formatted_history = "\n".join(history)
        user_prompt = conversation["utterances"][i - 1]
        reference_response = conversation["utterances"][i]

        system_message = self.get_system_prompt()
        user_message = self.get_user_prompt_template().format(formatted_history=formatted_history,
                                                              user_prompt=user_prompt)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return messages
