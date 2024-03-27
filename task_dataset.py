# author = liuwei
# date = 2023-03-06

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import random

random.seed(106524)

class BaseDataset(Dataset):
    def __init__(self, file_name_or_data, params):
        """
        single stream dataset
        Args:
            file_name:
            params:
        """
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1 # start from 0
        self.relation_type = params["relation_type"]
        self.use_conn = params["use_conn"]

        self.init_np_dataset(file_name_or_data)

    def init_np_dataset(self, file_name_or_data):
        if type(file_name_or_data) == str and os.path.exists(file_name_or_data):
            with open(file_name_or_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
        elif type(file_name_or_data) == list:
            lines = file_name_or_data
        else:
            raise Exception("Not file name or data list!!!")

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_label_ids = []
        all_possible_label_ids = []
        label_frequency = {}

        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if sample["relation_type"].lower() != self.relation_type:
                    continue
                # read and filter
                arg1 = sample["arg1"]
                arg2 = sample["arg2"]
                all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                if len(all_level_relation_class) > self.label_level:
                    relation_class = all_level_relation_class[self.label_level].lower()
                else:
                    relation_class = None
                possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                if (relation_class is None) or (relation_class not in self.label_list):
                    continue
                if relation_class in label_frequency:
                    label_frequency[relation_class] += 1
                else:
                    label_frequency[relation_class] = 1
                if "conn" in sample:
                    conn = sample["conn"].split("##")[0].lower()
                else:
                    conn = ""
                if self.use_conn:
                    arg2 = conn + " " + arg2
                text = arg1 + " " + arg2
                text_res = self.tokenizer(
                    text=text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                input_ids = text_res.input_ids[0]
                attention_mask = text_res.attention_mask[0]
                if "token_type_ids" in text_res:
                    token_type_ids = text_res["token_type_ids"][0]
                else:
                    token_type_ids = torch.zeros_like(attention_mask)

                label_id = self.label_list.index(relation_class)
                possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                for label in possible_relation_classes:
                    possible_label_ids[self.label_list.index(label)] = 1

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_token_type_ids.append(token_type_ids)
                all_label_ids.append(label_id)
                all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_token_type_ids), (len(all_input_ids), len(all_token_type_ids))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.token_type_ids = all_token_type_ids
        self.label_ids = np.array(all_label_ids)
        self.possible_label_ids = np.array(all_possible_label_ids)
        self.total_size = len(self.input_ids)
        print(label_frequency)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.attention_mask[index],
            self.token_type_ids[index],
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )


class TwoDataset(Dataset):
    def __init__(self, file_name_or_data, params):
        """
        single stream dataset
        Args:
            file_name:
            params:
        """
        self.relation_type = params["relation_type"]
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_list = params["label_list"]
        self.label_level = params["label_level"] - 1 # start from 0
        self.conn_list = params["conn_list"]
        self.use_conn = params["use_conn"]
        self.init_np_dataset(file_name_or_data)

    def init_np_dataset(self, file_name_or_data):
        if type(file_name_or_data) == str and os.path.exists(file_name_or_data):
            with open(file_name_or_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
        elif type(file_name_or_data) == list:
            lines = file_name_or_data
        else:
            raise Exception("Not file name or data list!!!")

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_mask_position_ids = []
        all_label_ids = []
        all_possible_label_ids = []
        all_conn_ids = []
        label_frequency = {}

        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if sample["relation_type"].lower() != self.relation_type:
                    continue

                # read and filter
                arg1 = sample["arg1"].replace("\n", ". ").strip()
                arg2 = sample["arg2"].replace("\n", ". ").strip()
                # conn = sample["conn"].split("##")[0].lower()
                if "conn" in sample:
                    conn = sample["conn"].split("##")[0].lower()
                else:
                    conn = ""
                all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                possible_all_level_relation_classes = [item.split(".") for item in sample["relation_class"].split("##")]
                if len(all_level_relation_class) > self.label_level:
                    relation_class = all_level_relation_class[self.label_level].lower()
                else:
                    relation_class = None
                possible_relation_classes = [item[self.label_level].lower() for item in possible_all_level_relation_classes if len(item) > self.label_level]
                possible_relation_classes = [item for item in possible_relation_classes if item in self.label_list]
                if (relation_class is None) or (relation_class not in self.label_list):
                    continue
                if relation_class in label_frequency:
                    label_frequency[relation_class] += 1
                else:
                    label_frequency[relation_class] = 1

                # convert to ids
                if self.use_conn:
                    arg2 = conn + " " + arg2
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    tokens = [self.tokenizer.cls_token] + tokens_1 + tokens_2
                else:
                    tokens_1 = self.tokenizer.tokenize(arg1)
                    tokens_2 = self.tokenizer.tokenize(arg2)
                    tokens = [self.tokenizer.cls_token] + tokens_1 + [self.tokenizer.mask_token] + tokens_2
                if len(tokens) > self.max_seq_length - 1:
                    tokens = tokens[:self.max_seq_length - 1]
                tokens = tokens + [self.tokenizer.sep_token]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask_position_id = len(tokens_1) + 1
                if mask_position_id >= self.max_seq_length:
                    continue
                assert mask_position_id < self.max_seq_length, (mask_position_id, self.max_seq_length)
                if conn in self.conn_list:
                    conn_id = self.conn_list.index(conn)
                else:
                    conn_id = self.conn_list.index("<unk>")
                label_id = self.label_list.index(relation_class)
                possible_label_ids = np.zeros(len(self.label_list), dtype=np.int)
                for label in possible_relation_classes:
                    possible_label_ids[self.label_list.index(label)] = 1

                # padding
                input_ids = np.ones(self.max_seq_length, dtype=np.int)
                attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                token_type_ids = np.ones(self.max_seq_length, dtype=np.int)
                input_ids = input_ids * self.tokenizer.pad_token_id
                input_ids[:len(token_ids)] = token_ids
                attention_mask[:len(token_ids)] = 1
                token_type_ids[:len(token_ids)] = 0

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_token_type_ids.append(token_type_ids)
                all_mask_position_ids.append(mask_position_id)
                all_conn_ids.append(conn_id)
                all_label_ids.append(label_id)
                all_possible_label_ids.append(possible_label_ids)

        assert len(all_input_ids) == len(all_attention_mask), (len(all_input_ids), len(all_attention_mask))
        assert len(all_input_ids) == len(all_token_type_ids), (len(all_input_ids), len(all_token_type_ids))
        assert len(all_input_ids) == len(all_mask_position_ids), (len(all_input_ids), len(all_mask_position_ids))
        assert len(all_input_ids) == len(all_conn_ids), (len(all_input_ids), len(all_conn_ids))
        assert len(all_input_ids) == len(all_label_ids), (len(all_input_ids), len(all_label_ids))
        assert len(all_input_ids) == len(all_possible_label_ids), (len(all_input_ids), len(all_possible_label_ids))

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.token_type_ids = all_token_type_ids
        self.mask_position_ids = all_mask_position_ids
        self.conn_ids = all_conn_ids
        self.label_ids = np.array(all_label_ids)
        self.possible_label_ids = np.array(all_possible_label_ids)
        self.total_size = len(self.input_ids)
        label_frequency = sorted(label_frequency.items(), key=lambda x:x[0])
        print(label_frequency)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.token_type_ids[index]),
            torch.tensor(self.mask_position_ids[index]),
            torch.tensor(self.conn_ids[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.possible_label_ids[index])
        )

