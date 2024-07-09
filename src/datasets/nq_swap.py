import gzip
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset, load_from_disk

from src.configs import DataConfigs, DecoderConfigs
from src.datasets.base_dataset import BaseDataset


# FIXME: Still copy pasted from TruthfulQA
class NQSwap(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []
        ds = load_dataset(self.data_dir)
        print(ds)

        for i in range(len(ds["validation"])):
            print(ds["validation"][i])
            data += [
                {
                    "idx": i,
                    "question": ds["validation"][i]["question"],
                    "org_context": ds["validation"][i]["org_context"],
                    "org_answer": ds["validation"][i]["org_answer"],
                    "sub_context": ds["validation"][i]["sub_context"],
                    "sub_answer": ds["validation"][i]["sub_answer"],
                }
            ]

        return data

    def build_prompt(self, sub_context, question):
        input_text_prompt = f"Context: {sub_context}\nQuestion: {question}\nAnswer:"
        return input_text_prompt

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample["prompted_question"] = self.build_prompt(
            sample["sub_context"], sample["question"]
        )
        sample["sub_answer"] = (
            sample["sub_answer"][0]
            if type(sample["sub_answer"]) == list
            else sample["sub_answer"]
        )

        return sample

    def __len__(self):
        return len(self.data)
