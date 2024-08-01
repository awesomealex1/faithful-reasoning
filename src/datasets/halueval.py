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


class HaluEval(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        pass

    def create_demo_text(self) -> List[str]:
        pass

    def build_prompt(self, question):
        return {
            "verbalised_instruction": None,
            "verbalised_icl_demo": None,
            "verbalised_contexts": None,
            "verbalised_question": None,
            "verbalised_answer_prefix": None,
            "prompted_question": None,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = self.build_prompt(sample["question"])

        # For attention analysis
        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]

        return sample

    def __len__(self):
        return len(self.data)
