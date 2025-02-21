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


class MuSiQue(BaseDataset):
    """
    Closed-book MuSiQue
    """

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(self.data_dir, "test_subsampled.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                data += [
                    {
                        "idx": instance["question_id"],
                        "question": instance["question_text"],
                        "answers": instance["answers_objects"][0]["spans"]
                    }
                ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def __len__(self):
        return len(self.data)
