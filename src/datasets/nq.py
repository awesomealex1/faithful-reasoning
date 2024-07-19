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


class NQ(BaseDataset):
    available_variations = {
        "oracle": "nq-open-oracle.jsonl.gz",
        "gold_at_0": "nq-open-30_total_documents_gold_at_0.jsonl.gz",
        "gold_at_4": "nq-open-30_total_documents_gold_at_4.jsonl.gz",
        "gold_at_9": "nq-open-30_total_documents_gold_at_9.jsonl.gz",
        "gold_at_14": "nq-open-30_total_documents_gold_at_14.jsonl.gz",
        "gold_at_19": "nq-open-30_total_documents_gold_at_19.jsonl.gz",
        "gold_at_24": "nq-open-30_total_documents_gold_at_24.jsonl.gz",
        "gold_at_29": "nq-open-30_total_documents_gold_at_29.jsonl.gz",
    }

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.data_filename = os.path.join(
            self.data_dir, self.available_variations[data_configs.variation]
        )

        # Prepare data
        self.data = self.parse_data()

    @staticmethod
    def concat(title, content):
        return f"(Title: {title.strip()}) {content.strip()}"

    def parse_data(self) -> List[dict]:
        # Open the gz file, and read the jsonl file
        data = []

        with gzip.open(self.data_filename, "rb") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                data += [
                    {
                        "idx": i,
                        "question": instance["question"],
                        "answers": instance["answers"],
                        "contexts": [
                            self.concat(context["title"], context["text"])
                            for context in instance["ctxs"]
                        ],
                    }
                ]

        return data

    def build_prompt(self, contexts, question):
        """
        Prompt design (From Lost in the Middle):
        Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

        {search_results}

        Question: {question}
        Answer:
        """

        instruction = [
            "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
        ]

        prompted_contexts = "\n".join(
            [f"Document [{i+1}]{context}" for i, context in enumerate(contexts)]
        )

        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction + [f"{prompted_contexts}\n\nQuestion: {question}\nAnswer: "]
            ]
            return input_text_prompt
        else:
            input_text_prompt = (
                instruction[0]
                + "\n\n"
                + (f"{prompted_contexts}\n\nQuestion: {question}\nAnswer: ")
            )
            return input_text_prompt

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample["prompted_question"] = self.build_prompt(
            sample["contexts"], sample["question"]
        )

        return sample

    def __len__(self):
        return len(self.data)
