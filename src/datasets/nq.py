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
        "closed_book": "nq-open-oracle.jsonl.gz",
        "gold_at_0": "nq-open-10_total_documents_gold_at_0.jsonl.gz",
        "gold_at_4": "nq-open-10_total_documents_gold_at_4.jsonl.gz",
        "gold_at_9": "nq-open-10_total_documents_gold_at_9.jsonl.gz",
    }

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

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
                if self.variation == "closed_book":
                    contexts = []
                else:
                    contexts = [
                        self.concat(context["title"], context["text"])
                        for context in instance["ctxs"]
                    ]

                data += [
                    {
                        "idx": i,
                        "question": instance["question"],
                        "answers": instance["answers"],
                        "contexts": contexts,
                    }
                ]

        return data

    def build_prompt(self, contexts, question):

        instruction = [
            "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Provide the answer in 5 words or less without any explanation.\n\n"
            + (
                "Document [1](Title: Example Title) Example Text\n\n"
                if len(contexts)
                else ""
            )
            + "Question: example question\nAnswer: march 2018"
        ]

        prompted_contexts = "\n".join(
            [f"Document [{i+1}]{context}" for i, context in enumerate(contexts)]
        )
        if prompted_contexts:
            prompted_contexts += "\n\n"

        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction + [f"{prompted_contexts}Question: {question}\nAnswer: "]
            ]
            return input_text_prompt
        else:
            input_text_prompt = (
                instruction[0]
                + "\n\n"
                + (f"{prompted_contexts}Question: {question}\nAnswer: ")
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
