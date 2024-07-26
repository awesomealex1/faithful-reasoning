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


class PopQA(BaseDataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []
        ds = load_from_disk(self.data_dir)

        for i in range(len(ds)):
            data += [
                {
                    "idx": ds[i]["id"],
                    "subj": ds[i]["subj"],
                    "prop": ds[i]["prop"],
                    "obj": ds[i]["obj"],
                    "subj_id": ds[i]["subj_id"],
                    "prop_id": ds[i]["prop_id"],
                    "obj_id": ds[i]["obj_id"],
                    "s_aliases": ds[i]["s_aliases"],
                    "o_aliases": ds[i]["o_aliases"],
                    "s_uri": ds[i]["s_uri"],
                    "o_uri": ds[i]["o_uri"],
                    "s_wiki_title": ds[i]["s_wiki_title"],
                    "o_wiki_title": ds[i]["o_wiki_title"],
                    "s_pop": ds[i]["s_pop"],
                    "o_pop": ds[i]["o_pop"],
                    "question": ds[i]["question"],
                    "possible_answers": ds[i]["possible_answers"],
                }
            ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_demo_text(self) -> List[str]:
        def build_prompt_with_answer(sub_context, question, sub_answer):
            input_text_prompt = (
                f"Context: {sub_context}\nQuestion: {question}\nAnswer: {sub_answer}"
            )
            return input_text_prompt

        questions, answers = [], []

        questions.append("who was the first band to play at woodstock")
        answers.append("Anna Paquin")

        questions.append("when did the vietnam war end what year")
        answers.append("August")

        questions.append("where did the beatles take the abbey road picture ")
        answers.append("Spice Girls")

        questions.append("who played freddie mercury in the movie bohemian rhapsody")
        answers.append("Erwin Schr\u00f6dinger")

        if self.kwargs["use_chat_template"]:
            demo_texts = [
                "Answer the following question based on the provided context:"
            ]
            for i in range(len(questions)):
                demo_texts += [
                    f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer:",
                    answers[i],
                ]
        else:
            # Concatenate demonstration examples ...
            demo_texts = [
                "Answer the following question based on the provided context:"
            ]
            for i in range(len(questions)):
                demo_texts += [
                    f"Context: {contexts[i]}\nQuestion: {questions[i]}\nAnswer: {answers[i]}"
                ]
        return demo_texts

    def build_prompt(self, sub_context, question):
        if self.kwargs["use_chat_template"]:
            demo = self.create_demo_text()
            input_text_prompt = [
                demo + [f"Context: {sub_context}\nQuestion: {question}\nAnswer:"]
            ]
            return input_text_prompt
        else:
            demo = self.create_demo_text()
            demo = "\n\n".join(demo)
            input_text_prompt = (
                demo
                + "\n\n"
                + (
                    # "Answer the following question based on the provided context:\n\n"
                    f"Context: {sub_context}\nQuestion: {question}\nAnswer:"
                )
            )
            return input_text_prompt

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample["prompted_question"] = self.build_prompt(
            sample["sub_context"], sample["question"]
        )

        return sample

    def __len__(self):
        return len(self.data)
