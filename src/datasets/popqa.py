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
                    "answers": ds[i]["possible_answers"],
                }
            ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_demo_text(self, curr_question) -> List[str]:
        questions, answers = [], []

        # First 9 instances from the test set, remove 1 if encountered as the current question
        questions.append("What is George Rankin's occupation?")
        answers.append("politician")

        questions.append("What is John Mayne's occupation?")
        answers.append("journalist")

        questions.append("What is Henry Feilden's occupation?")
        answers.append("politician")

        questions.append("What is Kathy Saltzman's occupation?")
        answers.append("politician")

        questions.append("What is Eleanor Davis's occupation?")
        answers.append("cartoonist")

        questions.append("What is Alexander Rinnooy Kan's occupation?")
        answers.append("mathematician")

        questions.append("What is Scooter Braun's occupation?")
        answers.append("talent manager")

        questions.append("What is Leona Detiège's occupation?")
        answers.append("politician")

        questions.append("What is William Murray, 1st Earl of Mansfield's occupation?")
        answers.append("politician")

        if curr_question in questions:
            # Remove current question from the list if it is already in the list
            idx = questions.index(curr_question)
            _ = questions.pop(idx)
            _ = answers.pop(idx)
        else:
            # Only take 8 examples
            _ = questions.pop()
            _ = answers.pop()

        if self.kwargs["use_chat_template"]:
            demo_texts = [
                "Answer the following question based on the provided context:"
            ]
            for i in range(len(questions)):
                demo_texts += [
                    f"Question: {questions[i]}\nAnswer:",
                    answers[i],
                ]
        else:
            # Concatenate demonstration examples ...
            demo_texts = [
                "Answer the following question based on the provided context:"
            ]
            for i in range(len(questions)):
                demo_texts += [f"Question: {questions[i]}\nAnswer: {answers[i]}"]
        return demo_texts

    def build_prompt(self, question):
        instruction = ["Answer the given question."]

        icl_demo = self.create_demo_text(question)

        verbalised_question = f"Q: {question}\n"
        answer_prefix = "A:"
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction + icl_demo + [f"{verbalised_question}{answer_prefix}"]
            ]
        else:
            instruction = instruction[0]
            icl_demo = "\n\n".join(icl_demo)
            input_text_prompt = (
                instruction
                + "\n\n"
                + icl_demo
                + "\n\n"
                + (f"{verbalised_question}{answer_prefix}")
            )
        return {
            "verbalised_instruction": instruction,
            "verbalised_icl_demo": icl_demo,
            "verbalised_contexts": "",
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = self.build_prompt(sample["question"])

        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]

        return sample

    def __len__(self):
        return len(self.data)
