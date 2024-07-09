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

        for i in range(len(ds["validation"])):
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

    def create_demo_text(self):
        def build_prompt_with_answer(sub_context, question, sub_answer):
            input_text_prompt = (
                "Answer the following question based on the provided context:\n\n"
                f"Context: {sub_context}\nQuestion: {question}\nAnswer: {sub_answer}"
            )
            return input_text_prompt

        contexts, questions, answers = [], [], []

        contexts.append(
            'The phrase "What happens in Vegas, stays in Vegas" was coined by the advertising agency R&R Partners as part of a marketing campaign for the Las Vegas Convention and Visitors Authority. This slogan was introduced in 2003 and has since become synonymous with the city\'s image of offering a place for visitors to indulge without repercussions.'
        )
        questions.append(
            "Who coined the phrase 'What happens in Vegas stays in Vegas'?"
        )
        answers.append("R&R Partners, an ad agency")

        contexts.append(
            "Johannes Gutenberg, a German blacksmith, goldsmith, printer, and publisher, introduced printing to Europe with his mechanical movable-type printing press. His invention played a key role in the spread of the Renaissance, Reformation, and the Scientific Revolution, laying the material basis for the modern knowledge-based economy and the spread of learning to the masses. Gutenberg started the mass production of printed books in the 15th century, which included the famous Gutenberg Bible."
        )
        questions.append("Who began the mass printing of Bibles five centuries ago?")
        answers.append("Johannes Gutenberg")

        contexts.append(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair, it is 324 meters tall, about the same height as an 81-story building."
        )
        questions.append("Who designed the Eiffel Tower?")
        answers.append("Gustave Eiffel")

        contexts.append(
            "The Great Wall of China is a series of fortifications that were built across the northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined by Qin Shi Huang, the first Emperor of China. Little of the Qin wall remains. Later on, many successive dynasties have repaired, rebuilt, and expanded sections of the wall."
        )
        questions.append(
            "Which emperor is credited with connecting the Great Wall of China?"
        )
        answers.append("Qin Shi Huang")

        # Concatenate demonstration examples ...
        demo_text = ""
        for i in range(len(questions)):
            demo_text += (
                build_prompt_with_answer(contexts[i], questions[i], answers[i]) + "\n\n"
            )
        return demo_text

    def build_prompt(self, sub_context, question):
        demo = self.create_demo_text()
        input_text_prompt = demo + (
            "Answer the following question based on the provided context:\n\n"
            f"Context: {sub_context}\nQuestion: {question}\nAnswer:"
        )
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
