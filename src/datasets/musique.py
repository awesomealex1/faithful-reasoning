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

        self.data_filename = os.path.join(self.data_dir, "musique_ans_v1.0_dev.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                if instance["answerable"]:
                    sample_id = instance["id"]
                    sample_type = sample_id.split("__")[0]
                    data += [
                        {
                            "idx": sample_id,
                            "type": sample_type,
                            "question": instance["question"],
                            "answer": [instance["answer"]] + instance["answer_aliases"],
                        }
                    ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        print(data)

        return data

    def create_demo_text(self) -> List[str]:
        # 8 Train QA Pairs: https://github.com/StonyBrookNLP/ircot/blob/main/prompts/musique/no_context_cot_qa_flan_t5.txt
        questions, answers = [], []

        questions.append("When was Neville A. Stanton's employer founded?")
        answers.append(
            "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
        )

        questions.append(
            "What is the headquarters for the organization who sets the standards for ISO 21500?"
        )
        answers.append(
            "The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is: Geneva."
        )

        questions.append(
            "What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?"
        )
        answers.append(
            "Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So the answer is: Central Jakarta."
        )

        questions.append(
            "When was the first railway line constructed between Kotri and the city where Marie Adelaide Leprosy Centre is located?"
        )
        answers.append(
            "Marie Adelaide Leprosy Centre is located in Karachi. The first railway line between Kotri and Karachi was constructed in April 1858. So the answer is: April 1858."
        )

        questions.append(
            "What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?"
        )
        answers.append(
            "Heritage Places Protection Act applies to the jurisdiction of Prince Edward Island. Hebron, Prince Edward Island is located in the Prince County. So the answer is: Prince County."
        )

        questions.append(
            "When did the first large winter carnival take place in the city where CIMI-FM is licensed to broadcast?"
        )
        answers.append(
            "CIMI-FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City took place in 1894. So the answer is: 1894."
        )

        questions.append(
            "When did the first large winter carnival happen in Olivier Robitaille's place of birth?"
        )
        answers.append(
            "Olivier Robitaille was born in Quebec City. The first large winter carnival in Quebec City happened in the 1894. So the answer is: 1894."
        )

        questions.append("When did Britain withdraw from the country containing Hoora?")
        answers.append(
            "Hoora is in the country of Bahrain. Britain withdrew from Bahrain in 1971. So the answer is: 1971."
        )

        questions.append(
            "When did Britain withdraw from the country where the village of Wadyan is found?"
        )
        answers.append(
            "Wadyan is in the country of Bahrain. Britain withdraw from Bahrain in 1971. So the answer is: 1971."
        )

        questions.append(
            "What did the publisher of Banjo-Tooie rely primarily on for its support?"
        )
        answers.append(
            "The publisher of Banjo-Tooie is Nintendo. Nintendo relied primarily for its support on first-party games. So the answer is: first-party games."
        )

        questions.append(
            "What shares a border with Rivière-Verte in the province WRSU-FM broadcasts in?"
        )
        answers.append(
            "WRSU-FM was licensed to broadcast to New Brunswick. Rivière-Verte, New Brunswick shares border with Edmundston. So the answer is: Edmundston."
        )

        questions.append(
            "When was the state of emergency declared in the country where the Senate is located?"
        )
        answers.append(
            "The Senate is in the country of Kenya. The state of emergency was declared in Kenya on 20 October 1952. So the answer is: 20 October 1952."
        )

        questions.append(
            "Where is the crying stone found in the country in which Raphael Tuju holds citizenship?"
        )
        answers.append(
            "Raphael Tuju is a citizen of Kenya. The crying stone in Kenya is found along the highway towards Kisumu. So the answer is: along the highway towards Kisumu."
        )

        questions.append(
            "Where does the Snake River start, in the state where Lima Mountain is located?"
        )
        answers.append(
            "Lima Mountain is located in the state of Minnesota. The snake river in Minnesota starts in southern Aitkin County. So the answer is: southern Aitkin County."
        )

        questions.append(
            "What genre is the record label of the performer of So Long, See You Tomorrow associated with?"
        )
        answers.append(
            "The performer of So Long, See You Tomorrow is Bombay Bicycle Club. The record label of Bombay Bicycle Club is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        )

        questions.append(
            "In which county was the birthplace of the Smoke in tha City performer?"
        )
        answers.append(
            "The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County. So the answer is: Los Angeles County."
        )

        questions.append(
            "What is the genre of the record label of the band that performed on the Crush Tour?"
        )
        answers.append(
            "The Crush Tour is performed by the band Bon Jovi. The record label of Bon Jovi is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        )

        questions.append(
            "How long is the US border with the country that borders the state where Finding Dory takes place?"
        )
        answers.append(
            "Finding Dory is supposed to take place in California. The country that shares a border with California is Mexico. The length of the us border with Mexico is 1,989 mi. So the answer is: 1,989 mi."
        )

        questions.append(
            "What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?"
        )
        answers.append(
            "The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald. So the answer is: Yale Herald."
        )

        questions.append(
            "How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran-Iraq war?"
        )
        answers.append(
            "Pacific National University is located in Khabarovsk, Russia Khabarovsk, Russian is in the continent of Asia. The entity that mediated the truce which ended the Iran-Iraq War is the UN. The number of member states that UN recognises in Asia is 53. So the answer is: 53."
        )

        demo_texts = []
        if self.kwargs["use_chat_template"]:
            for i in range(len(questions)):
                demo_texts += [
                    f"Q: \n{questions[i]}\nA:",
                    answers[i],
                ]
        else:
            for i in range(len(questions)):
                demo_texts += [f"Q: {questions[i]}\nA: {answers[i]}"]
        return demo_texts

    def build_prompt(self, question):
        instruction = ["Answer the following question by reasoning step-by-step."]

        icl_demo = self.create_demo_text()

        verbalised_question = f"Q: \n{question}\n"
        answer_prefix = "Answer:"
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction + icl_demo + [f"{verbalised_question}{answer_prefix}"]
            ]
        else:
            instruction = instruction[0]
            icl_demo = "\n\n".join(icl_demo) + "\n\n"
            input_text_prompt = (
                instruction + icl_demo + (f"{verbalised_question}{answer_prefix}")
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

        # For attention analysis
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
