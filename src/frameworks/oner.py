from abc import ABC, abstractmethod
from src.configs import FrameworkConfigs, DataConfigs
from src.frameworks.base_framework import BaseFramework
import os
from src.utils.retriever_server.elasticsearch_retriever import ElasticsearchRetriever

class OneR(BaseFramework):
    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        super().__init__(framework_configs, data_configs, model, **kwargs)
        self.corpus_name = data_configs.name.lower()
        self.retriever = ElasticsearchRetriever()

    def generate(self, _input):
        question = _input["question"][0]

        observation = self.retriever.retrieve_paragraphs(
                    corpus_name=self.corpus_name, 
                    query_text=question,
                    max_hits_count=5
        )

        prompted_question_wo_context = self.original_prompt + "\nQuestion: " + question
        prompted_question = prompted_question_wo_context + "\nContext: "

        for par in observation:
            par_text = par["paragraph_text"]
            prompted_question += par_text + " "
        
        print(prompted_question)
        
        model_input = {"prompted_question": [prompted_question], "verbalised_instruction": [""], "prompted_question_wo_context": [prompted_question_wo_context]}
        output = self.model.generate(model_input)
        return output




