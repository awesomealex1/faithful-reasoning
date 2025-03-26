from abc import ABC, abstractmethod
from src.configs import FrameworkConfigs, DataConfigs
from src.frameworks.base_framework import BaseFramework
import os

class Direct(BaseFramework):
    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        super().__init__(framework_configs, data_configs, model, **kwargs)

    def generate(self, _input):
        question = _input["question"][0]
        prompt = [self.original_prompt, "Question: " + question]
        
        model_input = {"prompted_question": [prompt], "verbalised_instruction": [self.original_prompt], "prompted_question_wo_context": []}
        output = self.model.generate(model_input)
        _input["decoded_text"] = output["decoded_text"]
        _input["prompted_question"] = prompt
        return _input




