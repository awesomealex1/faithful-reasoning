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
        prompted_question = self.original_prompt + "\nQuestion: " + question
        model_input = {"prompted_question": [prompted_question], "verbalised_instruction": [""], "prompted_question_wo_context": [""]}
        output = self.model.generate(model_input)
        return output




