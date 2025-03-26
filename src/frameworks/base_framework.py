from abc import ABC, abstractmethod
from src.configs import FrameworkConfigs, DataConfigs
import os


class BaseFramework(ABC):
    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        self.framework_configs = framework_configs
        self.data_configs = data_configs
        self.model = model
        self.kwargs = kwargs

        self.original_prompt = ""

        data_instruction_path = os.path.join(data_configs.data_dir, self.framework_configs.name.lower(), "instruction.txt")  
        with open(data_instruction_path, 'r') as f:
            for line in f.readlines():
                self.original_prompt += line
        
        self.original_prompt += '\n'
        
        data_examples_path = os.path.join(data_configs.data_dir, self.framework_configs.name.lower(), "examples.txt")  
        with open(data_examples_path, 'r') as f:
            for line in f.readlines():
                self.original_prompt += line
        
        if framework_configs.name == "ReAct" and self.model.model_configs.name == "LLaMA3-8b-Instruct":
            self.original_prompt += "\nExamples finished. Your thoughts should reason about the observations. Follow the exact format like shown in the examples. The following question is the one you need to answer:"
        
        if framework_configs.name == "OneR" and self.model.model_configs.name == "LLaMA3-8b-Instruct":
            self.original_prompt += "\nExamples finished. Follow the exact format like shown in the examples. The following question is the one you need to answer:"
        
        if framework_configs.name == "Direct" and self.model.model_configs.name == "LLaMA3-8b-Instruct":
            self.original_prompt += "\nExamples finished. Follow the exact format like shown in the examples. The following question is the one you need to answer:"
        
    @abstractmethod
    def generate(self):
        pass
