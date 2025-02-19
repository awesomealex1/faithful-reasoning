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
        
    @abstractmethod
    def generate(self):
        pass
