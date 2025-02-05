from abc import ABC
from src.configs import FrameworkConfigs
import re

class BaseMetric(ABC):

    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        **kwargs,
    ):
        self.framework_configs = framework_configs
        self.kwargs = kwargs

    def answer_extractor(self, answer: str) -> str:
        if self.framework_configs.name == "ReAct":
            pattern = r'Finish\[(.*?)\]'

            match = re.search(pattern, answer)

            if match:
                answer = match.group(1)
                return answer
            
            return answer
        else:
            # Adapted from https://github.com/StonyBrookNLP/ircot/blob/main/evaluate.py

            cot_regex = re.compile(".* answer is:? (.*)\\.?")
            match = cot_regex.match(answer)
            if match:
                output = match.group(1)
                if output.endswith("."):
                    output = output[:-1]
                return output

            return answer 