from itertools import combinations, product
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from src.configs import DecoderConfigs, ModelConfigs, PromptConfigs


class HFModel:
    def __init__(
        self,
        model_configs: ModelConfigs,
        prompt_configs: PromptConfigs,
    ):
        self.model_configs = model_configs
        self.prompt_configs = prompt_configs

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_configs.configs.model_name_or_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     low_cpu_mem_usage=True,
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_configs.configs.model_name_or_path
        # )

        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_seq_len = model_configs.configs.max_seq_len
        self.max_new_tokens = model_configs.configs.max_new_tokens

    def _verbalise_input(self, inputs) -> torch.Tensor:
        pass

    def generate(
        self,
        inputs,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        self.model.eval()

        contextualised_input = self._verbalise_input(inputs)
        contextualised_input = contextualised_input.to(self.model.device)

        # Predict
        with torch.inference_mode():
            output = self.model.generate(
                contextualised_input,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded_text = self.tokenizer.decode(
                output[0, contextualised_input.size(1) :], skip_special_tokens=False
            )

        return decoded_text

    @staticmethod
    def postprocess_prediction(answer):
        """
        TO DO
        """
        return {}
