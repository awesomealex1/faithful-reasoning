from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.configs import DecoderConfigs, ModelConfigs
from src.utils.modelling_llama import LlamaConfig, LlamaForCausalLM


class BaseModel(ABC):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        self.model = LlamaForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            use_flash_attention_2="flash_attention_2",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_configs.configs.model_name_or_path
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_seq_len = model_configs.configs.max_seq_len
        self.max_new_tokens = model_configs.configs.max_new_tokens

        self.model_configs = model_configs
        self.decoder_configs = decoder_configs

    def _verbalise_input(
        self,
        inputs: Union[list, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None,
        use_system_prompt: bool = True,
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        if tokenizer is None:
            tokenizer = self.tokenizer

        if self.model_configs.model_type == "instruct":
            chat_inputs = []
            if type(inputs) == list:
                for idx, input in enumerate(inputs):
                    if type(input) in [tuple, list]:
                        input = input[0]
                    if idx == 0 and use_system_prompt:
                        chat_inputs += [{"role": "system", "content": input}]
                    else:
                        if idx % 2 != 0:
                            chat_inputs += [{"role": "user", "content": input}]
                        else:
                            chat_inputs += [{"role": "assistant", "content": input}]
            else:
                if type(inputs) in [tuple, list]:
                    inputs = inputs[0]
                chat_inputs += [{"role": "user", "content": inputs}]
            inputs = tokenizer.apply_chat_template(
                chat_inputs,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                max_length=self.max_seq_len,
            )

        elif self.model_configs.model_type == "base":
            inputs = tokenizer(
                inputs,
                return_tensors="pt",
                max_length=self.max_seq_len,
            ).input_ids
        else:
            raise ValueError(
                f"Unknown model type: {self.model_configs.model_type}. "
                "Terminate tokenisation process."
            )

        return inputs

    @abstractmethod
    def generate(self, logits):
        pass

    @abstractmethod
    def lm_score(self, logits):
        pass
