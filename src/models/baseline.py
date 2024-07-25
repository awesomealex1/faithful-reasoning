from typing import List, Optional, Tuple, Union

import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel


class Baseline(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

    def _get_component_lengths(self, inputs, tokenised_inputs):
        print(inputs)
        if self.model_configs.model_type == "instruct":
            bos_length = 1
            # FIXME: 5 is <|begin_of_text|><|start_header_id|>user<|end_header_id|> in llama3-8b-instruct tokenizer
            instruction_tokens = self._verbalise_input(
                inputs["verbalised_instruction"][0]
            )[:, 5:]
            print("instruction_tokens: ", instruction_tokens)
            instruction_length = instruction_tokens.shape[-1]
            icl_demo_tokens = self._verbalise_input(inputs["verbalised_icl_demo"][0])[
                :, 5:
            ]
            print("icl_demo_tokens: ", icl_demo_tokens)
            icl_demo_length = icl_demo_tokens.shape[-1]
            contexts_tokens = self._verbalise_input(inputs["verbalised_contexts"][0])[
                :, 5:
            ]
            print("contexts_tokens: ", contexts_tokens)
            contexts_length = contexts_tokens.shape[-1]
            question_tokens = self._verbalise_input(inputs["verbalised_question"][0])[
                :, 5:
            ]
            print("question_tokens: ", question_tokens)
            question_length = question_tokens.shape[-1]
            answer_prefix_tokens = self._verbalise_input(
                inputs["verbalised_answer_prefix"][0]
            )[:, 5:]
            print("answer_prefix_tokens: ", answer_prefix_tokens)
            answer_prefix_length = answer_prefix_tokens.shape[-1]
        else:
            bos_length = 1
            # Start from 1 to skip the BOS token
            instruction_length = self._verbalise_input(
                inputs["verbalised_instruction"]
            )[:, 1:].shape[-1]
            icl_demo_length = self._verbalise_input(inputs["verbalised_icl_demo"])[
                :, 1:
            ].shape[-1]
            contexts_length = self._verbalise_input(inputs["verbalised_contexts"])[
                :, 1:
            ].shape[-1]
            question_length = self._verbalise_input(inputs["verbalised_question"])[
                :, 1:
            ].shape[-1]
            answer_prefix_length = self._verbalise_input(
                inputs["verbalised_answer_prefix"]
            )[1:].shape[-1]

        print("tokenised_inputs: ", tokenised_inputs)

        print("bos_length: ", bos_length)
        print("answer_prefix_length: ", answer_prefix_length)
        print("question_length: ", question_length)
        print("contexts_length: ", contexts_length)
        print("icl_demo_length: ", icl_demo_length)
        print("instruction_length: ", instruction_length)
        print(
            "sum: ",
            bos_length
            + answer_prefix_length
            + question_length
            + contexts_length
            + icl_demo_length
            + instruction_length,
        )
        print("tokenised_inputs.size(1): ", tokenised_inputs.size(1))
        assert (
            bos_length
            + answer_prefix_length
            + question_length
            + contexts_length
            + icl_demo_length
            + instruction_length
            == tokenised_inputs.size(1)
        ), "Tokenised inputs length does not match the sum of the lengths of the components"

        return {
            "bos": bos_length,
            "instruction": instruction_length,
            "icl_demo": icl_demo_length,
            "contexts": contexts_length,
            "question": question_length,
            "answer_prefix": answer_prefix_length,
        }

    def get_lookback_ratios(self, attentions, component_lengths, new_token_start_from):
        print(component_lengths)

        components = list(component_lengths.keys())
        # Define component order and initialize lookback ratio tensors
        num_layers = len(attentions[0])
        num_heads = attentions[0][0].shape[1]
        new_token_length = len(attentions)

        # Initialize lookback ratio tensors
        lookback_ratios = {
            comp: torch.zeros((num_layers, num_heads, new_token_length))
            for comp in components
        }
        lookback_ratios["new_tokens"] = torch.zeros(
            (num_layers, num_heads, new_token_length)
        )

        for i in range(new_token_length):
            for l in range(num_layers):
                curr_length = 0
                attn_sums = []

                # Calculate attention for each component
                for comp, length in component_lengths.items():
                    attn = attentions[i][l][
                        0, :, -1, curr_length : curr_length + length + 1
                    ].mean(-1)
                    lookback_ratios[comp][l, :, i] = attn
                    attn_sums.append(attn)
                    curr_length += length

                # Validate new token start
                assert (
                    new_token_start_from == curr_length
                ), "Mismatch in the length of the components"

                # Calculate attention for new tokens
                attn_new_tokens = attentions[i][l][
                    0, :, -1, new_token_start_from:
                ].mean(-1)
                lookback_ratios["new_tokens"][l, :, i] = attn_new_tokens
                attn_sums.append(attn_new_tokens)

                # Normalize ratios
                attn_sum = sum(attn_sums)
                for comp in lookback_ratios:
                    lookback_ratios[comp][l, :, i] /= attn_sum

        return lookback_ratios

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        # Calculate the length of each component
        component_lengths = self._get_component_lengths(inputs, tokenised_inputs)

        # Predict
        with torch.inference_mode():
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], use_cache=True, return_dict=True
            )
            generated_ids = []
            attentions = []
            last_input_token = tokenised_inputs[:, -1]
            past_kv = input_logits.past_key_values
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                    attn_mode="torch",
                )
                attentions += [outputs.attentions]
                past_kv = outputs.past_key_values
                last_input_token = outputs.logits[0, -1].argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        generation_output = {"decoded_text": decoded_text, "attentions": {}}
        if return_attentions:
            generation_output["attentions"] = self.get_lookback_ratios(
                attentions, component_lengths, tokenised_inputs.size(1)
            )

        return generation_output

    def lm_score(
        self,
        prompt,
        answer,
    ):
        prompt = prompt["prompted_question"][0]

        with torch.no_grad():
            if type(prompt) == list:
                input_text = prompt + [answer]
            else:
                input_text = prompt + answer
            input_ids = self._verbalise_input(input_text).to(self.model.device)
            prefix_ids = self._verbalise_input(prompt).to(self.model.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]

            outputs = self.model(input_ids)[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs
