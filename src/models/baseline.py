from typing import List, Optional, Tuple, Union

import torch

from src.configs import DecoderConfigs, ModelConfigs
from src.models.base_model import BaseModel
from src.models.utils import merge_attention_weights


class Baseline(BaseModel):
    def __init__(
        self,
        model_configs: ModelConfigs,
        decoder_configs: DecoderConfigs,
    ):
        super().__init__(model_configs, decoder_configs)

    def generate(
        self,
        inputs,
        return_attentions: bool = False,
    ) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)

        print(inputs)
        if self.model_configs.model_type == "instruct":
            bos_length = 1
            verbalised_question = self._verbalise_input(
                inputs["verbalised_question"][0]
            )
            question_length = self._verbalise_input(inputs["verbalised_question"][0])[
                :, 6:
            ].shape[
                -1
            ]  # FIXME: 6 is <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n in llama3-8b-instruct tokenizer
            context_length = tokenised_inputs.size(1) - question_length - bos_length
        else:
            bos_length = 1
            question_length = self._verbalise_input(
                inputs["verbalised_question"]
            ).shape[-1]
            context_length = tokenised_inputs.size(1) - question_length - bos_length
        print("bos_length: ", bos_length)
        print("question_length: ", question_length)
        print("context_length: ", context_length)

        print("tokenised_inputs[:bos_length]: ", tokenised_inputs[:, :bos_length])
        print(
            "tokenised_inputs[bos_length:context_length]: ",
            tokenised_inputs[:, bos_length : context_length + 1],
        )
        print(
            self.tokenizer.decode(tokenised_inputs[0][bos_length : context_length + 1])
        )
        print()
        print(
            "tokenised_inputs[bos_length+context_length:tokenised_inputs.size(1)]: ",
            tokenised_inputs[:, bos_length + context_length : tokenised_inputs.size(1)],
            self.tokenizer.decode(
                tokenised_inputs[0][
                    bos_length + context_length : tokenised_inputs.size(1)
                ]
            ),
        )
        print(
            "bos_length+context_length+question_length: ",
            bos_length + context_length + question_length,
        )
        print("tokenised_inputs.size(1): ", tokenised_inputs.size(1))
        exit()

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
                # print(outputs.attentions.size())
                attentions += [outputs.attentions]
                past_kv = outputs.past_key_values
                last_input_token = outputs.logits[0, -1].argmax()
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        generation_output = {"decoded_text": decoded_text}
        if return_attentions:
            attentions = merge_attention_weights(attentions)

            # print(attentions)

            context_length = attentions[0][0].shape[-1]
            new_token_length = len(attentions)
            num_layers = len(attentions[0])
            num_heads = attentions[0][0].shape[1]

            generation_output["attentions"] = {}
            lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
            for i in range(len(attentions)):  # iterating over the new tokens length
                for l in range(num_layers):
                    attn_on_bos = attentions[i][l][0, :, -1, 0].mean(-1)
                    attn_on_context = attentions[i][l][0, :, -1, :context_length].mean(
                        -1
                    )
                    attn_on_question = attentions[i][l][0, :, -1, context_length:].mean(
                        -1
                    )
                    attn_on_new_tokens = attentions[i][l][
                        0, :, -1, context_length:
                    ].mean(-1)
                    bos_lookback_ratio[l, :, i] = attn_on_bos / (
                        attn_on_bos
                        + attn_on_context
                        + attn_on_question
                        + attn_on_new_tokens
                    )
                    context_lookback_ratio[l, :, i] = attn_on_context / (
                        attn_on_bos
                        + attn_on_context
                        + attn_on_question
                        + attn_on_new_tokens
                    )
                    question_lookback_ratio[l, :, i] = attn_on_question / (
                        attn_on_bos
                        + attn_on_context
                        + attn_on_question
                        + attn_on_new_tokens
                    )
                    new_tokens_lookback_ratio[l, :, i] = attn_on_new_tokens / (
                        attn_on_bos
                        + attn_on_context
                        + attn_on_question
                        + attn_on_new_tokens
                    )

            generation_output["attentions"]["bos_lookback_ratio"] = bos_lookback_ratio
            generation_output["attentions"][
                "context_lookback_ratio"
            ] = context_lookback_ratio
            generation_output["attentions"][
                "question_lookback_ratio"
            ] = question_lookback_ratio
            generation_output["attentions"][
                "new_tokens_lookback_ratio"
            ] = new_tokens_lookback_ratio

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
