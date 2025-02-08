from src.utils.retriever_server.elasticsearch_retriever import ElasticsearchRetriever
import json
import re
from elasticsearch.exceptions import NotFoundError
from src.frameworks.base_framework import BaseFramework
from src.configs import FrameworkConfigs, DataConfigs
import os

class ReAct(BaseFramework):

    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        super().__init__(framework_configs, data_configs, model, **kwargs)
        self.max_steps = framework_configs.configs.max_steps
        self.corpus_name = data_configs.name.lower()
        self.retriever = ElasticsearchRetriever()

    def generate(self, _input):
        decoded_text = self.do_react(_input)
        _input["decoded_text"] = decoded_text
        return _input
    
    def do_react(self, input):
        step_i = 0
        prompt = self.original_prompt + "\nQuestion: " + input["question"][0] + "\n"
        title = None

        try:

            while step_i < self.max_steps:
                step_i += 1
                prompt += f"Thought {step_i}:"
                print("\n########")
                print(prompt)
                if step_i == 1:
                    prompt_wo_context = prompt
                reasoning_action = self.reason(prompt, prompt_wo_context, f"Observation {step_i}:")     # reasoning is of format  xxxxxx\n Act i: Action[param]\n
                print("RA:",reasoning_action)
                if self.contains_answer(reasoning_action):
                    return reasoning_action

                observation, title = self.act(reasoning_action, title)   # observation is some retrieved context\n
                print(observation)
                prompt += reasoning_action + f"Observation {step_i}: {observation}\n"
                prompt_wo_context = reasoning_action + f"Observation {step_i}: No new context."

        except Exception as e:
            print(e)
        
        return "No answer found"
    
    def reason(self, prompt, prompt_wo_context, stop):
        _input = {"prompted_question": [prompt], "verbalised_instruction": [""], "prompted_question_wo_context": [prompt_wo_context]}
        output = self.model.generate(_input)
        output = output["decoded_text"]

        if stop in output:
            output = output[:output.find(stop)]

        output = output.rstrip()
        output += '\n'


        return output

    def act(self, input, title):
        action_type, action_value = self.extract_action(input)

        if action_type == "Search":
            try:
                observation = self.retriever.retrieve_titles(
                    corpus_name=self.corpus_name,
                    query_text=action_value
                )
                exact_title_matches = [obs for obs in observation if obs["title"] == action_value]
                if len(exact_title_matches) > 0:
                    title = exact_title_matches[0]["title"]
                    observation_val = exact_title_matches[0]["paragraph_text"]
                else:
                    title = observation[0]["title"]
                    observation_val = observation[0]["paragraph_text"]
            except NotFoundError:
                observation = self.retriever.retrieve_paragraphs(
                    corpus_name=self.corpus_name, 
                    query_text=action_value,
                    max_hits_count=5
                )
                similar_titles = [doc["title"] for doc in observation]
                prefix = f"Could not find [{action_value}]. "
                if len(observation) > 0:
                    observation_val = prefix + f"Similar: {similar_titles}"
                else:
                    observation_val = prefix + "No similar entries found."
        elif action_type == "Lookup":
            observation = self.retriever.retrieve_first_paragraph_with_keyword(
                corpus_name=self.corpus_name,
                query_text=action_value,
                page_title=title
            )
            if len(observation) > 0:
                title = observation[0]["title"]
                observation_val = observation[0]["paragraph_text"]
            else:
                observation_val = f"Could not find [{action_value}]. "

        return observation_val, title
    
    def extract_action(self, reasoning_action):
        search_pattern = r'Search\[(.*?)\]'

        match = re.search(search_pattern, reasoning_action)

        if match:
            action_value = match.group(1)
            return "Search", action_value

        lookup_pattern = r'Lookup\[(.*?)\]'

        match = re.search(lookup_pattern, reasoning_action)

        if match:
            action_value = match.group(1)
            return "Lookup", action_value
        
        raise ValueError("Passed an action that doesn't contain search or lookup")
    
    def contains_answer(self, reasoning_action):
        return "Finish[" in reasoning_action