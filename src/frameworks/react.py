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
        super().__init__(framework_configs, data_configs, model , **kwargs)
        self.max_steps = framework_configs.max_steps
        self.corpus_name = data_configs.name.lower()
        self.retriever = ElasticsearchRetriever()

    def generate(self, _input):
        decoded_text = self.do_react(_input)
        _input["decoded_text"] = decoded_text
        return _input
    
    def do_react(self, input):
        prompt = self.original_prompt + "\n\nQuestion: " + input["question"][0] + "\n"
        title = None
        i = 1

        try:
            while i <= self.max_steps:
                prompt += f"Thought {i}: "

                if i == 1:
                    prompt_wo_context = prompt

                thought = self.reason(prompt, prompt_wo_context, f"Observation {i}:")
                prompt += f"{thought}\n"
                
                if self.contains_answer(thought):
                    print("###### START")
                    print(prompt[len(self.original_prompt)+2:].rstrip())
                    print("###### END")
                    return prompt[len(self.original_prompt)+2:].rstrip()
                
                action_type, action_value = self.extract_action(thought)
                observation, title = self.act(action_type, action_value, title)

                prompt += f"Observation {i}: {observation}\n"
                prompt_wo_context = thought + f"Observation {i}: Could not retrieve new context."

                i += 1

        except Exception as e:
            print(e)
        
        print(prompt[len(self.original_prompt)+2:])
        return "No answer found"
    
    def reason(self, prompt, prompt_wo_context, stop):
        _input = {"prompted_question": [prompt], "verbalised_instruction": [""], "prompted_question_wo_context": [prompt_wo_context]}
        output = self.model.generate(_input)
        output = output["decoded_text"]
        print(output)

        if stop in output:
            output = output[:output.find(stop)]

        output = output.rstrip()
        return output

    def act(self, action_type, action_value, title):
        if action_type == "Search":
            try:
                observation = self.retriever.retrieve_titles(
                    corpus_name=self.corpus_name,
                    query_text=action_value
                )
                exact_title_matches = [obs for obs in observation if obs["title"] == action_value]
                if len(exact_title_matches) > 0:    # Check for exact title match first. Investigate if exact title always is more similar?
                    title = exact_title_matches[0]["title"]
                    observation_val = exact_title_matches[0]["paragraph_text"]
                else:
                    title = observation[0]["title"]
                    observation_val = observation[0]["paragraph_text"]
            except NotFoundError:   # If no good title is found, retrieve similar paragraphs and return the titles of those paragraphs.
                observation = self.retriever.retrieve_paragraphs(
                    corpus_name=self.corpus_name, 
                    query_text=action_value,
                    max_hits_count=5
                )
                similar_titles = [doc["title"] for doc in observation]
                prefix = f"Could not find [{action_value}]. "
                if len(observation) > 0:
                    observation_val = prefix + f"Titles containing similar information: {similar_titles}."
                else:
                    observation_val = prefix + "No similar entries found."
        elif action_type == "Lookup":   # Only for hotpot
            observation = self.retriever.retrieve_first_paragraph_with_keyword(
                corpus_name=self.corpus_name,
                query_text=action_value,
                page_title=title
            )
            if len(observation) > 0:
                title = observation[0]["title"]
                observation_val = observation[0]["paragraph_text"]
            else:
                observation_val = f"Could not find [{action_value}] on page."
        else:
            return "Did not pass a valid action.", title

        return observation_val, title
    
    def extract_action(self, thought):
        search_pattern = r'Search\[(.*?)\]'
        match = re.search(search_pattern, thought)

        if match:
            action_value = match.group(1)
            return "Search", action_value

        lookup_pattern = r'Lookup\[(.*?)\]'
        match = re.search(lookup_pattern, thought)

        if match:
            action_value = match.group(1)
            return "Lookup", action_value
        
        raise ValueError("Passed an action that doesn't contain search or lookup")
    
    def contains_answer(self, thought):
        return "Finish[" in thought