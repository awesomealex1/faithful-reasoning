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
        super().__init__(framework_configs, model, kwargs)
        self.max_steps = framework_configs.configs.max_steps
        
        instruction = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \
                            (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\
                            (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage. \
                            (3) Finish[answer], which returns the answer and finishes the task. \
                            Here are some examples."
        
        data_prompt_path = os.path.join(data_configs.data_dir, "prompt.txt")  
        self.corpus_name = data_configs.configs.name.lower()
        with open(data_prompt_path, 'r') as f:
            dataset_prompt = f.readline()
        
        self.original_prompt = instruction + dataset_prompt
        self.retriever = ElasticsearchRetriever()
        

    def generate(self, input):
        return self.do_react(input)
    
    def do_react(self, input):
        step_i = 0
        prompt = self.original_prompt + "\nQuestion: " + input + "\n"
        title = None

        while step_i < self.max_steps:
            step_i += 1
            prompt += f"Thought {step_i}:"
            print("\n########")
            print(prompt)
            reasoning_action = self.reason(prompt, f"Observation {step_i}:")     # reasoning is of format  xxxxxx\n Act i: Action[param]\n
            print(reasoning_action)
            if self.contains_answer(reasoning_action):
                return self.extract_answer(reasoning_action)

            observation, title = self.act(reasoning_action, title)   # observation is some retrieved context\n
            print(observation)
            prompt += reasoning_action + f"Observation {step_i}: {observation}\n"
        
        return "No answer found"
    
    def reason(self, prompt, stop):
        output = self.model(prompt)[0]["generated_text"]

        if output.find(stop):
            output = output[:output.find(stop)]

        output.rstrip()
        output += '\n'

        return output

    def act(self, input):
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
    
    def extract_answer(self, reasoning_action):
        pattern = r'Finish\[(.*?)\]'

        match = re.search(pattern, reasoning_action)

        if match:
            answer = match.group(1)
            return answer
        
        raise ValueError("Passed an action that doesn't contain an answer")


class HotpotReAct(ReAct):

    def __init__(self, model, max_steps):
        self.model = model
        self.max_steps = max_steps

        instruction = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \
                            (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\
                            (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage. \
                            (3) Finish[answer], which returns the answer and finishes the task. \
                            Here are some examples."
        prompt_file = 'src/frameworks/prompts_naive.json'
        with open(prompt_file, 'r') as f:
            prompt_dict = json.load(f)

        webthink_examples = prompt_dict['webthink_simple6']
        self.webthink_prompt = instruction + webthink_examples
        self.retriever = ElasticsearchRetriever()


    def do_react(self, question):
        step_i = 0
        prompt = self.webthink_prompt + "\nQuestion: " + question + "\n"
        title = None

        while step_i < self.max_steps:
            step_i += 1
            prompt += f"Thought {step_i}:"
            print("\n########")
            print(prompt)
            reasoning_action = self.reason(prompt, f"Observation {step_i}:")     # reasoning is of format  xxxxxx\n Act i: Action[param]\n
            print(reasoning_action)
            if self.contains_answer(reasoning_action):
                return self.extract_answer(reasoning_action)

            observation, title = self.act(reasoning_action, title)   # observation is some retrieved context\n
            print(observation)
            prompt += reasoning_action + f"Observation {step_i}: {observation}\n"
        
        return "No answer found"

    def reason(self, prompt, stop):
        output = self.model(prompt)[0]["generated_text"]

        if output.find(stop):
            output = output[:output.find(stop)]

        output.rstrip()
        output += '\n'

        return output
    
    def act(self, reasoning_action, title):
        action_type, action_value = self.extract_action(reasoning_action)

        if action_type == "Search":
            try:
                observation = self.retriever.retrieve_titles(
                    corpus_name="hotpotqa",
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
                    corpus_name="hotpotqa", 
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
                corpus_name="hotpotqa",
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
    
    def extract_answer(self, reasoning_action):
        pattern = r'Finish\[(.*?)\]'

        match = re.search(pattern, reasoning_action)

        if match:
            answer = match.group(1)
            return answer
        
        raise ValueError("Passed an action that doesn't contain an answer")
