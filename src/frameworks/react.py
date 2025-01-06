from src.utils import retriever
import json
import re

class ReAct():

    def __init__(self, model):
        self.model = model

    def generate(self, input):
        return self.do_react(input)
    
    def do_react(self, input):
        raise NotImplementedError("Didnt implement react step")
    
    def reason(self, input):
        raise NotImplementedError("Didn't implement reasoning step")

    def act(self, input):
        raise NotImplementedError("Didn't implement acting step")
    
    def stop_react(self, input):
        raise NotImplementedError("Didn't implement stop react")


class MuSiQueReAct(ReAct):

    def __init__(self, model, retriever, max_steps):
        self.model = model
        self.retriever = retriever
        self.max_steps = max_steps

        instruction = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \
                            (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\
                            (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage. \
                            (3) Finish[answer], which returns the answer and finishes the task. \
                            Here are some examples."
        prompt_file = 'prompts_naive.json'
        with open(prompt_file, 'r') as f:
            prompt_dict = json.load(f)

        webthink_examples = prompt_dict['webthink_simple6']
        self.webthink_prompt = instruction + webthink_examples
    
    def do_react(self, question):
        step_i = 0
        prompt = self.webthink_prompt + "\nQuestion: " + question + "\n"

        while step_i < self.max_steps:
            step_i += 1
            prompt += f"Thought {step_i}:"
            reasoning_action = self.reason(prompt, f"Observation {step_i}:")     # reasoning is of format  xxxxxx\n Act i: Action[param]\n

            if self.contains_answer(reasoning_action):
                return self.extract_answer(reasoning_action)

            observation = self.act(reasoning_action)   # observation is some retireved context\n
            prompt += reasoning_action + f"Observation {step_i}: {observation}"
        
        return None

    def reason(self, prompt, stop):
        output = self.model.generate(prompt)

        if output.find(stop):
            output = output[:output.find(stop)]

        output.rstrip()
        output += '\n'

        return output
    
    def act(self, reasoning_action):
        action_type, action_value = self.extract_action(reasoning_action)

        if action_type == "Search":
            observation = self.retriever.search(action_value)
        elif action_type == "Lookup":
            observation = self.retriever.lookup(action_value)

        return observation
    
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