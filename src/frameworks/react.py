from src.frameworks.base_framework import BaseFramework
from src.configs import FrameworkConfigs, DataConfigs
import os
from torch.utils.data import Dataset
from src.configs import DataConfigs
import json
import alfworld.agents.environment as alf_env
import alfworld.agents.modules.generic as generic
import numpy as np
import sys
from contextlib import contextmanager
import textworld.gym
import gym
import textworld
import tempfile
import wandb


examples = {
  "react_fetch_1": "You are in a kitchen. You see a fridge, a table, and a cupboard.\nYour task is to: fetch an apple and place it on the table.\n> think: To solve the task, I need to find an apple, take it, and then put it on the table.\nOK.\n> think: Apples are likely found in the fridge or cupboard. I will check the fridge first.\nOK.\n> go to fridge\nYou are now in front of the fridge. It is closed.\n> open fridge\nYou open the fridge. Inside, you see an apple and a bottle of water.\n> take apple from fridge\nYou take the apple from the fridge.\n> go to table\nYou go to the table. It is empty.\n> put apple on table\nYou put the apple on the table.",
  
  "react_fetch_0": "You are in a kitchen. You see a pantry, a counter, and a trashcan.\nYour task is to: fetch a banana and place it on the counter.\n> think: I need to locate a banana, take it, and place it on the counter.\nOK.\n> think: Bananas might be found in the pantry. I’ll start there.\nOK.\n> go to pantry\nYou arrive at the pantry. It is closed.\n> open pantry\nYou open the pantry. Inside, you see a banana and a cereal box.\n> take banana from pantry\nYou take the banana.\n> go to counter\nYou walk to the counter.\n> put banana on counter\nYou put the banana on the counter."
}

@contextmanager
def patch_sys_argv(new_args):
    old_argv = sys.argv
    sys.argv = new_args
    try:
        yield
    finally:
        sys.argv = old_argv

class ReAct(BaseFramework):

    def __init__(
        self,
        framework_configs: FrameworkConfigs,
        data_configs: DataConfigs,
        model,
        **kwargs,
    ):
        super().__init__(framework_configs, data_configs, model , **kwargs)
        self.data_configs = data_configs
        self.i = 0
        self.total_r = 0
        self.count = 0
        
    
    def generate(self):
        _input = {}
        reasoning_chain = self.do_react()
        decoded_text = "\n".join(reasoning_chain).rstrip()
        _input["reasoning_chain"] = reasoning_chain
        _input["decoded_text"] = decoded_text
        return _input

    def do_react(self):
        self.i += 1
        self.env_id = textworld.gym.register_game(f"tw_games/custom_game_{self.i}.z8",
                                     max_episode_steps=50)

        self.env = textworld.gym.make(self.env_id)  # Start the environment.
        ob, info = self.env.reset()
        reasoning_chain = []

        r, reasoning_chain = self.alfworld_run(ob=ob)

        self.count += 1
        self.total_r += r

        wandb.log({"acc": self.total_r/self.count})
        
        return reasoning_chain
    
    def alfworld_run(self, ex1=None, ex2=None, to_print=True, ob=''):
        p1 = 'You are a ReAct agent, who needs to solve Text World problems. You need to think and act. Here are some examples: \n' + examples["react_fetch_0"] + "\n" + examples["react_fetch_1"] + "\n"
        p2 = 'Here is the task:\n' + ob + '\n'
        prompt = [p1, p2]
        reasoning_chain = []

        if to_print:
            print(ob)
            sys.stdout.flush()

        for i in range(1, 50):
            action_dict = self.model.generate({"prompted_question": [prompt], "verbalised_instruction": [p1], "prompted_question_wo_context": [p2]}, stop_strings=['\n'])
            action = action_dict["decoded_text"].removeprefix('>').strip()

            obs, reward, done, info = self.env.step(action)
            if action.startswith('think:'):
                observation = 'OK.'
            if to_print:
                print(f'Act {i}: {action}\nObs {i}: {obs}')
                sys.stdout.flush()

            reasoning_chain.append(f'Act {i}: {action}')
            reasoning_chain.append(f'\nObs {i}: {obs}')
            prompt.append(f' {action}\n')
            prompt.append(f' {obs}\n>')

            if done:
                return reward, reasoning_chain

        return 0, reasoning_chain

    
    def process_ob(self, ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]    
        return ob