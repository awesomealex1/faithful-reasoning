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

        # Create a temporary directory to hold the game
        game_dir = tempfile.gettempdir()

        # Generate a simple game: get a key and open a door
        game_file, game_id = textworld.generation.make_game_with(
            quest_length=1,
            nb_rooms=5,
            nb_objects=10,
            quest_breadth=1,
            grammar=None,
            seed=42,
            path=game_dir,
            force_rebuild=True
        )

        # Register game
        request_infos = textworld.EnvInfos(admissible_commands=True, won=True, lost=True, description=True, inventory=True)
        env_id = textworld.gym.register_games([game_file], request_infos=request_infos, max_episode_steps=50)

        # Create the environment
        self.env = gym.make(env_id)
        self.env.reset()  # Initial reset
    
    def generate(self):
        _input = {}
        reasoning_chain = self.do_react()
        decoded_text = "\n".join(reasoning_chain).rstrip()
        _input["reasoning_chain"] = reasoning_chain
        _input["decoded_text"] = decoded_text
        return _input

    def do_react(self):
        ob, info = self.env.reset()
        reasoning_chain = []

        r, reasoning_chain = self.alfworld_run(ob=ob)
        return reasoning_chain
    
    def alfworld_run(self, ex1=None, ex2=None, to_print=True, ob=''):
        prompt = ['Here is the task:\n' + ob + '\n']
        reasoning_chain = []

        if to_print:
            print(ob)
            sys.stdout.flush()

        for i in range(1, 50):
            action_dict = self.model.generate({"prompted_question": [prompt], "verbalised_instruction": [""]}, stop_strings=['\n'])
            action = action_dict["decoded_text"].removeprefix('>').strip()

            obs, reward, done, info = self.env.step(action)

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