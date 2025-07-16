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
        self.num_episodes = data_configs.num_samples if data_configs.num_samples > 0 else 100  # default
        config_file = os.path.expanduser("configs/alfworld_base_config.yaml")
        with patch_sys_argv([sys.argv[0], config_file]):
            self.config = generic.load_config()
        self.env_type = 'AlfredTWEnv'  # text-based
        self.env = alf_env.get_environment(self.env_type)(self.config, train_eval='eval_out_of_distribution')
        self.env = self.env.init_env(batch_size=1)
        self.cnts = [0] * 6
        self.rs = [0] * 6
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }
        self.prompt_file = 'alfworld_3prompts.json'
        with open(self.prompt_file, 'r') as f:
            self.d = json.load(f)
    
    def generate(self):
        _input = {}
        reasoning_chain = self.do_react()
        decoded_text = "\n".join(reasoning_chain).rstrip()
        _input["reasoning_chain"] = reasoning_chain
        _input["decoded_text"] = decoded_text
        return _input

    def do_react(self):
        ob, info = self.env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        reasoning_chain = []
        for i, (k, v) in enumerate(self.prefixes.items()):
            if name.startswith(k):
                prompt = 'Interact with a household to solve a task. Here are two examples.\n<EXAMPLE START>\n' + self.d[f'react_{v}_1'] + self.d[f'react_{v}_0'] + '\n<EXAMPLE END>'
                r, reasoning_chain = self.alfworld_run(prompt, ob=ob)
                self.rs[i] += r
                self.cnts[i] += 1
                break
        print('r', r, 'rs', self.rs, 'cnts', self.cnts, 'sum(rs)/sum(cnts)', sum(self.rs) / sum(self.cnts))
        print('------------\n')
        return reasoning_chain
    
    def alfworld_run(self, prompt, to_print=True, ob=''):
        original_prompt = prompt
        prompt = [original_prompt, 'If an action fails repeatedly (returns "Nothing happens"), try: 1. Alternative action syntax 2. Different target locations 3. Re-examine the environment description. Follow the exact format like shown in the examples. Be concise and to the point. The problem is solvable and will only end once you solved it. You can do the following actions when you are not thinking: 1. go to, 2. open, 3. close, 4. put, 5. take, 6. cool, 7. heat, 8. use.\nHere is the task.\n' + ob + '\n>']
        print("PROMPTTTTT", prompt)
        reasoning_chain = []
        if to_print:
            print(ob)
            sys.stdout.flush()
        for i in range(1, 50):
            action_dict = self.model.generate({"prompted_question": [prompt], "verbalised_instruction": [original_prompt]}, stop_strings=['\n'])
            action = action_dict["decoded_text"].removeprefix('>').strip()
            print("ACTION", action, "        ", action_dict["decoded_text"])    
            observation, reward, done, info = self.env.step([action])
            observation, reward, done = self.process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith('think:'):
                observation = 'OK.'
            if to_print:
                print(f'Act {i}: {action}\nObs {i}: {observation}')
                sys.stdout.flush()
            reasoning_chain.append(f'Act {i}: {action}')
            reasoning_chain.append(f'\nObs {i}: {observation}')
            prompt.append(f' {action}\n')
            prompt.append(f' {observation}\n>')
            if done:
                return reward, reasoning_chain
        return 0, reasoning_chain
    
    def process_ob(self, ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]    
        return ob