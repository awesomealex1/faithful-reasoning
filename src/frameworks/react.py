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
                r, reasoning_chain = self.alfworld_run(ex1=self.d[f'react_{v}_1'], ex2=self.d[f'react_{v}_0'], ob=ob)
                self.rs[i] += r
                self.cnts[i] += 1
                break
        print('r', r, 'rs', self.rs, 'cnts', self.cnts, 'sum(rs)/sum(cnts)', sum(self.rs) / sum(self.cnts))
        print('------------\n')
        return reasoning_chain
    
    def alfworld_run(self, ex1, ex2, to_print=True, ob=''):
        #prompt = [original_prompt, 'If an action fails repeatedly (returns "Nothing happens"), try: 1. Alternative action syntax 2. Different target locations 3. Re-examine the environment description. Follow the exact format like shown in the examples. Be concise and to the point. The problem is solvable and will only end once you solved it. You can do the following actions when you are not thinking: 1. go to, 2. open, 3. close, 4. put, 5. take, 6. cool, 7. heat, 8. use.\nHere is the task.\n' + ob + '\n>']
        p1 = '''
You are an intelligent agent designed to solve household tasks in the ALFWorld environment. Your responses must strictly adhere to the format specified below.

CRITICAL: Response Format

For EVERY turn, your response MUST contain both a think step and an action step in this exact format. Do not separate them into different turns.
Code snippet

> think: [Your reasoning about the current objective and why you are choosing the next action.]
OK.
> [action] [target]

RULES:

    NEVER write conversational text like "I will..." or "My apologies...".

    ALWAYS start your response with > think:.

    ALWAYS follow the think line with OK. on a new line.

    ALWAYS follow the OK. line with > [action] [target] on a new line.

    NEVER output only a think step. Every response must be a complete think/OK/action block.

Valid Actions

    go to [target]

    open [target]

    close [target]

    take [item] from [location]

    put [item] in/on [location]

    cool [item] with [appliance]

    heat [item] with [appliance]

    use [item]

Task Solving Strategy

1. Analyze the Task:

    Identify the exact items and quantities needed.

    Identify the target location for placement.

    Form a plan: Find item 1 -> Place item 1 -> Find item 2 -> etc.

2. Smart Search Strategy:
Use your knowledge of typical item locations to search efficiently.

    Bathroom items (e.g., soapbar, towel, toiletpaper): sinkbasin, cabinet, toilet, handtowelholder, towelholder

    Kitchen items (e.g., plate, mug, knife, apple): countertop, cabinet, drawer, fridge, microwave

    Bedroom/Living Room items (e.g., cellphone, book, remotecontrol, pillow): coffeetable, sidetable, sofa, bed, dresser, desk

3. Systematic Execution:

    Think before acting: Your think step must state your immediate goal and reasoning.

    Check containers: Always open closed doors, drawers, and cabinets in promising locations.

    Be efficient: Check all likely spots in one room before moving to another.
    
Here are two examples: \n''' + ex1 + '\n' + ex2

        p2 = '\nHere is the task: \n' + ob + '\n'
        prompt = [p1, p2]
        reasoning_chain = []
        if to_print:
            print(ob)
            sys.stdout.flush()
        for i in range(1, 50):
            action_dict = self.model.generate({"prompted_question": [prompt], "verbalised_instruction": [p1]}, stop_strings=['\n'])
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
