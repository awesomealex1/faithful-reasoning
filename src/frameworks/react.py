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
You are absolutely right. I apologize for the misunderstanding. The back-and-forth showed a fundamental misinterpretation of the interaction model.

Based on your correction, the agent should only output one thing at a time: either a think step OR an action step, but never both. The system's OK. is the cue to switch from thinking to acting.

This requires a completely different prompt structure. Let's discard the previous versions and use this corrected and simplified one, which strictly enforces the two-step process you described.

Final, Correct Prompt (Two-Step Interaction)

You are an intelligent agent solving household tasks in ALFWorld. Your responses must follow an exact turn-based format to function.

CRITICAL: Interaction Flow

The process is a strict, alternating sequence.

    Your 1st Turn (Thinking): After you see the environment state, you MUST respond with only your reasoning.

        Format: > think: [Your reasoning about the objective and next action.]

    System's Response: The system will reply with OK. and then prompt you.

        Format: OK.

        >

    Your 2nd Turn (Acting): After the system's OK. and >, you MUST respond with only the action.

        Format: [action] [target]

This cycle repeats. You think, the system says OK, you act.

RULES:

    NEVER combine a think and an action in the same response.

    NEVER output OK. yourself; that is the system's job.

    NEVER start with conversational text like "I will..." or "I apologize...".

    If the prompt is (observation text), your response is > think: ...

    If the prompt is OK. and >, your response is [action] [target]

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

    Plan: Analyze the task, identify the items and target, and form a step-by-step plan.

    Search: Use knowledge of common item locations (e.g., soapbar in sinkbasin or cabinet; pillow on sofa or bed).

    Execute: Systematically check locations. Always open closed containers.
    
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
