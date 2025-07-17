import torch
from torch.utils.data import Dataset
from src.configs import DataConfigs

import alfworld.agents.environment as alf_env
import alfworld.agents.modules.generic as generic
import numpy as np
import os
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

class ALFWorldDataset(Dataset):
    def __init__(self, data_configs: DataConfigs, **kwargs):
        self.data_configs = data_configs
        self.num_episodes = data_configs.num_samples if data_configs.num_samples > 0 else 100  # default
        self.batch_size = kwargs.get('batch_size', 1)
        config_file = os.path.expanduser("configs/alfworld_base_config.yaml")
        with patch_sys_argv([sys.argv[0], config_file]):
            self.config = generic.load_config()
        self.env_type = 'AlfredTWEnv'  # text-based
        #self.env = alf_env.get_environment(self.env_type)(self.config, train_eval='eval_out_of_distribution')
        #self.env = self.env.init_env(batch_size=self.batch_size)
        #self.episodes = []
        #self._generate_episodes()

    def _generate_episodes(self):
        for _ in range(self.num_episodes):
            obs, info = self.env.reset()
            done = [False] * self.batch_size
            episode = []
            while not all(done):
                admissible_commands = list(info['admissible_commands'])
                # For now, take random actions from admissible commands
                actions = [np.random.choice(cmds) if len(cmds) > 0 else 'look' for cmds in admissible_commands]
                next_obs, scores, done, info = self.env.step(actions)
                episode.append({
                    'observation': obs,
                    'action': actions,
                    'reward': scores,
                    'done': done,
                    'info': info
                })
                obs = next_obs
            self.episodes.append(episode)
        print(self.episodes)

    def __getitem__(self, idx):
        print(idx)
        print(self.episodes[idx])
        # Return the full episode for now
        return self.episodes[idx]

    def __len__(self):
        return len(self.episodes) 