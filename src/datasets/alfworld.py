import torch
from torch.utils.data import Dataset
from src.configs import DataConfigs

import alfworld.agents.environment as alf_env
import alfworld.agents.modules.generic as generic
import numpy as np
import os

class ALFWorldDataset(Dataset):
    def __init__(self, data_configs: DataConfigs, **kwargs):
        self.data_configs = data_configs
        self.num_episodes = data_configs.num_samples if data_configs.num_samples > 0 else 100  # default
        self.batch_size = kwargs.get('batch_size', 1)
        config_file = getattr(data_configs, 'alfworld_config_file', None)
        if config_file is None:
            config_file = os.path.expanduser("~/.cache/alfworld/configs/base_config.yaml")
        self.config = generic.load_config(config_file=config_file)

        self.env_type = 'AlfredTWEnv'  # text-based
        self.env = alf_env.get_environment(self.env_type)(self.config, train_eval='train')
        self.env = self.env.init_env(batch_size=self.batch_size)
        self.episodes = []
        self._generate_episodes()

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

    def __getitem__(self, idx):
        # Return the full episode for now
        return self.episodes[idx]

    def __len__(self):
        return len(self.episodes) 