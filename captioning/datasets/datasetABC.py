import random
import numpy as np
import os
from pathlib import Path
import numpy as np
import numpy as np
from sklearn.manifold import TSNE
from r3m import load_r3m
import torch
from torch.utils.data import Dataset
from utils.utils import AttrDict


class ABCDataset(Dataset):
    def __init__(self, data_path, caption_path, env_info_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_path = data_path
        self.env_info =  np.load(env_info_path, allow_pickle=True).item()
        self.caption_data = self.load_caption_data(caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]
        self.tasks_per_scene  = self.load_indicies_for_tasks(caption_path)

    def __len__(self):
        return len(self.caption_data)
        # return len(self.data_files)
    
    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"], annotations["language"]["task"]))
        return annotations

    def load_indicies_for_tasks(self, caption_path):
        tasks = np.load(f"{caption_path}", allow_pickle=True).item()
        tasks = list(zip(tasks["info"]["indx"], tasks["language"]["task"]))
        tasks_per_scene = {}
        for task in tasks:
            scene = ''
            if task[0][1] <= self.env_info['calvin_scene_B'][1]:
                scene = 'B'
            elif task[0][1] <= self.env_info['calvin_scene_C'][1]:
                scene = 'C'
            elif task[0][1] <= self.env_info['calvin_scene_A'][1]:
                scene = 'A'
            if scene == '':
                continue
            if task[1] not in tasks_per_scene:
                tasks_per_scene[task[1]] = {}
            if scene not in tasks_per_scene[task[1]]:
                tasks_per_scene[task[1]][scene] = []
            tasks_per_scene[task[1]][scene].append(task[0])     
        return tasks_per_scene


    def __getitem__(self, idx):
        annotation = self.caption_data[idx]

        task = annotation[2]
        caption = annotation[1]
        caption_index = annotation[0]

        tokens = self.tokenizer.encode(caption, add_special_tokens=True, max_length=self.max_seq_length, truncation=True)

        padding_length = self.max_seq_length - len(tokens)
        if padding_length > 0:
            tokens += [-1] * padding_length            
        gpt_tokens = torch.tensor(tokens)

        gpt_mask = gpt_tokens.ge(0)
        gpt_tokens[~gpt_mask] = 0
        gpt_mask = gpt_mask.float()

        start_episodes = {'A': 0,
                          'B': 0,
                          'C': 0}
    
        if caption_index[1] <= self.env_info['calvin_scene_B'][1]:
            scene = 'calvin_scene_B'
            start_episodes['B'] = annotation[0][0]
        elif caption_index[1] <= self.env_info['calvin_scene_C'][1]:
            scene = 'calvin_scene_C'
            start_episodes['C'] = annotation[0][0]
        elif caption_index[1] <= self.env_info['calvin_scene_A'][1]:
            scene = 'calvin_scene_A'
            start_episodes['A'] = annotation[0][0]

        for env in ['A','B','C']:
            if start_episodes[env] is 0:
                start_episodes[env] = random.choice(self.tasks_per_scene[task][env])[0]


        actions = {'A': torch.zeros(64, 7),
                   'B': torch.zeros(64, 7),
                   'C': torch.zeros(64, 7)}
        state = {'A': torch.zeros(64, 15),
                 'B': torch.zeros(64, 15),
                 'C': torch.zeros(64, 15)}
        observations = {'A': torch.zeros(64, 2048),
                        'B': torch.zeros(64, 2048),
                        'C': torch.zeros(64, 2048)}

        for env in ['A','B','C']:
            start_epi = start_episodes[env]
            for i in range(0, 63):
                epi_num = str(start_epi + i).zfill(7)
                file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
                data = np.load(file_path)
                actions[env][i] = torch.tensor(data['actions'])
                state[env][i] = torch.tensor(data['robot_obs'])
                observations[env][i] = torch.tensor(data['observations'])

        return AttrDict({'gpt_tokens': gpt_tokens, 'gpt_mask': gpt_mask, 'instruction': caption,'caption_index': caption_index,'actions': actions,'state': state ,'observations': observations})
    