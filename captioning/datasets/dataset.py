import numpy as np
import os
from pathlib import Path
import numpy as np
import numpy as np
from sklearn.manifold import TSNE
from r3m import load_r3m
import torch
from torch.utils.data import Dataset
from utils.util import AttrDict
from transformers import GPT2Tokenizer
import config as CFG

"""
    function to get either d_d or abc_d datasets
"""
def get_dataset(dataset = 'd', observation_data='static_and_gripper'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if dataset == 'abc_d':
        datapath_training_parsed = CFG.datapath_training_abcd_parsed
        datapath_val_parsed = CFG.datapath_val_abcd_parsed
        caption_path_training = '{}/lang_annotations/auto_lang_ann.npy'.format(datapath_training_parsed)
        caption_path_val = '{}/lang_annotations/auto_lang_ann.npy'.format(datapath_val_parsed)
    else:
        datapath_training_parsed = CFG.datapath_training_parsed
        datapath_val_parsed = CFG.datapath_val_parsed
        caption_path_training = '{}/lang_annotations/auto_lang_ann.npy'.format(datapath_training_parsed)
        caption_path_val = '{}/lang_annotations/auto_lang_ann.npy'.format(datapath_val_parsed)
    observation_data = 'static_and_gripper'
    train_dataset = CustomDataset(datapath_training_parsed, caption_path_training, tokenizer, CFG.max_seq_length, observation_data)
    val_dataset  = CustomDataset(datapath_val_parsed, caption_path_val, tokenizer, CFG.max_seq_length, observation_data)
    return train_dataset, val_dataset, datapath_training_parsed, datapath_val_parsed, caption_path_training, caption_path_val


'''
    Dataset for captioning model:
    used in captioning.ipynb
'''
class CustomDataset(Dataset):
    def __init__(self, data_path, caption_path, tokenizer, max_seq_length, observation_data = 'static_and_gripper'):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_path = data_path
        self.caption_data = self.load_caption_data(caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]
        self.observation_data = observation_data

    def __len__(self):
        return len(self.caption_data)
        # return len(self.data_files)
    
    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        return annotations

    def __getitem__(self, idx):
        annotation = self.caption_data[idx]

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
        start_epi = annotation[0][0]

        actions = torch.zeros(64, 7) 
        state = torch.zeros(64, 15) 
        observations = torch.zeros(64, 2048) 

        for i in range(0, 63):
            epi_num = str(start_epi + i).zfill(7)
            file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
            data = np.load(file_path)
            actions[i] = torch.tensor(data['actions'])
            state[i] = torch.tensor(data['robot_obs'])

            if self.observation_data == 'static':
                observations[i] = torch.tensor(data['rgb_static'])
            elif self.observation_data == 'gripper':
                observations[i] = torch.tensor(data['rgb_gripper'])
            else:
                observations[i] = torch.tensor(data['observations'])



        return AttrDict({'gpt_tokens': gpt_tokens, 'gpt_mask': gpt_mask, 'instruction': caption,'caption_index': caption_index,'actions': actions,'state': state ,'observations': observations})
    