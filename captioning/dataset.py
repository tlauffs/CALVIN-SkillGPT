import numpy as np
import glob
import os

from pathlib import Path
import cv2
import numpy as np
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from enum import Enum
from typing import Optional
from r3m import load_r3m
import clip
import config as CFG

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

'''
def r3m_preprocess(n_px=224):
    return Compose([
        ToPILImage(mode='RGB'),
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])
'''

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class CustomDataset(Dataset):
    def __init__(self, data_path, caption_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_path = data_path
        self.caption_data = self.load_caption_data(caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.r3m = load_r3m("resnet50")
        self.r3m.eval()
        self.r3m.to(self.device)
        self.transform =  r3m_preprocess()
        '''

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

        tokens = self.tokenizer.encode(caption, add_special_tokens=True, max_length=self.max_seq_length, truncation=True)

        padding_length = self.max_seq_length - len(tokens)
        if padding_length > 0:
            tokens += [-1] * padding_length            
        gpt_tokens = torch.tensor(tokens)

        gpt_mask = gpt_tokens.ge(0)
        gpt_tokens[~gpt_mask] = 0
        gpt_mask = gpt_mask.float()

       
       # clip_text_features = clip_text_encoder(clip.tokenize(instruction).to(device)).detach().cpu().numpy()

        start_epi = annotation[0][0]

        actions = torch.zeros(64, 7) 
        observations = torch.zeros(64, 2048) 

        """
        j = 0
        for i in range(0, 63):
            epi_num = str(start_epi + i).zfill(7)
            file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
            data = np.load(file_path)
            actions[i] = torch.tensor(data['actions'])   
            if(i == 0 or i == 15 or i == 31 or i == 47 or i == 63):
                observations[j] = torch.tensor(data['rgb_static'])
                observations[j+5] = torch.tensor(data['rgb_gripper'])
                j += 1
        """
        for i in range(0, 63):
            epi_num = str(start_epi + i).zfill(7)
            file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
            data = np.load(file_path)
            actions[i] = torch.tensor(data['actions'])   
            observations[i] = torch.tensor(data['rgb_static'])



        return AttrDict({'gpt_tokens': gpt_tokens, 'gpt_mask': gpt_mask, 'instruction': caption,'actions': actions,'observations': observations})
    