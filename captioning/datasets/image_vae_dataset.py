import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import pdb
import h5py
import config as CFG
from transformers import GPT2Tokenizer
from torchvision import transforms
import tensorflow_datasets as tfds
import os
from utils.utils import AttrDict

BICUBIC = Image.BICUBIC


# create a pytorch dataset
class ImageVAEDataset(Dataset):
    def __init__(self, dataset_path=None, caption_path=None):

        self.data_path = dataset_path
        self.data_files = [f for f in os.listdir(dataset_path) if f.startswith('episode_')]
        self.caption_data = self.load_caption_data(caption_path)
        
        #builder = tfds.builder_from_directory(dataset_path)
        #if phase == 'train':
        #    self.ds = builder.as_dataset(split='train', shuffle_files=True)

        #self.ids = iter(self.ds)

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((CFG.img_size)),
            transforms.ToTensor() 
        ])

    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        return annotations
    
    def __len__(self):
        # return self.ds.__len__().numpy()
        return len(self.caption_data)
    

    def __getitem__(self, idx):
        annotation = self.caption_data[idx]
        start_epi = annotation[0][0]


        im_idx = np.random.randint(0, 64)
        epi_num = str(start_epi + im_idx).zfill(7)
        file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
        data = np.load(file_path)
        img_static = data['rgb_static']
        img_gripper = data['rgb_gripper']

        img_static = self.transform(img_static)
        img_gripper = self.transform(img_gripper)

        #tensor_img = self.transform(img)
        # tensor_img = torch.cat((img_static, img_gripper), dim=0)
        return AttrDict({"img_static": img_static, "img_gripper": img_gripper})
    
        """
        file_name = self.data_files[idx]
        episode_path = os.path.join(self.data_path, file_name)
        data = np.load(episode_path, allow_pickle=True)
        img = data["rgb_gripper"]
        tensor_img = self.transform(img)
        return AttrDict({"image": tensor_img})
        """
    

class ImageVAEDatasetFull(Dataset):
    def __init__(self, dataset_path=None):

        self.data_path = dataset_path
        self.data_files = [f for f in os.listdir(dataset_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((CFG.img_size)),
            transforms.ToTensor() 
        ])
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):    
        file_name = self.data_files[idx]
        episode_path = os.path.join(self.data_path, file_name)
        data = np.load(episode_path, allow_pickle=True)
        img_static = data["rgb_static"]
        img_static = self.transform(img_static)
        img_gripper = data["rgb_gripper"]
        img_gripper = self.transform(img_gripper)
        return AttrDict({"img_static": img_static, "img_gripper": img_gripper})
