from torch.utils.data import Dataset
import numpy as np
import os
import pdb
import tensorflow_datasets as tfds
from torchvision import transforms
import clip
import config as CFG
import torch
from utils.utils import AttrDict

class BCDataset(Dataset):

    def __init__(self, data_path, caption_path): 

        self.data_path = data_path
        self.caption_data = self.load_caption_data(caption_path)
        self.data_files = [f for f in os.listdir(data_path) if f.startswith('episode_')]

        self.transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize((CFG.img_size)),
            transforms.ToTensor() 
        ])
        self.device = CFG.device
        clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=True)
        self.clip_text_encoder = clip_model.encode_text


        #self.action_stats = AttrDict(mean=np.array([0.0001016613095998764, 0.0026315886061638594], dtype=np.float32),
        #                             std=np.array([0.008755197748541832, 0.010186241939663887]), dtype=np.float32)

    #    self.action_stats = AttrDict(mean=np.array([ 0.05244775, -0.12080607, 0.50815218, 1.01496132, -0.03902264, 1.56418701, 0.13438409], dtype=np.float32),
     #                                std=np.array([0.15992226, 0.10983621, 0.0623301, 2.90982278, 0.10183952, 0.34633791, 0.99092932]), dtype=np.float32)
                                    
        self.action_stats = AttrDict(mean=np.array([ 0.05829782, -0.11443839,  0.5123094,   0.97989569, -0.03639337,  1.55281177, 0.01424668], dtype=np.float32),
                                     std=np.array([0.1525908,  0.09533093, 0.05158806, 2.920689, 0.10144497, 0.56220544, 0.99989851]), dtype=np.float32)

    def __len__(self):
        return len(self.caption_data)
    
    def load_caption_data(self, caption_path):
        annotations = np.load(f"{caption_path}", allow_pickle=True).item()
        annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))
        return annotations

    def __getitem__(self, idx):
        annotation = self.caption_data[idx]
        instruction = annotation[1]
        start_epi = annotation[0][0]

        with torch.no_grad():
            clip_text_features = self.clip_text_encoder(clip.tokenize(instruction).to(self.device)).detach().cpu().numpy()[0]

        im_idx = np.random.randint(0, 64)
        epi_num = str(start_epi + im_idx).zfill(7)
        
        file_path = os.path.join(self.data_path, "episode_{}.npz".format(epi_num))
        data = np.load(file_path)
        img_static = data['rgb_static']
        img_gripper = data['rgb_gripper']
        action = data['actions']
        rel_action = data['rel_actions'].astype(np.float32)
        robot_obs = data['robot_obs'].astype(np.float32)
        # normalize action
        action = ((action - self.action_stats.mean) / self.action_stats.std).astype(np.float32)
        img_static = self.transform(img_static)
        img_gripper = self.transform(img_gripper)

        return AttrDict(img_static=img_static,
                        img_gripper=img_gripper,
                        robot_obs=robot_obs,
                        action=action,
                        rel_action=rel_action,
                        text_encoding=clip_text_features
                        )

    def _sample_seq(self):
        return np.random.choice(self.dataset)


    
        
