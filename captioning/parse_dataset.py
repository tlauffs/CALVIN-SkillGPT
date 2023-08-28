import os
import numpy as np
from r3m import load_r3m
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch
import shutil


def r3m_preprocess(n_px=224):
    return Compose([
        ToPILImage(mode='RGB'),
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])


path = '/home/tim/'
src_folder= '{}calvin_debug_dataset/training'.format(path)
parse_folder= '{}calvin_debug_dataset_parsed/training'.format(path)

device = "cuda" if torch.cuda.is_available() else "cpu"
r3m = load_r3m("resnet50")
r3m.eval()
r3m.to(device)
transform =  r3m_preprocess()

data_episodes = [f for f in os.listdir(src_folder) if f.startswith('episode_')]


ann_source = "{}/lang_annotations".format(src_folder)
ann_dest = "{}/lang_annotations".format(parse_folder)

shutil.copytree(ann_source, ann_dest)

for data_episode in data_episodes:
    data_episode_path = os.path.join(src_folder, data_episode)
    data = np.load(data_episode_path, allow_pickle=True)
    static = transform(data['rgb_static'].astype(np.uint8)) * 255.0
    gripper = transform(data['rgb_gripper'].astype(np.uint8)) * 255.0
    static_transform = torch.tensor(r3m(static.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
    gripper_transform = torch.tensor(r3m(gripper.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
    
    #debug
    print(data['rgb_static'].shape)
    print(data['rgb_gripper'].shape)
    print(static_transform.shape)
    print(gripper_transform.shape)


    processed_npz_path = os.path.join(parse_folder, data_episode)
    np.savez(processed_npz_path, rgb_static=static_transform, rgb_gripper=gripper_transform, actions=data['actions'])
