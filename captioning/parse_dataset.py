import os
import numpy as np
from r3m import load_r3m
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch
import shutil
import config as CFG

def r3m_preprocess(n_px=224):
    return Compose([
        ToPILImage(mode='RGB'),
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])


#src_folder= CFG.datapath_training
#parse_folder= CFG.datapath_training_parsed

def parse(): 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3m = load_r3m("resnet50")
    r3m.eval()
    r3m.to(device)
    transform =  r3m_preprocess()

    data_episodes = [f for f in os.listdir(src_folder) if f.startswith('episode_')]


    ann_source = "{}/lang_annotations".format(src_folder)
    ann_dest = "{}/lang_annotations".format(parse_folder)

    length = len(data_episodes)

    shutil.copytree(ann_source, ann_dest)

    for idx, data_episode in enumerate(data_episodes):
        data_episode_path = os.path.join(src_folder, data_episode)
        data = np.load(data_episode_path, allow_pickle=True)
        static = transform(data['rgb_static'].astype(np.uint8)) * 255.0
        gripper = transform(data['rgb_gripper'].astype(np.uint8)) * 255.0
        static_transform = torch.tensor(r3m(static.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
        gripper_transform = torch.tensor(r3m(gripper.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
    
    
        #debug
        '''
        print(data['rgb_static'].shape)
        print(data['rgb_gripper'].shape)
        print(static_transform.shape)
        print(gripper_transform.shape)
        '''

        processed_npz_path = os.path.join(parse_folder, data_episode)
        np.savez(processed_npz_path, rgb_static=static_transform, rgb_gripper=gripper_transform, actions=data['actions'])

        if os.name == 'nt': 
            os.system('cls')
        else:
            os.system('clear')
        print(idx, ": ", length)

def add_stop_token(stop_token: str = ' \n'):
    ann_source = "{}/lang_annotations".format(src_folder)
    ann_dest = "{}/lang_annotations".format(parse_folder)
    shutil.rmtree(ann_dest)
    shutil.copytree(ann_source, ann_dest)  
    ann_path = "{}/lang_annotations/auto_lang_ann.npy".format(parse_folder)
    annotations = np.load(ann_path, allow_pickle=True).item()

    annotations_language = annotations["language"]["ann"]
    annotations_info = annotations["info"]["indx"]

    stop_annotations = []
    for annotation in annotations_language:
            stop_annotation = annotation + stop_token
            stop_annotations.append(stop_annotation)

    stop_annotations = np.array(stop_annotations)

    np.save(ann_path, {"language": {"ann": stop_annotations}, "info": {"indx": annotations_info}}, allow_pickle=True)
    print("added stop token to annotations")



src_folder= CFG.datapath_training
parse_folder= CFG.datapath_training_parsed
parse_val = True
if(parse_val):
    print("parsing validation data, change flase to False to parse train data")
    src_folder= CFG.datapath_val
    parse_folder= CFG.datapath_val_parsed 



parse()
add_stop_token()


"""
ann_source = "{}/lang_annotations".format(src_folder)
ann_dest = "{}/lang_annotations".format(parse_folder)
shutil.rmtree(ann_dest)
shutil.copytree(ann_source, ann_dest)
"""
"""
ann_path = "{}/lang_annotations/auto_lang_ann.npy".format(parse_folder)
annotations = np.load(ann_path, allow_pickle=True).item()
annotations = annotations["language"]["ann"]
for annotation in annotations:
    print(annotation)
"""