import os
import numpy as np
from r3m import load_r3m
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch
import shutil
import config as CFG
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def r3m_preprocess(n_px=224):
    return Compose([
        ToPILImage(mode='RGB'),
        Resize(224),
        # CenterCrop(224),
        ToTensor()
    ])

def r3m_preprocess_2(n_px=224):
    return Compose([
        ToPILImage(mode='RGB'),
        Resize((224,112)),
        #CenterCrop(224),
        ToTensor()
    ])

def parse(): 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3m = load_r3m("resnet50")
    r3m.eval()
    r3m.to(device)
    transform =  r3m_preprocess()
    transform2 =  r3m_preprocess_2()

    data_episodes = [f for f in os.listdir(src_folder) if f.startswith('episode_')]

    length = len(data_episodes)

    for idx, data_episode in enumerate(data_episodes):
        data_episode_path = os.path.join(src_folder, data_episode)
        data = np.load(data_episode_path, allow_pickle=True)

        static = transform(data['rgb_static'].astype(np.uint8)) * 255.0
        gripper = transform(data['rgb_gripper'].astype(np.uint8)) * 255.0

        static_small = transform2(data['rgb_static'].astype(np.uint8)) * 255.0
        gripper_small = transform2(data['rgb_gripper'].astype(np.uint8)) * 255.0
        observations = torch.cat((static_small, gripper_small), dim=-1)

        static_transform = torch.tensor(r3m(static.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
        gripper_transform = torch.tensor(r3m(gripper.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())
        observations_transform = torch.tensor(r3m(observations.unsqueeze(dim=0).to(device)).squeeze().detach().cpu().numpy())


        processed_npz_path = os.path.join(parse_folder, data_episode)
        np.savez(processed_npz_path, observations=observations_transform, rgb_static=static_transform, rgb_gripper=gripper_transform, actions=data['actions'], robot_obs=data["robot_obs"])

        if os.name == 'nt': 
            os.system('cls')
        else:
            os.system('clear')
        print(idx, ": ", length)
        #debug 
        print(static.shape)
        print(gripper.shape)
        print(observations.shape)
        print(observations_transform.shape)
        print(static_transform.shape)
        print(gripper_transform.shape)
        """
        combined_img = to_pil_image(observations) 
        plt.imshow(combined_img)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()
        """

def add_lang_annotations(stop_token: str = ' \n'):
    ann_source = "{}/lang_annotations".format(src_folder)
    ann_dest = "{}/lang_annotations".format(parse_folder)
    # shutil.rmtree(ann_dest)
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


parse_val = False

#src_folder= CFG.datapath_training
#parse_folder= '/media/tim/E/datasets/task_D_D_parsed_2/training'

src_folder= CFG.datapath_training_abcd
parse_folder= '/media/tim/E/datasets/task_ABC_D_parsed_2/training'

parse()
add_lang_annotations()

parse_val = True

print("parsing validation data, change flase to False to parse train data")

src_folder= CFG.datapath_val_abcd
parse_folder= '/media/tim/E/datasets/task_ABC_D_parsed_2/validation'

parse()
add_lang_annotations()