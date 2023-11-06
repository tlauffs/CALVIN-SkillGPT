import numpy as np
import os
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from datasets.dataset import AttrDict 
from utils.beamsearch import beamsearch
from models.caption_model import ClipCaptionModel
from models.generalized_caption_model import ClipGeneralizedCaptionModel
import config as CFG

"""
    generate captions using sliding window
"""
def generate_annotations(start, stop, threshold, data_path, model_path): 
    window_size = 64
    stride = 32
    start_epi = start
    end_epi = stop
    threshold =  threshold

    best_model = ClipCaptionModel(prefix_length=10, clip_length=10).to(CFG.device)
    best_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    best_model = best_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    generated_captions = []
    print("generating annotations...")
    for i in tqdm(range(start_epi, end_epi, stride)):
        if os.path.exists(f"{data_path}/episode_{i:07d}.npz") and os.path.exists(f"{data_path}/episode_{i + window_size:07d}.npz"):
            actions = torch.zeros(64, 7) 
            #state = torch.zeros(64, 15) 
            observations = torch.zeros(64, 2048) 
            for idx, j in enumerate(range(i, i + window_size)):
                # print(idx, j)
                data = np.load(f"{data_path}/episode_{j:07d}.npz", allow_pickle=True)
                actions[idx] = torch.tensor(data['actions'])
                #state[idx] = torch.tensor(data['robot_obs'])
                observations[idx] = torch.tensor(data['observations'])
            actions = actions.to(CFG.device)
            observations = observations.to(CFG.device)
            #state = state.to(CFG.device)
            #src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0), state=state.unsqueeze(0))
            src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0))
            behaviour_encoding = best_model.behaviour_encoder(src)
            prefix_embed = best_model.project_to_gpt(behaviour_encoding)
            generated_caption =  beamsearch(best_model, tokenizer, prefix_embed)
            score = generated_caption[0][1].item()
            if score > threshold:
                generated_captions.append((generated_caption, (i, i + window_size)))

    return generated_captions

"""
    generate single caption at index start
"""
def generate_annotation(start, data_path, model_path): 
    start_epi = start
    window_size = 64

    best_model = ClipCaptionModel(prefix_length=10, clip_length=10).to(CFG.device)
    best_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    best_model = best_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if os.path.exists(f"{data_path}/episode_{start_epi:07d}.npz") and os.path.exists(f"{data_path}/episode_{start_epi + window_size:07d}.npz"):
        actions = torch.zeros(64, 7) 
        #state = torch.zeros(64, 15) 
        observations = torch.zeros(64, 2048) 
        for idx, j in enumerate(range(start_epi, start_epi + window_size)):
            data = np.load(f"{data_path}/episode_{j:07d}.npz", allow_pickle=True)
            actions[idx] = torch.tensor(data['actions'])
            #state[idx] = torch.tensor(data['robot_obs'])
            observations[idx] = torch.tensor(data['observations'])
        actions = actions.to(CFG.device)
        observations = observations.to(CFG.device)
        #state = state.to(CFG.device)
        #src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0), state=state.unsqueeze(0))
        src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0))
        behaviour_encoding = best_model.behaviour_encoder(src)
        prefix_embed = best_model.project_to_gpt(behaviour_encoding)
        generated_caption =  beamsearch(best_model, tokenizer, prefix_embed)       
        return generated_caption


"""
    generate single caption at index start using ClipGeneralizedCaptionModel
"""
def generate_annotation_abc(start, data_path, model_path): 
    start_epi = start
    window_size = 64

    best_model = ClipGeneralizedCaptionModel(prefix_length=10, clip_length=10, mode='val').to(CFG.device)
    best_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    best_model = best_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if os.path.exists(f"{data_path}/episode_{start_epi:07d}.npz") and os.path.exists(f"{data_path}/episode_{start_epi + window_size:07d}.npz"):
        actions = torch.zeros(64, 7) 
        #state = torch.zeros(64, 15) 
        observations = torch.zeros(64, 2048) 
        for idx, j in enumerate(range(start_epi, start_epi + window_size)):
            data = np.load(f"{data_path}/episode_{j:07d}.npz", allow_pickle=True)
            actions[idx] = torch.tensor(data['actions'])
            observations[idx] = torch.tensor(data['observations'])
        actions = actions.to(CFG.device)
        observations = observations.to(CFG.device)
        #state = state.to(CFG.device)
        #src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0), state=state.unsqueeze(0))
        src = AttrDict(observations=observations.unsqueeze(0), actions=actions.unsqueeze(0))
        behaviour_encoding = best_model.behaviour_encoder(src)
        prefix_embed = best_model.project_to_gpt(behaviour_encoding)
        generated_caption =  beamsearch(best_model, tokenizer, prefix_embed)       
        return generated_caption
    