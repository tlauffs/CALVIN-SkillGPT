import os
import numpy as np
import pandas as pd
import torch
import clip
from models.caption_model import ClipCaptionModel
from models.generalized_caption_model import ClipGeneralizedCaptionModel
import config as CFG
from torch.nn import functional as nnf
import matplotlib.pyplot as plt

"""
    Calculate loss of caption model
"""


def evaluate_loss(path, dataloader):
    clip_model, _ = clip.load("ViT-B/32", device=CFG.device, jit=True)
    clip_text_encoder = clip_model.encode_text
    mapper_model = ClipCaptionModel(prefix_length=10, clip_length=10).to(CFG.device)
    mapper_model.load_state_dict(torch.load(path, map_location=CFG.device))
    mapper_model = mapper_model.eval()
    total_loss = 0
    for data in dataloader:
        data.observations = data.observations.to(CFG.device)
        data.actions = data.actions.to(CFG.device)
        # data.state = data.state.to(CFG.device)
        data.instruction = clip_text_encoder(clip.tokenize(data.instruction).to(CFG.device)).to(CFG.device)
        data.gpt_tokens = data.gpt_tokens.to(CFG.device)
        data.gpt_mask = data.gpt_mask.to(CFG.device)

        outputs = mapper_model(data)

        with torch.no_grad():
            logits = outputs.logits[:, data.observations.shape[1] - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), data.gpt_tokens.flatten(), ignore_index=0)
            total_loss += loss.item()
    return total_loss / len(dataloader)


"""
    Calculate loss of generalized caption model
"""


def evaluate_loss_abc_sim(path, dataloader, envs, mode):
    clip_model, _ = clip.load("ViT-B/32", device=CFG.device, jit=True)
    clip_text_encoder = clip_model.encode_text

    model = ClipGeneralizedCaptionModel(prefix_length=10, clip_length=10, mode=mode).to(CFG.device)
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model = model.eval()

    total_loss = 0
    for data in dataloader:
        for env in envs:
            data.observations[env] = data.observations[env].to(CFG.device)
            data.actions[env] = data.actions[env].to(CFG.device)
        # data.state = data.state.to(CFG.device)
        data.instruction = clip_text_encoder(clip.tokenize(data.instruction).to(CFG.device)).to(CFG.device)
        data.gpt_tokens = data.gpt_tokens.to(CFG.device)
        data.gpt_mask = data.gpt_mask.to(CFG.device)

        outputs = model(data)
        losses = []
        logits_shape = data.observations[envs[0]].shape
        for output in outputs:
            logits = output.logits[:, logits_shape[1] - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), data.gpt_tokens.flatten(), ignore_index=0)
            losses.append(loss)
        average_loss = sum(losses) / len(losses)
        total_loss += average_loss.item()

    return total_loss / len(dataloader)


"""
    plot and compare loss data
"""


def plot_losses(csv_folder):
    for filename in os.listdir(csv_folder):
        csv_filepath = os.path.join(csv_folder, filename)

        loss_data = pd.read_csv(csv_filepath)

        if filename == 'abcd_loss_improved_full.csv':
            print('DGDFG')
            loss_data = loss_data.iloc[::2]
            loss_data['epoch'] = range(1, len(loss_data) + 1)

        loss_data = loss_data[loss_data['epoch'] <= 40]
        epochs = loss_data['epoch']
        val_loss = loss_data['val_loss']
        train_loss = loss_data['train_loss']

        val_loss = np.log(val_loss)
        train_loss = np.log(train_loss)

        plt.plot(epochs, val_loss, label='Validation Loss', marker='.')
        plt.plot(epochs, train_loss, label='Training Loss', marker='.')
        plt.xlabel('Epoch')
        plt.ylabel('Log(Loss)')
        title = 'Loss Over Epochs for: ' + csv_filepath
        plt.title(title)
        plt.legend()
        # plt.savefig('./results/abc_d/loss.png')
        plt.grid(True)
        plt.show()

        best_val_loss = float('inf')
        best_epoch = 0
        for epoch, val_loss in enumerate(loss_data['val_loss']):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
        print("best loss for ", csv_filepath, " for epoch ", best_epoch, ": ", best_val_loss)

    for filename in os.listdir(csv_folder):
        csv_filepath = os.path.join(csv_folder, filename)
        loss_data = pd.read_csv(csv_filepath)
        if filename == 'abcd_loss_improved_full.csv':
            print('DGDFG')
            loss_data = loss_data.iloc[::2]
            loss_data['epoch'] = range(1, len(loss_data) + 1)
        loss_data = loss_data[loss_data['epoch'] <= 45]
        # loss_data = loss_data[loss_data['epoch'] >= 10]
        epochs = loss_data['epoch']
        val_loss = loss_data['val_loss']
        val_loss = np.log(val_loss)
        plt.plot(epochs, val_loss, label=filename[:-4].replace('_', ' '), marker='.')

    plt.xlabel('Epoch')
    plt.ylabel('Log(Loss)')
    title = 'comapre validation loss'
    plt.title(title)
    plt.legend()
    plt.show()

    for filename in os.listdir(csv_folder):
        csv_filepath = os.path.join(csv_folder, filename)
        loss_data = pd.read_csv(csv_filepath)
        if filename == 'abcd_loss_improved_full.csv':
            print('DGDFG')
            loss_data = loss_data.iloc[::2]
            loss_data['epoch'] = range(1, len(loss_data) + 1)
        loss_data = loss_data[loss_data['epoch'] <= 45]
        epochs = loss_data['epoch']
        train_loss = loss_data['train_loss']
        train_loss = np.log(train_loss)
        plt.plot(epochs, train_loss, label=filename[:-4].replace('_', ' '), marker='.')

    plt.xlabel('Epoch')
    plt.ylabel('Log(Loss)')
    title = 'comapre train loss'
    plt.title(title)
    plt.legend()
    plt.show()
