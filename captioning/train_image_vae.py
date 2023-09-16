import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize
from models.visual_autoencoder import VisualAutoencoder
import matplotlib.pyplot as plt
import wandb
from dataset import AttrDict
from image_vae_dataset import ImageVAEDataset
import config as CFG
import h5py
import pdb
from tqdm import tqdm
from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

wandb.init(project="ImageVAE")

def visualize_reconstruction(original, reconstructed):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.permute(1, 2, 0))
    axs[0].set_title('Original')
    axs[1].imshow(reconstructed.detach().permute(1, 2, 0))
    axs[1].set_title('Reconstructed')
    plt.close()
    # return figure for wandb
    return fig

def train():
    # Train the model
    step = 0
    #use tqdm for progress bar
    for epoch in range(num_epochs):
        train_dataset.ids = iter(train_dataset.ds)
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, data in enumerate(tqdm(train_loader)):
            # Get the inputs and move them to the GPU if available
            inputs = data['image']
            inputs = inputs.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            reconstruction_loss = criterion(outputs, inputs)
            #ssim_loss = SSIM_criterion(outputs, inputs) * 0.001

            loss = reconstruction_loss #+ ssim_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Log the loss
            wandb.log({"Total Loss": loss.item()}, step=step)
            wandb.log({"Reconstruction Loss": reconstruction_loss.item()}, step=step)
            #wandb.log({"SSIM Loss": ssim_loss.item()}, step=step)

            # Log reconstructed and original images every 10 batches
            if step % 100 == 0:
                #visualize_reconstruction(inputs[0].detach().cpu(), outputs[0].detach().cpu())
                wandb.log({"Training Images": wandb.Image(visualize_reconstruction(inputs[0].detach().cpu(), outputs[0].detach().cpu()))}, step=step)
                #save model weights
                torch.save(model.state_dict(), "checkpoints/image_vae_high_res.pth")

            step += 1



# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
# encoding_size = 512
learning_rate = 0.001
# batch_size = 32
# num_epochs = 1000

encoding_size = 512
batch_size = 16
num_epochs = 100

# Load the CIFAR10 dataset
# dataset = h5py.File(CFG.dataset_path, 'r')

dataset_path = "gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1/"
train_dataset = ImageVAEDataset(phase='train', dataset_path=dataset_path)
train_loader = DataLoader(train_dataset,
                          #num_workers=16,
                          #prefetch_factor=6, 
                          batch_size=batch_size, 
                          shuffle=True)


# Create the model and optimizer
model = VisualAutoencoder(encoding_size)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load saved weights of model
# model.load_state_dict(torch.load("checkpoints/image_vae_.pth"))

# Define the loss function
criterion = nn.MSELoss()
SSIM_criterion = SSIMLoss().cuda()

train()