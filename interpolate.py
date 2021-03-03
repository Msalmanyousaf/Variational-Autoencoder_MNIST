# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:44:55 2021

@author: Salman
"""
import torch
import torchvision.transforms as transforms
from model import*
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST  
from torchvision.utils import make_grid
from torch.utils.data import DataLoader 


def plot_latent_space (model, dataloader):
    fig_latent = plt.figure()
    plt.figure(fig_latent.number)
    for data, labels in dataloader:
        z, _ = model.encoder (data)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    return (fig_latent)
        
def in_between_dig (model, digit_1, digit_2, n):
    # n is total number of interpolations
    fig = plt.figure()
    
    z1, _ = model.encoder(digit_1)
    z2, _ = model.encoder(digit_2)
    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
    inter_images = model.decoder(z)
    inter_images = inter_images.view(inter_images.size(0), 1, 28, 28)
    inter_images = inter_images.clamp(0, 1)   
        
    plt.figure(fig.number)
    plt.title("Interpolated images")
    plt.imshow(np.transpose(make_grid(inter_images, n, 1).detach().numpy(), (1, 2, 0)))
    plt.show()      
    return(fig)

# loading data
trainset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset  = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

trainloader = DataLoader(trainset, batch_size=100, shuffle=False)
testloader  = DataLoader(testset, batch_size=100, shuffle=False)


# loading pretrained models weights
vae_2_latents = VAE(e_hidden = 500, d_hidden = 500, latent_dim = 2)
vae_6_latents = VAE(e_hidden = 500, d_hidden = 500, latent_dim = 6)

vae_2_latents.load_state_dict(torch.load("model_2_latents.pt"))
vae_2_latents.eval()

vae_6_latents.load_state_dict(torch.load("model_6_latents.pt"))
vae_6_latents.eval()


# plotting latent space for 2d latents
latent_space_2d = plot_latent_space(vae_2_latents, trainloader)
latent_space_2d.savefig("latent_space_2d.png", dpi=300)

# generating in-between images
x, y = next(iter(trainloader))      # grab a batch
x_1 = x[y == 2][0] # find a 2 and then get its first occurence
x_2 = x[y == 6][0] # find a 7 and then get its first occurence

in_between_2d = in_between_dig (vae_2_latents, x_1, x_2, 15) # 15 is total interpolations
in_between_2d.savefig("interpol_2d.png", dpi = 300)

in_between_6d = in_between_dig (vae_6_latents, x_1, x_2, 15)
in_between_6d.savefig("interpol_6d.png", dpi = 300)
 
    