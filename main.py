# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:42:52 2021

@author: Salman
"""
# Import libraries
import torchvision.transforms as transforms    # to normalize, scale etc the dataset
from torch.utils.data import DataLoader        # to load data into batches (for SGD)
from torchvision.utils import make_grid        # Plotting. Makes a grid of tensors
from torchvision.datasets import MNIST         
import matplotlib.pyplot as plt                
import numpy as np
import torch
import torch.nn.functional as F                # contains activation functions, sampling layers etc
import torch.optim as optim                    # For optimization routines such as SGD, ADAM, ADAGRAD, etc

from model import*


# Loss
def vae_loss(image, reconstruction, mu, logvar):
  """Loss for the Variational AutoEncoder."""
  # Binary Cross Entropy for batch
  BCE = F.binary_cross_entropy(input=reconstruction.view(-1, 28*28), target=image.view(-1, 28*28), reduction='sum')
  # Closed-form KL Divergence
  KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE - KLD

def fit (trainloader, model):
    epoch_loss = 0
    number_of_batches = 0
    vae.train()
    # Grab the batch, we are only interested in images not on their labels
    for images, _ in trainloader:
        # Save batch to GPU, remove existing gradients from previous iterations
        # images = images.to(device)
        optimizer.zero_grad()

        # Feed images to VAE. Compute Loss for the current batch.
        reconstructions, latent_mu, latent_logvar = model(images)
        batch_loss = vae_loss(images, reconstructions, latent_mu, latent_logvar)
    
        # Backpropagate the loss & perform optimization step with such gradients
        batch_loss.backward()
        optimizer.step()
    
        # Add loss to the cumulative sum
        epoch_loss += batch_loss.item()  
        number_of_batches += 1
        
    epoch_loss /= number_of_batches
    return epoch_loss, reconstructions # here, reconstructions will be of last batch
  
def validate (testloader, model):
    model.eval()
    epoch_val_loss = 0
    num_of_batches = 0
    with torch.no_grad():
        for test_images, _ in testloader:
            
            # Send images to the GPU/CPU
            # test_images = test_images.to(device)
        
            # Feed images through the VAE to obtain their reconstruction & compute loss
            reconstructions, latent_mu, latent_logvar = model(test_images)
            batch_loss = vae_loss(test_images, reconstructions, latent_mu, latent_logvar)
        
            # Cumulative loss & Number of batches
            epoch_val_loss += batch_loss.item()
            num_of_batches += 1
 
        epoch_val_loss /= num_of_batches
        
    # note that the returned reconstructions are for the last batch.
    return epoch_val_loss, reconstructions


# ToTensor() transforms images to pytorch tensors AND scales the pixel values to be within [0, 1]. 

trainset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset  = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

batch_size = 100
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)


# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




e_hidden = 500        # Number of hidden units in the encoder. 
d_hidden = 500        # Number of hidden units in the decoder. 
latent_dim = 6        # Dimension of latent space. 
learning_rate = 0.001 
weight_decay = 1e-5   
epochs = 50           # Number of sweeps through the whole dataset


# figures for ploting
fig_orig = plt.figure()
fig_recons = plt.figure()
fig_losses = plt.figure()

   
# # Instantiate VAE with Adam optimizer
vae = VAE(e_hidden, d_hidden, latent_dim)
# vae = vae.to(device)    # send weights to GPU. Do this BEFORE defining Optimizer
optimizer = optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []  # to capture losses over the epochs
test_losses = []
for epoch in range(epochs):
    
    # training
    train_loss, _ = fit(trainloader, vae)
    train_losses.append(train_loss)
    print('Epoch [%d / %d] average train reconstruction error: %f' % (epoch+1, epochs, train_losses[-1]))    
    
    #validation
    test_loss, test_recons = validate(testloader, vae)

    test_losses.append(test_loss)
    print(f'Epoch {epoch+1}/{epochs}: average test reconstruction error: {test_losses[-1]}')
    
    # plot some reconstructed images of the final batch after the final epoch
    if epoch == epochs - 1: 
        with torch.no_grad():
            # reshape into image dimensions
            
            test_recons = test_recons.view(test_recons.size(0), 1, 28, 28)
            # bring to cpu
            # test_recons = test_recons.cpu()
            test_recons = test_recons.clamp(0, 1) # not required because I used sigmoid, but to be on safe side
            
            # choose first 50 images to show
            test_recons = test_recons[:50]
            
            plt.figure(fig_recons.number)
            plt.title("Reconstructed images")
            plt.imshow(np.transpose(make_grid(test_recons, 10, 5).numpy(), (1, 2, 0)))
            plt.show()        
            fig_recons.savefig("reconstructed.png", dpi = 300)
        
# save the trained model
torch.save(vae.state_dict(), "dict_model.pt")
torch.save(vae, "entire_model.pt")

# plot losses
plt.figure(fig_losses.number)
plt.plot(range(epochs), train_losses, label = "Train set")
plt.plot(range(epochs), test_losses, label = "Test set")
plt.title("Loss vs epochs")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
fig_losses.savefig("losses.png", dpi = 300)


# Display original test images from the last batch
with torch.no_grad():
    print("Original Images")
    # test images have dimensions [examples, 1, 28, 28]
    # test_images = test_images.cpu()
    *_, original_imgs = iter(testloader) # gets the last batch
    # labels are not required
    original_imgs, _ = original_imgs
    original_imgs = original_imgs.clamp(0, 1)
    original_imgs = original_imgs[:50]
    original_imgs = make_grid(original_imgs, 10, 5) # 10 is no. of images in each row
    original_imgs = original_imgs.numpy()
    original_imgs = np.transpose(original_imgs, (1, 2, 0))
    plt.figure(fig_orig.number)
    plt.title("Original images")
    plt.imshow(original_imgs)
    plt.show()
    fig_orig.savefig("original.png", dpi = 300)
    
    
    
#########################################################################    
# # generating new data
# vae.eval()
# with torch.no_grad():
#     # Sample from standard normal distribution
#     # z = torch.randn(50, latent_dim, device=device)
#     z = torch.randn(50, latent_dim)
    
#     # Reconstruct images from sampled latent vectors
#     recon_images = vae.decoder(z)
#     recon_images = recon_images.view(recon_images.size(0), 1, 28, 28)
#     # recon_images = recon_images.cpu()
#     recon_images = recon_images.clamp(0, 1)
    
#     # Plot Generated Images
#     plt.imshow(np.transpose(make_grid(recon_images, 10, 5).numpy(), (1, 2, 0)))    
    
# # visualizing latent space

# with torch.no_grad():
#   # Create empty (x, y) grid
#   latent_x = np.linspace(-1.5, 1.5, 20)
#   latent_y = np.linspace(-1.5, 1.5, 20)
#   latents = torch.FloatTensor(len(latent_x), len(latent_y), 2)
#   # Fill up the grid
#   for i, lx in enumerate(latent_x):
#     for j, ly in enumerate(latent_y):
#       latents[j, i, 0] = lx
#       latents[j, i, 1] = ly
#   # Flatten the grid
#   latents = latents.view(-1, 2)
#   # Send to GPU
#   # latents = latents.to(device)
#   # Find their representation
#   reconstructions = vae.decoder(latents).view(-1, 1, 28, 28)
#   reconstructions = reconstructions.cpu()
#   # Finally, plot
#   fig, ax = plt.subplots(figsize=(10, 10))
#   plt.imshow(np.transpose(make_grid(reconstructions.data[:400], 20, 5).clamp(0, 1).numpy(), (1, 2, 0))) 