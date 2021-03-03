# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:39:40 2021

@author: Salman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, e_hidden = 500, d_hidden = 500, latent_dim = 2):
        """Variational Auto-Encoder Class"""
        super(VAE, self).__init__()
        # Encoding Layers
        self.e_input2hidden = nn.Linear(in_features=784, out_features=e_hidden)
        self.e_hidden2mean = nn.Linear(in_features=e_hidden, out_features=latent_dim)
        self.e_hidden2logvar = nn.Linear(in_features=e_hidden, out_features=latent_dim)
        
        # Decoding Layers
        self.d_latent2hidden = nn.Linear(in_features=latent_dim, out_features=d_hidden)
        self.d_hidden2image = nn.Linear(in_features=d_hidden, out_features=784)
        
    def encoder(self, x):
        # Shape Flatten image to [batch_size, input_features]
        x = x.view(-1, 784)
        # Feed x into Encoder to obtain mean and logvar
        x = F.relu(self.e_input2hidden(x))
        return self.e_hidden2mean(x), self.e_hidden2logvar(x)
        
    def decoder(self, z):
        return torch.sigmoid(self.d_hidden2image(F.relu(self.d_latent2hidden(z))))
        
    def forward(self, x):
        # Encoder image to latent representation mean & std
        mu, logvar = self.encoder(x)
        
        # Sample z from latent space using mu and logvar
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
        else:
            z = mu
        
        # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar



