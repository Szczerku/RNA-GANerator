# its good to check 256 - 1024 latent space for example
# 128, 256, 512, 1024
# latenet space should be parameter 
import torch
import numpy as np

def generate_noise(latent_dim, sample_size):
    noise = np.random.normal(0, 1, (sample_size, latent_dim))
    noise = torch.tensor(noise).float()
    return noise



