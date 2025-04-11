import torch
import numpy as np

"""
Generates a batch of random noise vectors sampled from a standard normal distribution.
These vectors serve as latent inputs for generative models such as GANs.
Its good to check 256 - 1024 latent space,
for example: 128, 256, 512, 1024
"""
def generate_noise(latent_dim, sample_size):
    noise = np.random.normal(0, 1, (sample_size, latent_dim))
    noise = torch.tensor(noise).float()
    return noise



