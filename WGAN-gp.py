import torch
import torch.nn as nn

from noise import generate_noise 
from generatorRNA import generatorRNA
from discriminatorRNA import discriminatorRNA
from critic import Critic

import torch.nn.init as init
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

from torch.utils.data import DataLoader
from dataLoader import datasetRNA
from fastDataLoader import fastdatasetRNA

import os
from datetime import datetime
import sys 


# Hyperparameters
latent_dim = 256
batch_size = 128
sequence_length = 120

# Model architecture
num_layers = 2 # layers in transformer encoder
num_heads = 8
d_model = 128
d_ff = 512
dropout = 0.25

# Training parameters
n_epochs = 200
n_critic = 2 # Increase if the generator is weak, decrease if the generator is strong
lambda_gp = 3  # Increase if the discriminator learns too aggressively, decrease if it learns too slowly

# Learning rates:+ higher for discriminator, lower for generator
lr_c = 0.0002
lr_g = 0.0005

# Create directories for logs
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

torch.cuda.empty_cache()
# Ensure CUDA is properly initialized before running the main script
def init_cuda():
    try:
        if torch.cuda.is_available():
            # Explicitly set device and initialize
            device = torch.device('cuda')
            torch.cuda.set_device(0)
            torch.cuda.init()
            print(f"CUDA Initialized. Device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print("CUDA not available. Using CPU.")
            return torch.device('cpu')
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        sys.exit(1)

# Use this at the start of your script
device = init_cuda()

def initialize_weights_critic(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)

def initialize_weights_generator(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# Initialize models
generator = generatorRNA(latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff).to(device)
critic = Critic(sequence_length).to(device)

generator.apply(initialize_weights_generator)
critic.apply(initialize_weights_critic)

# Dataset setup
#file_path = "/home/michal/Desktop/RNA_Monster/GANbert-RNA/dataset_Rfam_6320_13classes.fasta"
#family = "5S_rRNA"
#dataset = datasetRNA(file_path, family, sequence_length, only_positive=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# print(f"Dataset loaded: {len(dataset)} samples, batch size: {dataloader.batch_size}")

file_path = r"C:\Users\michi\Desktop\RNA_Monster\RF00097.fa"

fastdataset = fastdatasetRNA(file_path, sequence_length, only_positive=False)
print(f"Średnia długość sekwencji: {fastdataset.average_sequence_length()}")

dataloader = DataLoader(fastdataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Dataset loaded: {len(fastdataset)} samples, batch size: {dataloader.batch_size}")

# Optimizers with adjusted learning rates
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.999))

# optimizer_G = torch.optim.rmsprop(generator.parameters(), lr=lr_g)
# optimizer_C = torch.optim.rmsprop(critic.parameters(), lr=lr_c)

def gradient_penalty(critic, real_samples, fake_samples, device="cpu"):
    batch_size, sequence_length, nucleotides = real_samples.shape
    epsilon = torch.rand((batch_size, 1, 1)).to(device)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples

    with torch.backends.cudnn.flags(enabled=False):
        mixed_scores = critic(interpolated)
    

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# Gradient penalty for WGAN-GP
# def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=1.0, device="cpu"):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    
#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

#     with torch.backends.cudnn.flags(enabled=False):  # Wyłącz CuDNN TYLKO dla gradient penalty
#         critic_interpolates = critic(interpolates)   
#     # Get gradient w.r.t. interpolates
#     gradients = torch.autograd.grad(
#         outputs=critic_interpolates,
#         inputs=interpolates,
#         grad_outputs=torch.ones_like(critic_interpolates),
#         create_graph=True,
#         retain_graph=True,
#     )[0]
    
#     gradients = gradients.reshape(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return lambda_gp * gradient_penalty

# Helper function to log metrics
def log_metrics(epoch, batch, d_loss, g_loss, critic_real, critic_fake):
    # Obliczanie median wartości krytyka dla dynamicznego progu
    real_threshold = torch.median(critic_real).item()
    fake_threshold = torch.median(critic_fake).item()
    
    # Nowe metryki
    real_accuracy = torch.mean((critic_real > fake_threshold).float()).item() * 100
    fake_accuracy = torch.mean((critic_fake < real_threshold).float()).item() * 100
    gen_fooling_rate = torch.mean((critic_fake > torch.median(critic_real)).float()).item() * 100

    
    log_str = (
        f"[Epoch {epoch}/{n_epochs}] [Batch {batch}/{len(dataloader)}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real Acc: {real_accuracy:.2f}%] [Fake Acc: {fake_accuracy:.2f}%] "
        f"[Gen Fooled: {gen_fooling_rate:.2f}%] "
        f"[Real val: {torch.mean(critic_real).item():.4f}] [Fake val: {torch.mean(critic_fake).item():.4f}]"
    )
    print(log_str)
    with open(log_file, "a") as f:
        f.write(log_str + "\n")
    
    #Wizualizacja rozkładu wartości krytyka co 500 batchy
    if batch % 100 == 0:
        plot_critic_scores(critic_real, critic_fake, epoch, batch)

def plot_critic_scores(critic_real, critic_fake, epoch, batch):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.hist(critic_real.cpu().detach().numpy(), bins=50, alpha=0.5, label="Real")
    plt.hist(critic_fake.cpu().detach().numpy(), bins=50, alpha=0.5, label="Fake")
    plt.legend()
    plt.xlabel("Critic Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Critic Scores - Epoch {epoch} Batch {batch}")
    plt.show()

# Training loop
batches_done = 0

generator.train()
critic.train()

model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)

def save_generator(generator, epoch):
    model_path = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
    torch.save(generator.state_dict(), model_path)
    print(f"Generator model saved: {model_path}")

for epoch in range(n_epochs):
    for i, (real_data, _) in enumerate(dataloader):
        if i % 150 == 0:
            save_generator(generator, i)
        batch_size = real_data.size(0)
        real_rna = real_data.float().to(device)
        
        z = generate_noise(latent_dim, batch_size).to(device)
        fake_rna = generator(z)

        critic_real = critic(real_rna)
        critic_fake = critic(fake_rna.detach())
        
        gp = gradient_penalty(critic, real_rna, fake_rna, device)
        critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + (gp * lambda_gp)
        critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        # print("\n=== Sprawdzanie gradientów krytyka ===")
        # for name, param in critic.named_parameters():
        #     if param.grad is None:
        #         print(f"❌ Brak gradientu dla warstwy: {name}")
        #     else:
        #         max_grad = param.grad.abs().max().item()
        #         mean_grad = param.grad.abs().mean().item()
        #         print(f"✅ Gradient OK w warstwie: {name}, max: {max_grad:.6f}, mean: {mean_grad:.6f}")

        optimizer_C.step()

        if i % n_critic == 0:
            
            z = generate_noise(latent_dim, batch_size).to(device)

            fake_rna = generator(z)
            generator_loss = -torch.mean(critic(fake_rna))
            generator.zero_grad()
            generator_loss.backward()
            print("\n=== Sprawdzanie gradientów generatora ===")
            for name, param in generator.named_parameters():
                if param.grad is None:
                    print(f"❌ Brak gradientu dla warstwy: {name}")
                else:
                    max_grad = param.grad.abs().max().item()
                    print(f"✅ Gradient OK w warstwie: {name}, max: {max_grad:.6f}")
            optimizer_G.step()

        log_metrics(epoch, i, critic_loss.item(), generator_loss.item(), critic_real, critic_fake)


