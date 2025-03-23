import torch
import torch.nn as nn

from noise import generate_noise 
from generatorRNA import generatorRNA
from discriminatorRNA import discriminatorRNA

import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

from torch.utils.data import DataLoader
from dataLoader import datasetRNA
from fastDataLoader import fastdatasetRNA

import os
from datetime import datetime

# Hyperparameters
latent_dim = 256
batch_size = 64
sequence_length = 120

# Model architecture
num_layers = 8 # layers in transformer encoder
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.3

# Training parameters
n_epochs = 200
n_critic = 5  # Increase if the generator is weak, decrease if the generator is strong
lambda_gp = 10  # Increase if the discriminator learns too aggressively, decrease if it learns too slowly

# Learning rates: higher for discriminator, lower for generator
lr_d = 0.0001
lr_g = 0.001

# Create directories for logs
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Initialize models
generator = generatorRNA(latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff)
discriminator = discriminatorRNA(sequence_length)

# GPU setup
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    print("Using GPU for training")
else:
    print("Using CPU for training")

# Dataset setup
#file_path = "/home/michal/Desktop/RNA_Monster/GANbert-RNA/dataset_Rfam_6320_13classes.fasta"
#family = "5S_rRNA"
#dataset = datasetRNA(file_path, family, sequence_length, only_positive=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# print(f"Dataset loaded: {len(dataset)} samples, batch size: {dataloader.batch_size}")

file_path = "/home/michal/Desktop/RNA_Monster/GANbert-RNA/RF00097.fa"

fastdataset = fastdatasetRNA(file_path, sequence_length, only_positive=False)
print(f"Średnia długość sekwencji: {fastdataset.average_sequence_length()}")

dataloader = DataLoader(fastdataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Dataset loaded: {len(fastdataset)} samples, batch size: {dataloader.batch_size}")

# Optimizers with adjusted learning rates
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# Gradient penalty for WGAN-GP
def gradient_penalty(D, real_samples, fake_samples, lambda_gp=1.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    alpha = alpha.expand_as(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gradient_penalty

# Helper function to log metrics
def log_metrics(epoch, batch, d_loss, g_loss, real_validity, fake_validity, real_accuracy, fake_accuracy, gen_fooling_rate):
    log_str = (
        f"[Epoch {epoch}/{n_epochs}] [Batch {batch}/{len(dataloader)}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real Acc: {real_accuracy:.2f}%] [Fake Acc: {fake_accuracy:.2f}%] "
        f"[Gen Fooled: {gen_fooling_rate:.2f}%] "
        f"[Real val: {torch.mean(real_validity).item():.4f}] [Fake val: {torch.mean(fake_validity).item():.4f}]"
    )
    print(log_str)
    with open(log_file, "a") as f:
        f.write(log_str + "\n")

# Training loop
batches_done = 0

for epoch in range(n_epochs):
    for i, (real_data, _) in enumerate(dataloader):
        real_rna = Variable(real_data.type(Tensor))
        
        z = generate_noise(latent_dim, batch_size)
        fake_rna = generator(z).detach()

        # Obliczenie predykcji dyskryminatora dla prawdziwych i fałszywych próbek
        real_preds = discriminator(real_rna).detach()
        fake_preds = discriminator(fake_rna).detach()

        # Konwersja do wartości procentowych
        real_as_real = (real_preds > 0.5).float().mean().item() * 100  # Prawdziwe uznane za prawdziwe
        real_as_fake = 100 - real_as_real  # Prawdziwe uznane za fałszywe

        fake_as_fake = (fake_preds < 0.5).float().mean().item() * 100  # Fałszywe uznane za fałszywe
        fake_as_real = 100 - fake_as_fake  # Fałszywe uznane za prawdziwe
        
        # Obliczenie skuteczności dyskryminatora
        d_accuracy = (real_as_real + fake_as_fake) / 2  # Średnia skuteczność D
        g_fooling_rate = 100 - fake_as_fake  # Ile fałszywych D uznał za prawdziwe

        # Liczenie strat dyskryminatora
        loss_real = real_preds.mean()
        loss_fake = fake_preds.mean()
        gp = gradient_penalty(discriminator, real_rna, fake_rna, lambda_gp)
        d_loss = loss_fake - loss_real + gp

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        if i % n_critic == 0:
            z = generate_noise(latent_dim, batch_size)
            fake_rna = generator(z)
            g_loss = -discriminator(fake_rna).mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                  f"[D Accuracy: {d_accuracy:.2f}%] "
                  f"[Real as Real: {real_as_real:.2f}%] [Real as Fake: {real_as_fake:.2f}%] "
                  f"[Fake as Fake: {fake_as_fake:.2f}%] [Fake as Real: {fake_as_real:.2f}%] "
                  f"[G Fooling Rate: {g_fooling_rate:.2f}%]")

            batches_done += n_critic