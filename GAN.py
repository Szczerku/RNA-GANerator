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
num_layers = 2 # layers in transformer encoder
num_heads = 8
d_model = 512
d_ff = 2048
dropout = 0.3

# Training parameters
n_epochs = 200
n_critic = 1  # higher number whan generator is weak smaller when generator is strong
lambda_gp = 1.0  # Reduced from 10 to 1.0 for better gradient penalty balance
clip_value = 0.5  # For gradient clipping

# Learning rates: higher for discriminator, lower for generator
lr_d = 0.0001
lr_g = 0.001

# Create directories for saving models and logs
save_dir = "./saved_models"
log_dir = "./logs"
os.makedirs(save_dir, exist_ok=True)
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
def compute_gradient_penalty(D, real_samples, fake_samples, lambda_gp=1.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
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
    gradients = gradients.reshape(gradients.size(0), -1)
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
best_fooling_rate = 0
best_d_loss = float('inf')

for epoch in range(n_epochs):
    for i, (real_data, _) in enumerate(dataloader):
        real_rna = Variable(real_data.type(Tensor))
        
        # -----------------
        # Train Discriminator
        # -----------------
        optimizer_D.zero_grad()

        # Generate fake RNA sequences
        z = generate_noise(latent_dim, batch_size)
        with torch.no_grad():
            fake_rna = generator(z).detach()

        # Measure discriminator's ability to classify real and fake samples
        real_validity = discriminator(real_rna)
        fake_validity = discriminator(fake_rna)

        # Calculate gradient penalty
        gp = compute_gradient_penalty(discriminator, real_rna.data, fake_rna.data, lambda_gp)

        # Discriminator loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp
        d_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
        
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        # Train generator every n_critic iterations
        if i % n_critic == 0:
            optimizer_G.zero_grad()

            # Generate fake RNA sequences
            z = generate_noise(latent_dim, batch_size)
            fake_rna = generator(z)
            
            # Calculate generator's ability to fool the discriminator
            fake_validity = discriminator(fake_rna)
            
            # Add L2 regularization to generator
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in generator.parameters())
            
            # Generator loss
            g_loss = -torch.mean(fake_validity) + l2_lambda * l2_norm
            g_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
            
            optimizer_G.step()

            # Calculate metrics using more robust thresholds
            real_mean = torch.mean(real_validity).item()
            fake_mean = torch.mean(fake_validity).item()
            threshold = (real_mean + fake_mean) / 2
            
            real_pred = real_validity > threshold  # True if real samples classified as real
            fake_pred = fake_validity < threshold  # True if fake samples classified as fake

            real_accuracy = real_pred.float().mean().item() * 100
            fake_accuracy = fake_pred.float().mean().item() * 100
            gen_fooling_rate = 100 - fake_accuracy
            
            # Log training progress
            log_metrics(epoch, i, d_loss.item(), g_loss.item(), real_validity, fake_validity, 
                        real_accuracy, fake_accuracy, gen_fooling_rate)
            
            batches_done += n_critic
            
            # Save best models based on metrics
            if gen_fooling_rate > best_fooling_rate and epoch > 10:
                best_fooling_rate = gen_fooling_rate
                torch.save(generator.state_dict(), os.path.join(save_dir, "best_generator_fooling.pth"))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, "best_discriminator_fooling.pth"))
                print(f"Saved best model with fooling rate: {best_fooling_rate:.2f}%")
            
            if d_loss.item() < best_d_loss and epoch > 10:
                best_d_loss = d_loss.item()
                torch.save(generator.state_dict(), os.path.join(save_dir, "best_generator_dloss.pth"))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, "best_discriminator_dloss.pth"))
                print(f"Saved best model with D loss: {best_d_loss:.4f}")
    
    # Save checkpoint after each epoch
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'best_fooling_rate': best_fooling_rate,
            'best_d_loss': best_d_loss,
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
        print(f"Saved checkpoint at epoch {epoch}")

print("Training complete!")