import torch
import torch.nn as nn

from utils.noise_generator import generate_noise 
from models.generator_rna import GeneratorRNA
from models.resnet_generator_rna import ResNetGenerator
from models.critic import Critic
from utils.init_device import init_cuda
from utils.init_weights import initialize_weights

import torch.nn.init as init
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

from loaders.fasta_data_loader import FastDatasetRNA

import os
import csv
from datetime import datetime
import sys 
import subprocess

"""
Hyperparameters and model parameters

latent_dim: Latent dimension of the noise vector.
batch_size: Size of each batch during training.
sequence_length: Length of the RNA sequences to be generated.(If None it will be percentile - 98 of the dataset)

model_architecture: Parameters for the transformer encoder architecture.
num_layers: Number of layers in the transformer encoder.
num_heads: Number of attention heads in the transformer encoder.
d_model: Dimension of the model (hidden size).
d_ff: Dimension of the feedforward layer in the transformer encoder.
dropout: Dropout rate for the transformer encoder.

training_parameters: Parameters for training the model.
n_critic: Number of critic iterations per generator iteration.
lambda_gp: Gradient penalty coefficient for WGAN-GP.
"""


metrics_log_path = "training_metrics.csv"
if not os.path.exists(metrics_log_path):
    with open(metrics_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "d_loss", "g_loss", "d_real", "d_fake", "wasserstein_distance"])

# Hyperparameters
latent_dim = 256
batch_size = 64
sequence_length = 109

# # Training parameters
n_critic = 5 # Increase if the generator is weak, decrease if the generator is strong
lambda_gp = 10 # Increase if the discriminator learns too aggressively, decrease if it learns too slowly

lr_c = 0.0001
lr_g = 0.0005


device = init_cuda()





generator = ResNetGenerator(latent_dim, sequence_length).to(device)
initialize_weights(generator)
critic = Critic(sequence_length).to(device)

file_path = r"C:\Users\michi\Desktop\RNA_Monster\data\RF00097.fa"

data = FastDatasetRNA(file_path, batch_size=batch_size, sequence_length=sequence_length)
print(f"Średnia długość sekwencji: {data.percentile_length(98)}")
print(f"Dataset loaded: {len(data)} samples, batch size: {data.batch_size}")

# Optimizers with adjusted learning rates
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.999))


def gradient_penalty(critic, real_samples, fake_samples, device="cpu"):
    batch_size, sequence_length, nucleotides = real_samples.shape
    epsilon = torch.rand((batch_size, 1, 1)).to(device)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)
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


# Helper function to log metrics
def log_metrics(batch, d_loss, g_loss, critic_real, critic_fake):
    real_threshold = torch.median(critic_real).item()
    fake_threshold = torch.median(critic_fake).item()

    real_accuracy = torch.mean((critic_real > fake_threshold).float()).item() * 100
    fake_accuracy = torch.mean((critic_fake < real_threshold).float()).item() * 100
    gen_fooling_rate = torch.mean((critic_fake > torch.median(critic_real)).float()).item() * 100

    d_real_mean = torch.mean(critic_real).item()
    d_fake_mean = torch.mean(critic_fake).item()
    wasserstein_distance = d_real_mean - d_fake_mean
    with open(metrics_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            batch,
            d_loss,
            g_loss,
            d_real_mean,
            d_fake_mean,
            wasserstein_distance
        ])
    log_str = (
        f"[Batch {batch+1}/{len(data.dataloader)}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real Acc: {real_accuracy:.2f}%] [Fake Acc: {fake_accuracy:.2f}%] "
        f"[Gen Fooled: {gen_fooling_rate:.2f}%] "
        f"[Real val: {d_real_mean:.4f}] [Fake val: {d_fake_mean:.4f}] "
        f"[D(real) - D(fake): {wasserstein_distance:.4f}]"
    )
    print(log_str)


generator.train()
critic.train()

model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)

def save_generator(generator, epoch):
    model_path = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
    torch.save(generator.state_dict(), model_path)
    print(f"Generator model saved: {model_path}")


def save_generated_fasta_batches(generator, batch_idx, base_output_dir="infernal_hits"):
    generator.eval()
    with torch.no_grad():
        parent_dir = os.path.join(base_output_dir, str(batch_idx))
        os.makedirs(parent_dir, exist_ok=True)

        nucleotides = ['A', 'C', 'G', 'U']

        def decode_sequence(one_hot_sequence):
            decoded_seq = []
            for one_hot in one_hot_sequence:
                max_index = np.argmax(one_hot)
                if np.sum(one_hot) == 1:
                    decoded_seq.append(nucleotides[max_index])
                else:
                    decoded_seq.append('N')
            return "".join(decoded_seq)

        for sub_idx in range(10):  # 10 razy po 1000
            z = generate_noise(latent_dim, 1000).to(device)
            generated_sequences = generator(z).cpu().numpy()

            fasta_path = os.path.join(parent_dir, f"gen_{sub_idx+1}.fasta")

            with open(fasta_path, "w") as f:
                for i, seq in enumerate(generated_sequences):
                    decoded_seq = decode_sequence(seq)
                    f.write(f">Generated_{batch_idx}_{sub_idx+1}_{i+1}\n{decoded_seq}\n")

            print(f"✅ Zapisano {fasta_path}")

    generator.train()







total_batches = 0
num_epochs = 15
for epoch in range(num_epochs):
    print(f"\n--- EPOCH {epoch+1}/{num_epochs} ---")
    for i, (real_data) in enumerate(data.dataloader):
        total_batches += 1
        if total_batches % 100 == 0:
            save_generated_fasta_batches(generator, total_batches)


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

        log_metrics( total_batches, critic_loss.item(), generator_loss.item(), critic_real, critic_fake)


