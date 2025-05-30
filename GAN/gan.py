# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from datetime import datetime

# from torch.utils.data import DataLoader
# from torch.autograd import Variable

# from utils.noise_generator import generate_noise
# from models.generator_rna import GeneratorRNA
# from models.discriminator_rna import DiscriminatorRNA
# from loaders.fasta_data_loader import FastDatasetRNA

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} for training")

# latent_dim = 256
# batch_size = 64
# sequence_length = 109
# num_layers = 4 
# num_heads = 8
# d_model = 128
# d_ff = 512
# dropout = 0.1

# n_epochs = 10
# lr_d = 0.0001 
# lr_g = 0.0005  

# log_dir = "./logs"
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# generator = GeneratorRNA(latent_dim, sequence_length, d_model, num_layers, num_heads, d_ff).to(device)
# discriminator = DiscriminatorRNA(sequence_length).to(device)

# file_path = r"C:\Users\michi\Desktop\RNA_Monster\data\RF00097.fa"
# dataloader = FastDatasetRNA(file_path, batch_size=batch_size, sequence_length=sequence_length)
# # print(f"Średnia długość sekwencji: {fastdataset.average_sequence_length()}")

# # dataloader = DataLoader(fastdataset, batch_size=batch_size, shuffle=True, drop_last=True)
# # print(f"Dataset loaded: {len(fastdataset)} samples, batch size: {dataloader.batch_size}")

# optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# criterion = nn.BCELoss()

# def add_noise(data, std=0.05):
#     return data + std * torch.randn_like(data)


# for epoch in range(n_epochs):
#     generator.train()
#     discriminator.train()

#     for i, (real_data) in enumerate(dataloader):
#         real_rna = real_data.to(device).float()
#         batch_size = real_rna.size(0)

#         real_rna = add_noise(real_rna)

#         z = generate_noise(latent_dim, batch_size).to(device)
#         fake_rna = generator(z).detach()

#         discriminatorReal = discriminator(real_rna)
#         discriminatorFake = discriminator(fake_rna)

#         d_loss_real = criterion(discriminatorReal, torch.ones_like(discriminatorReal, device=device))
#         d_loss_fake = criterion(discriminatorFake, torch.zeros_like(discriminatorFake, device=device))
#         d_loss = d_loss_real + d_loss_fake

#         optimizer_D.zero_grad()
#         d_loss.backward()
#         optimizer_D.step()

#         z = generate_noise(latent_dim, batch_size).to(device)
#         fake_rna = generator(z)
#         g_loss = criterion(discriminator(fake_rna), torch.ones_like(discriminator(fake_rna), device=device))

#         optimizer_G.zero_grad()
#         g_loss.backward()
#         print("\n=== Sprawdzanie gradientów generatora ===")
#         for name, param in generator.named_parameters():
#             if param.grad is None:
#                 print(f"Brak gradientu dla warstwy: {name}")
#             else:
#                 max_grad = param.grad.abs().max().item()
#                 print(f"Gradient OK w warstwie: {name}, max: {max_grad:.6f}")

#         optimizer_G.step()

#         real_pred = (discriminatorReal > 0.5).float()  # 1 jeśli zaklasyfikowane jako real
#         fake_pred = (discriminatorFake > 0.5).float()  # 1 jeśli zaklasyfikowane jako real

#         real_as_real = (real_pred == 1).sum().item()
#         fake_as_real = (fake_pred == 1).sum().item()
#         real_as_fake = (real_pred == 0).sum().item()
#         fake_as_fake = (fake_pred == 0).sum().item()

#         real_as_real_pct = (real_as_real / batch_size) * 100
#         fake_as_real_pct = (fake_as_real / batch_size) * 100
#         real_as_fake_pct = (real_as_fake / batch_size) * 100
#         fake_as_fake_pct = (fake_as_fake / batch_size) * 100
#         generator_fooling_pct = (fake_as_real / (fake_as_real + fake_as_fake)) * 100 if (fake_as_real + fake_as_fake) > 0 else 0

#         print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
#               f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
#               f"Real as Real: {real_as_real_pct:.2f}% | "
#               f"Fake as Real: {fake_as_real_pct:.2f}% | "
#               f"Real as Fake: {real_as_fake_pct:.2f}% | "
#               f"Fake as Fake: {fake_as_fake_pct:.2f}% | "
#               f"Generator Fooling: {generator_fooling_pct:.2f}%")
