import torch
import os
import csv
from utils.noise_generator import generate_noise


def log_metrics(epoch, batch_idx, num_batches, d_loss, g_loss, critic_real, critic_fake, log_path):
    # real_threshold = torch.median(critic_real).item()
    # fake_threshold = torch.median(critic_fake).item()
    # real_accuracy = torch.mean((critic_real > fake_threshold).float()).item() * 100
    # fake_accuracy = torch.mean((critic_fake < real_threshold).float()).item() * 100
    # gen_fooling_rate = torch.mean((critic_fake > torch.median(critic_real)).float()).item() * 100
    d_real_mean = torch.mean(critic_real).item()
    d_fake_mean = torch.mean(critic_fake).item()
    wasserstein_distance = d_real_mean - d_fake_mean

    print(
        f"[Epoch {epoch+1}] [Batch {batch_idx+1}/{num_batches}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real val: {d_real_mean:.4f}] [Fake val: {d_fake_mean:.4f}] "
        f"[Wasserstein: {wasserstein_distance:.4f}]"
    )
    if log_path:
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_idx,
                d_loss,
                g_loss,
                d_real_mean,
                d_fake_mean,
                wasserstein_distance
            ])



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




def train_wgan_gp(generator, critic, dataset, args, device):
    generator.train()
    critic.train()
    # directory for generator models
    os.makedirs(args.save_dir, exist_ok=True)
    # directory for training metrics (only if log_file is provided)
    # Prepare metric logging if enabled
    log_path = os.path.join(args.log_dir, "metrics.csv") if args.log_dir else None

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, "metrics.csv")
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "batch", "d_loss", "g_loss",
                "critic_real_mean", "critic_fake_mean", "wasserstein_distance"
            ])
    else:
        log_path = None


    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.999))

    total_batches = 0

    for epoch in range(args.epochs):
        for i, (real_data) in enumerate(dataset.dataloader):
            total_batches += 1

            if total_batches % 100 == 0:
                #save generator model
                torch.save(generator.state_dict(), os.path.join(args.save_dir, f"generator_epoch_{epoch+1}_batch_{total_batches}.pth"))
            
            batch_size = real_data.size(0)
            real_rna = real_data.float().to(device)

            z = generate_noise(args.latent_dim, batch_size).to(device)
            fake_rna = generator(z)

            critic_real = critic(real_rna)
            critic_fake = critic(fake_rna.detach())

            gp = gradient_penalty(critic, real_rna, fake_rna, device)
            critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + (gp * args.lambda_gp)
            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_C.step()

            if i % args.n_critic == 0:

                z = generate_noise(args.latent_dim, batch_size).to(device)
                fake_rna = generator(z)
                generator_loss = -torch.mean(critic(fake_rna))
                generator.zero_grad()
                generator_loss.backward()
                optimizer_G.step()
    
            log_metrics(epoch, i, len(dataset.dataloader), critic_loss.item(), generator_loss.item(), critic_real, critic_fake, log_path)



