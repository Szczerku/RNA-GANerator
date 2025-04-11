import torch
import os
import csv
from utils.noise_generator import generate_noise

metrics_log_path = "training_metrics.csv"
if not os.path.exists(metrics_log_path):
    with open(metrics_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "d_loss", "g_loss", "d_real", "d_fake", "wasserstein_distance"])

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
        f"[Batch {batch+1}] "
        f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}] "
        f"[Real Acc: {real_accuracy:.2f}%] [Fake Acc: {fake_accuracy:.2f}%] "
        f"[Gen Fooled: {gen_fooling_rate:.2f}%] "
        f"[Real val: {d_real_mean:.4f}] [Fake val: {d_fake_mean:.4f}] "
        f"[D(real) - D(fake): {wasserstein_distance:.4f}]"
    )
    print(log_str)


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

    os.makedirs(args.save_dir, exist_ok=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.999))

    if not os.path.exists(args.log_file):
        with open(args.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "d_loss", "g_loss", "wasserstein_distance"])

    total_batches = 0

    for epoch in range(args.epochs):
        for i, (real_data) in enumerate(dataset.dataloader):
            total_batches += 1
            if total_batches % 100 == 0:
                #save generator model
                torch.save(generator.state_dict(), os.path.join(args.save_dir, f"generator_epoch_{epoch}_batch_{total_batches}.pth"))
            
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
            
                log_metrics(total_batches, critic_loss.item(), generator_loss.item(), critic_real, critic_fake)


