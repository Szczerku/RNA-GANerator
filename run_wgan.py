import argparse
from utils.init_device import init_cuda
from loaders.fasta_data_loader import FastDatasetRNA
from models.resnet_generator_rna import ResNetGenerator
from utils.init_weights import initialize_weights
from models.critic import Critic
from train_wgan_gp import train_wgan_gp

"""
Simple example of use:
python run_gan.py --data data/RF00097.fa
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run WGAN-GP for training RNA sequences")
    parser.add_argument("--data", type=str, default="data/RF00097.fa", help="Path to the RNA sequence data file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=None, help="If not set, use 98th percentile")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--save_dir", type=str, default="saved_models") # default none 
    parser.add_argument("--log_dir", type=str, help="Path to the log file (optional)")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="Generator learning rate")
    parser.add_argument("--lr_c", type=float, default=0.0001, help="Critic learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = init_cuda()

    print("Loading dataset...")
    dataset = FastDatasetRNA(args.data, batch_size=args.batch_size, sequence_length=args.seq_len)
    args.seq_len = dataset.sequence_length
    print(f"Dataset loaded with sequence length: {args.seq_len}")
    print(f"Dataset size: {len(dataset)} samples")

    generator = ResNetGenerator(args.latent_dim, args.seq_len).to(device)
    initialize_weights(generator)
    critic = Critic(args.seq_len).to(device)

    train_wgan_gp(generator, critic, dataset, args, device)


if __name__ == "__main__":
    main()

