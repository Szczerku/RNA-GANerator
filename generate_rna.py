import os
import numpy as np
import torch
import argparse
from utils.noise_generator import generate_noise
from models.resnet_generator_rna import ResNetGenerator

def save_generated_fasta(generator, total_sequences, output_dir="sequences", latent_dim=256, device="cpu"):
    generator.to(device)
    generator.eval()
    
    os.makedirs(output_dir, exist_ok=True)
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

    with torch.no_grad():
        z = generate_noise(latent_dim, total_sequences).to(device)
        generated_sequences = generator(z).cpu().numpy()
        fasta_path = os.path.join(output_dir, "generated_sequences.fasta")

        with open(fasta_path, "w") as f:
            for i, seq in enumerate(generated_sequences):
                decoded_seq = decode_sequence(seq)
                f.write(f">Generated_{i+1}\n{decoded_seq}\n")
        
        print("Successfully saved generated sequences to", fasta_path)

def main():
    parser = argparse.ArgumentParser(description="Generate RNA sequences using a trained generator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained generator model")
    parser.add_argument("--total_sequences", type=int, default=1000, help="Number of sequences to generate")
    parser.add_argument("--sequence_length", type=int, default=109, help="Length of the generated sequences")
    parser.add_argument("--output_dir", type=str, default="generated_fasta", help="Directory to save generated sequences")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension for noise generation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for generation (cpu or cuda)")
    args = parser.parse_args()

    generator = ResNetGenerator(args.latent_dim, args.sequence_length).to(args.device)
    generator.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))


    save_generated_fasta(
        generator,
        total_sequences=args.total_sequences,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        device=args.device
    )

if __name__ == "__main__":
    main()
