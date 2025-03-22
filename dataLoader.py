import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from Bio import SeqIO

# IUPAC encoding with guaranteed T to U conversion
def IUPAC(nucleotide):
    # First convert T to U
    if nucleotide == 'T':
        return 'U'
        
    nucleotide_map = {
        'R': ['A', 'G'],
        'Y': ['C', 'U'],  # Changed T to U here as well
        'S': ['G', 'C'],
        'W': ['A', 'U'],  # Changed T to U here as well
        'K': ['G', 'U'],  # Changed T to U here as well
        'M': ['A', 'C'],
        'B': ['C', 'G', 'U'],  # Changed T to U here as well
        'D': ['A', 'G', 'U'],  # Changed T to U here as well
        'H': ['A', 'C', 'U'],  # Changed T to U here as well
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'U']  # Changed T to U here as well
    }
    
    if nucleotide in nucleotide_map:
        elements = nucleotide_map[nucleotide]
        return random.choice(elements)
    return nucleotide

# One-hot encoding - only for RNA nucleotides (A, C, G, U, N)
nuc = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
    'N': [0, 0, 0, 0]
}

# Convert sequence to RNA format (ensure all T are converted to U)
def convert_to_rna(sequence):
    return sequence.replace('T', 'U')

# One-hot encoding
def one_hot_encoding(sequence):
    # Make sure sequence is in RNA format
    rna_sequence = convert_to_rna(sequence)
    return [nuc[nucleotide] for nucleotide in rna_sequence]

class datasetRNA(Dataset):
    def __init__(self, file_path, family, sequence_length=120, only_positive=False):
        self.data = []
        self.sequence_length = sequence_length
        self.family = family
        
        for record in SeqIO.parse(file_path, "fasta"):
            header_parts = record.description.split()
            seq_id = header_parts[0]
            seq_type = header_parts[1] if len(header_parts) > 1 else "Unknown"
            sequence = str(record.seq).upper()
            label = 1 if seq_type == family else 0
            
            # Jeśli only_positive=True, pomijamy próbki z label=0
            if only_positive and label == 0:
                continue

            # Trim or pad sequence to the specified length
            sequence = sequence[:self.sequence_length]
            if len(sequence) < self.sequence_length:
                sequence += "N" * (self.sequence_length - len(sequence))
            
            # Convert any T to U before applying IUPAC
            sequence = convert_to_rna(sequence)
            
            # Apply IUPAC conversion for each nucleotide
            sequence = "".join([IUPAC(nucleotide) for nucleotide in sequence])
            
            self.data.append([seq_id, seq_type, sequence, label])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_id, seq_type, sequence, label = self.data[idx]
        one_hot = one_hot_encoding(sequence)
        return torch.tensor(one_hot, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# file_path = "/home/michal/Desktop/RNA_Monster/GANbert-RNA/dataset_Rfam_6320_13classes.fasta"
# family = "5S_rRNA"  # Selected family
# batch_size = 32
# sequence_length = 120

# dataset = datasetRNA(file_path, family, sequence_length, only_positive=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Liczba batchy
# num_batches = len(dataloader)
# print(f"Liczba batchy w zbiorze danych: {num_batches}")

# # Check the data in DataLoader
# for inputs, labels in dataloader:
#     print(f"Batch shapes - Inputs: {inputs.shape}, Labels: {labels.shape}")
#     print(f"Label distribution: {labels.sum().item()}/{len(labels)} positive samples")
#     break  # Break after the first batch