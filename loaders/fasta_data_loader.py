import torch
from torch.utils.data import Dataset, DataLoader
import random
import math
import numpy as np
from Bio import SeqIO

"""
fasta_data_loader.py 

is a module for loading RNA sequences from FASTA files.
It includes functionality for converting nucleotide sequences to RNA format,
using IUPAC codes, and one-hot encoding the sequences.

"""

# Convert sequence to RNA format (ensure all T are converted to U)
def convert_to_rna(sequence):
    return sequence.replace('T', 'U')

"""
IUPAC nucleotide code converter for RNA sequences.
Based on: https://www.bioinformatics.org/sms/iupac.html
"""

def iupac_converter(nucleotide):

    nucleotide_map = {
        'R': ['A', 'G'],
        'Y': ['C', 'U'],
        'S': ['G', 'C'],
        'W': ['A', 'U'],
        'K': ['G', 'U'],
        'M': ['A', 'C'],
        'B': ['C', 'G', 'U'],
        'D': ['A', 'G', 'U'],
        'H': ['A', 'C', 'U'],
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'U']
    }
    
    if nucleotide in nucleotide_map:
        elements = nucleotide_map[nucleotide]
        return random.choice(elements)
    return nucleotide


# One-hot encoding
def one_hot_encoding(sequence):
    nuc = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1],
    }
    # Make sure sequence is in RNA format
    return [nuc[nucleotide] for nucleotide in sequence]


class FastDatasetRNA(Dataset):
    def __init__(self, file_path, sequence_length=None, batch_size=32, shuffle=True, drop_last=True):

        self.data = []
        self.sequence_length = sequence_length
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._load_sequences()
        self.dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    def _load_sequences(self):
        raw_sequences = []
        
        for record in SeqIO.parse(self.file_path, "fasta"):
            header_parts = record.description.split()
            seq_id = header_parts[0]
            seq_type = header_parts[-1] if len(header_parts) > 1 else "Unknown"
            sequence = str(record.seq).upper()
            raw_sequences.append((seq_id, seq_type, sequence))

        if self.sequence_length is None:
            lengths = [len(seq[2]) for seq in raw_sequences]
            self.sequence_length = math.ceil(np.percentile(lengths, 98))

        for seq_id, seq_type, sequence in raw_sequences:
            sequence = sequence[:self.sequence_length]
            if len(sequence) < self.sequence_length:
                sequence += 'N' * (self.sequence_length - len(sequence))
            
            sequence = convert_to_rna(sequence)
            sequence = ''.join(iupac_converter(nuc) for nuc in sequence)

            self.data.append((seq_id, seq_type, sequence))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_id, seq_type, sequence = self.data[idx]
        one_hot = one_hot_encoding(sequence)
        return torch.tensor(one_hot, dtype=torch.float32)

    def __iter__(self):
        return iter(self.dataloader)

    def percentile_length(self, percentile):
        lengths = [len(seq[2]) for seq in self.data]
        return np.percentile(lengths, percentile)
    

# file_path = r"C:\Users\michi\Desktop\RNA_Monster\data\RF00097.fa"

# fastdataset = FastDatasetRNA(file_path, batch_size = 64)
# print(fastdataset[0])
# print(f"Średnia długość sekwencji: {fastdataset.percentile_length(98)}")
# print(f"Dataset loaded: {len(fastdataset)} samples, batch size: {fastdataset.batch_size}")

