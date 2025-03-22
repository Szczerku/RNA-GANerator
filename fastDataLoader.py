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

class fastdatasetRNA(Dataset):
    def __init__(self, file_path, sequence_length=120, only_positive=False):
        self.data = []
        self.sequence_length = sequence_length
        
        for record in SeqIO.parse(file_path, "fasta"):
            header_parts = record.description.split()
            seq_id = header_parts[0]
            # Załóżmy, że rodzina to ostatnia część opisu nagłówka
            seq_type = header_parts[-1] if len(header_parts) > 1 else "Unknown"
            sequence = str(record.seq).upper()

            # Przyjmujemy, że każda sekwencja ma etykietę zależną od jej typu
            # Możesz dostosować ten warunek, jeśli chcesz inne podejście do etykiet
            label = 1  # Przyjmujemy, że wszystkie sekwencje są "pozytywne" (możesz zmienić logikę)

            # Jeśli only_positive=True, pomijamy próbki z etykietą = 0
            if only_positive and label == 0:
                continue

            # Przycinamy lub uzupełniamy sekwencję do wymaganej długości
            sequence = sequence[:self.sequence_length]
            if len(sequence) < self.sequence_length:
                sequence += "N" * (self.sequence_length - len(sequence))
            
            # Konwertujemy T na U przed zastosowaniem IUPAC
            sequence = convert_to_rna(sequence)
            
            # Zastosowanie konwersji IUPAC dla każdego nukleotydu
            sequence = "".join([IUPAC(nucleotide) for nucleotide in sequence])
            
            self.data.append([seq_id, seq_type, sequence, label])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_id, seq_type, sequence, label = self.data[idx]
        one_hot = one_hot_encoding(sequence)
        return torch.tensor(one_hot, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def average_sequence_length(self):
        # Obliczamy średnią długość sekwencji
        total_length = sum(len(seq[2]) for seq in self.data)  # seq[2] to sekwencja
        return total_length / len(self.data)  # Średnia długość

