# ladowanie pliku fasta
# podzial na rodziny 
# wybor rodziny do labelowania
# zamiana na nukleotydy - IUPAC 
# output batch tensorow z one hot encodingiem


# 1. wczytanie pliku fasta

import pandas as pd
from Bio import SeqIO
import random

file_path = "dataset_Rfam_6320_13classes.fasta"

# Odczytanie wszystkich sekwencji z pełnym opisem

data = []

for record in SeqIO.parse(file_path, "fasta"):
    header_parts = record.description.split()  # Rozdzielenie nagłówka
    seq_id = header_parts[0]  # Pierwsza część to np. RF00001_M28193_1_1-119
    seq_type = header_parts[1] if len(header_parts) > 1 else "Unknown"  # Druga część np. 5S_rRNA
    sequence = str(record.seq)
    label = 0  # Domyślnie ustawiamy label na 0
    
    data.append([seq_id, seq_type, sequence, label])

# Tworzenie DataFrame
df = pd.DataFrame(data, columns=["ID", "Type", "Sequence", "Label"])

#print all types
#print(df["Type"].unique())
#  wybor rodziny dla ktorej label bedzie 1 a nie 0
# ALL familes: ['5S_rRNA' '5_8S_rRNA' 'tRNA' 'ribozyme' 'CD-box' 'miRNA' 'Intron_gpI' 'Intron_gpII' 'HACA-box' 'riboswitch' 'IRES' 'leader' 'scaRNA']

family = "5S_rRNA"
df.loc[df["Type"] == family, "Label"] = 1

# stats for family with label 1 for example mean length ITS NECESSARY TO SET THIS VALUE IN GENERATOR - SEQUENCE LENGTH
print(df[df["Label"] == 1]["Sequence"].apply(len).mean()) # 120.0


#IUPAC encoding

# Mapowanie nukleotydów na jedna z zasad
def IPUAC(nucleotide):
    nucleotide_map = {
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'S': ['G', 'C'],
        'W': ['A', 'T'],
        'K': ['G', 'T'],
        'M': ['A', 'C'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'T']
    }
    if nucleotide == 'T':  
        return 'U'
    
    if nucleotide in nucleotide_map:
        elements = nucleotide_map[nucleotide]
        return random.choice(elements)

    return nucleotide

# Zamiana sekwencji na IUPAC
df["Sequence"] = df["Sequence"].apply(lambda x: "".join([IPUAC(nucleotide) for nucleotide in x]))


nuc = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1]
}

# one hot encoding
def one_hot_encoding(sequence):
    return [nuc[nucleotide] for nucleotide in sequence]

# Tworzenie tensora z one hot encodingiem
# batch_size - rozmiar batcha   
batch_size = 32

# tenosr 

