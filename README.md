<!-- # GANbert-RNA -->
<!-- ## Project Schema -->
<!-- ![Diagram](Project_schema/project.drawio.svg) -->

# RNA-GANerator: ncRNA Sequence Generation using GANs(WGAN-GP)
![Logo](Project_schema/logos.png)

**RNA-GANerator** is a tool for generating biologically plausible RNA sequences using Generative Adversarial Networks (GANs). It's designed for researchers and developers working in bioinformatics, synthetic biology, or machine learning applied to genomics.

## Features

- Generate RNA sequences of customizable length
- Train on custom datasets in FASTA format
- Built with PyTorch
- Easy-to-use

## Instalation

Clone the repo

```bash
git clone https://github.com/Szczerku/RNA-GANerator.git
cd RNA-GANerator
conda env create -f environment.yml
conda init
conda activate rna_gan_env
```
## Project Structure

```plaintext
RNA-GANerator/
├── data/
│   └── RF00097.fa
├── Project_schema/
│   ├── logos.png
│   └── project.drawio.svg
├── environment.yml
├── README.md
└── run_wgan.py
```

## Generator training

```bash
python run_wgan.py --data data\RF00097.fa
```

optional flags:

