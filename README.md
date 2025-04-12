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

1. Clone the repo

```bash
git clone https://github.com/Szczerku/RNA-GANerator.git
cd RNA-GANerator
```

2. Create and activate the conda environment

Make sure you have conda installed. Then, create the environment from the provided environment.yml file:

```bash
conda env create -f environment.yml
conda init
conda activate rna_gan_env
```


## Project Structure

```plaintext
RNA-GANerator/
├── data/
│   └── RF00097.fa
├── GAN
├── loaders/
│   └── fasta_data_loader.py
├── models/
│   ├── critic.py
│   ├── embedding.py
│   └── resnet_generator_rna.py
├── Project_schema
├── utils/
│   ├── init_device.py
│   ├── init_weights.py
│   └── noise_generator.py
├── .gitignore
├── environment.yml
├── generate_rna.py
├── README.md
├── run_wgan.py
└── train_wgan_gp.py
```

## Generator Training

To train the WGAN-GP on RNA sequences:

```bash
python run_wgan.py --data data\RF00097.fa
```

optional flags:
| Flag               | Type     | Default               | Description |
|--------------------|----------|-----------------------|-------------|
| `--data`           | `str`    | `data\RF00097.fa`     | Path to the input FASTA file with RNA sequences |
| `--epochs`         | `int`    | `15`                  | Number of training epochs |
| `--batch_size`     | `int`    | `64`                  | Batch size used during training |
| `--seq_len`        | `int`    | *None*                | if not set - uses 98th percentile of dataset |
| `--latent_dim`     | `int`    | `256`                 | Dimension of the latent noise vector |
| `--n_critic`       | `int`    | `5`                   | Number of critic updates per generator update |
| `--lambda_gp`      | `float`  | `10.0`                | Gradient penalty coefficient |
| `--save_dir`       | `str`    | `saved_models/`       | Directory to save trained generator models |
| `--log_dir`        | `str`    | `training_metrics/`   | File path for saving training logs |
| `--lr_g`           | `float`  | `0.0005`              | Learning rate for the generator |
| `--lr_c`           | `float`  | `0.0001`              | Learning rate for the critic |


