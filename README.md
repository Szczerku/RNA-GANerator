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

To train the WGAN-GP model on RNA sequences, you can use the default configuration or customize it with your own dataset and hyperparameters.

1. Quick Start

Simply run the following command to start training with the default dataset and parameters:
```bash
python run_wgan.py
```

2. Custom Dataset

If you want to use your own RNA dataset in FASTA format, place it inside the data/ directory. Below is an example using a custom file named RF00097.fa:
```bash
python run_wgan.py --data data\RF00097.fa
```

3. Custom Training Options

You can further customize training parameters such as sequence length, batch size, and more:
```bash
python run_wgan.py --data data\RF00097.fa --seq_len 120 --batch_size 32
```

Available Options:
| Flag               | Type     | Default               | Description |
|--------------------|----------|-----------------------|-------------|
| `--data`           | `str`    | `data\RF00097.fa`     | Path to the input FASTA file with RNA sequences |
| `--epochs`         | `int`    | `15`                  | Number of training epochs |
| `--batch_size`     | `int`    | `64`                  | Batch size used during training |
| `--seq_len`        | `int`    | *None*                | If not set - uses 98th percentile of dataset |
| `--latent_dim`     | `int`    | `256`                 | Dimension of the latent noise vector |
| `--n_critic`       | `int`    | `5`                   | Number of critic updates per generator update |
| `--lambda_gp`      | `float`  | `10.0`                | Gradient penalty coefficient |
| `--save_dir`       | `str`    | `saved_models/`       | Directory to save trained generator models |
| `--log_dir`        | `str`    | *optional*            | File path for saving training logs |
| `--lr_g`           | `float`  | `0.0005`              | Learning rate for the generator |
| `--lr_c`           | `float`  | `0.0001`              | Learning rate for the critic |


## Generating New Sequences
Co 100 batchy zapisywany jest model w formacie .... do folderu saved_models lub do tego do ktorego podales w fladze.
te modele moga byc nastepnie uzyte do generowania sekwencji RNA.

| Flag                 | Type     | Default              | Description |
|----------------------|----------|----------------------|-------------|
| `--model_path`       | `str`    | **(required)**       | Path to the trained generator model |
| `--total_sequences`  | `int`    | `1000`               | Number of sequences to generate |
| `--sequence_length`  | `int`    | `109`                | Length of the generated sequences |
| `--output_dir`       | `str`    | `generated_fasta`    | Directory to save generated sequences |
| `--latent_dim`       | `int`    | `256`                | Latent dimension for noise generation |
| `--device`           | `str`    | `cpu`                | Device to use for generation (`cpu` or `cuda`) |