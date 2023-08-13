## Setup Instructions

This guide provides step-by-step instructions to set up the environment and use the implementation for the paper [Universal 3-Dimensional Perturbations for Black-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

### 1. Install Anaconda

Begin by installing [Anaconda](https://www.anaconda.com/download) and then navigate to the desired conda environment:

```bash
conda activate <YOUR-ENVIRONMENT>
```

### 2. Install Rust

Install Rust by executing the following commands:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source .bashrc
source "$HOME/.cargo/env"
```

### 3. Clone the Repository

Clone the repository by running:

```bash
git clone https://github.com/alarst13/u3d
```

### 4. Create Rust Bindings

To create Rust bindings for the Perlin noise library and use them in Python, follow these steps:

1. Install `cargo-binstall`:

   ```bash
   cargo install cargo-binstall
   ```

2. Install `maturin` using `cargo-binstall`:

   ```bash
   cargo binstall maturin
   ```

3. Navigate to the `rust/perlin` directory.

4. Build the Rust bindings using `maturin`:

   ```bash
   maturin develop
   ```

### 5. Run the Attack

For this step, work within the `python/` subdirectory:

```bash
python u3d-attack-C3D.py
```
