# U3D

This repository contains an implementation based on the paper [Universal 3-Dimensional Perturbations for Black-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

## Setup Instructions

This guide outlines the steps to set up your environment and use the provided implementation effectively.

### Step 1: Install Anaconda

1. Start by installing [Anaconda](https://www.anaconda.com/download).
2. Activate your desired conda environment:

```bash
conda activate <YOUR-ENVIRONMENT>
```

### Step 2: Install Rust

1. Install Rust by executing the following commands:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source .bashrc
source "$HOME/.cargo/env"
```

### Step 3: Clone the Repository

1. Clone this repository using Git:

```bash
git clone https://github.com/alarst13/u3d
cd u3d
```

### Step 4: Create Rust Bindings

1. Install `cargo-binstall`:

```bash
cargo install cargo-binstall
```

2. Install `maturin` using `cargo-binstall`:

```bash
cargo binstall maturin
```

3. Navigate to the `rust/perlin` directory:

```bash
cd rust/perlin
```

4. Build the Rust bindings with `maturin`:

```bash
maturin develop
```

### Step 5: Run the Attack

1. For the final step, move to the `python/` subdirectory:

```bash
cd ../python
```

2. Run the attack script:

```bash
python u3d-attack-C3D.py
```
