This repository provides an implementation for the paper [Universal 3-Dimensional Perturbations for Black-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

## Setup Instructions

This repository provides an implementation for the paper [Universal 3-Dimensional Perturbations for Black-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

## Setup Instructions

To use the provided implementation, you'll need to set up Rust bindings for the Perlin noise library and ensure you have Anaconda installed. Follow these steps:

### 1. Install Rust

Rust is required to build the Rust bindings for the Perlin noise library. If you don't have Rust installed, you can do so by following these steps:

#### Install Rust using Rustup

1. Visit the official Rust website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
2. Follow the installation instructions based on your operating system.

### 2. Install Anaconda

Anaconda provides a convenient way to manage environments and dependencies. If you don't have Anaconda installed, you can do so by following these steps:

#### Install Anaconda

1. Visit the official Anaconda website: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Download and install Anaconda based on your operating system.
3. Open a terminal and create a new Anaconda environment by running:
   ```bash
   conda create -n perlin python=3.10

### 2. Install Anaconda

To create Rust bindings for the Perlin noise library and use them in Python, follow these steps:

1. If you don't have `cargo-binstall` installed, run the following command to install it:

   ```bash
   cargo install cargo-binstall
   ```

2. Once `cargo-binstall` is installed, run the following command to install `maturin`:
   ```bash
   cargo binstall maturin
   ```

3. Then, navigate to the `rust/perlin` directory.
4. In the `rust/perlin` directory, run the following command:
   ```bash
   maturin develop
   ```
