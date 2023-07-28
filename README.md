This repository provides an implementation for the paper [Universal 3-Dimensional Perturbations forBlack-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

## Setup Instructions

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
