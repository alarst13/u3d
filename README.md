# U3D

This repository contains an implementation based on the paper [Universal 3-Dimensional Perturbations for Black-Box Attacks on Video Recognition Systems](https://arxiv.org/pdf/2107.04284.pdf).

## Setup Instructions

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

### Step 4: Build Rust Bindings

1. Run `build.sh` to automatically build Rust bindings for both `pso` and `perlin` crates:

```bash
./build.sh
```

### Step 5: Run the Attack

1. For the final step, move to the `python/` subdirectory:

```bash
cd python
```

2. Run the attack script with optional arguments to adjust parameters (if needed):

'''bash
python u3d-attack-C3D.py --lb &lt;lower_bounds&gt; --ub &lt;upper_bounds&gt; --swarmsize &lt;swarm_size&gt; --omega &lt;omega_value&gt; --phip &lt;phip_value&gt; --phig &lt;phig_value&gt; --maxiter &lt;max_iterations&gt; --T &lt;frames_for_perlin&gt; --epsilon &lt;max_perturbation&gt; --alpha &lt;alpha_value&gt; --I &lt;num_iterations&gt;
'''

3. Note: Default parameters are chosen based on the paper's settings. To reproduce the same results as in the paper, you can run the attack script without adding any optional arguments:
```bash
python u3d-attack-C3D.py
```
