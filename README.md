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

2. Run the attack script without optional arguments to use default settings (to reproduce paper's results):

```bash
python u3d-attack-C3D.py
```

### Fine-tune the Attack (Optional)

If you wish to fine-tune the behavior and appearance of the perturbation during the black-box attack, you can adjust the optional arguments. Below is a guide to these arguments:

**Optional Arguments:**

- `--lb`: Lower bounds for optimization parameters
  - **Format**: `--lb num_octaves (int) wavelength_x (float) wavelength_y (float) wavelength_t (float) color_period (float)`
  - **Example**: `--lb 1 2.0 2.0 2.0 1.0`

- `--ub`: Upper bounds for optimization parameters
  - **Format**: `--ub num_octaves (int) wavelength_x (float) wavelength_y (float) wavelength_t (float) color_period (float)`
  - **Example**: `--ub 5 180.0 180.0 180.0 60.0`

- `--swarmsize`: Size of the swarm in particle swarm optimization (PSO)
- `--omega`: Inertia weight for particle swarm optimization (PSO)
- `--phip`: Scaling factor for personal best in PSO
- `--phig`: Scaling factor for global best in PSO
- `--maxiter`: Maximum number of iterations in PSO
- `--T`: Number of frames for Perlin noise generation
- `--epsilon`: Maximum perturbation allowed for U3D attack
- `--alpha`: Alpha parameter for power normalization
- `--I`: Number of iterations for sampling


To use the optional arguments and customize the attack parameters, run the following command:

```bash
python u3d-attack-C3D.py --lb <lower_bounds> --ub <upper_bounds> --swarmsize <swarm_size> --omega <omega_value> --phip <phip_value> --phig <phig_value> --maxiter <max_iterations> --T <frames_for_perlin> --epsilon <max_perturbation> --alpha <alpha_value> --I <num_iterations>
```

Feel free to adjust these parameters according to your requirements and desired outcomes.
