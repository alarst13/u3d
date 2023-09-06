<div align="center">
  <table align="center">
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/c2559255-3a2e-4c21-a196-3b76d93c83c4"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/bf1b70e0-b72b-4951-9e6e-81e74149f660"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/f8cae79b-067b-4407-96a6-431dead120a5"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4641c4e6-b0a9-4c11-9162-5e833a658bec"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/dfe1fcc6-231b-46f5-bb24-e7cc82391c6a"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/1f2a6f6e-eb3e-42f2-9efd-b5d7e3ea908e"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4050afc3-1930-48f7-a8bb-636887f3fc07"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/69b01438-8dbd-434f-8532-f9439d5629ce"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/c5ca6d0d-826a-4f9f-a07d-128bf7dd7fd6"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4d2af2a4-6e30-4b36-b43f-b15990ea06a5"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/10515241-7934-4803-8679-a310dfc04af9"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/63894eea-e9b5-4dd9-9f2c-456a5868c60f"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/0c14014c-078d-4dbc-a585-a19edcbebfc0"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/0c868631-cd67-46c3-bad1-d5b9940b7ea1"></td>
    </tr>
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/6db8d88e-f2ab-4d70-a2ee-9409b0a899e3"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/912d488c-4b52-41da-b12a-5af7bda8296a"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/53a55aaa-3eb0-4f84-96ab-454bb52fa6fe"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/008d5e15-9041-44a1-b25f-d46ddb5e1b9c"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/cb420b46-b423-466b-bd32-d8f5dee25204"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/a02fb917-e0ef-466d-ac5e-03134a785d2b"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/6e40e649-71cc-434f-9f41-ac8bdd142597"></td>
    </tr>
    <tr>
      <td colspan="7">Example of a perturbed video with U3D. For greater imperceptibility, choose a smaller epsilon.</td>
    </tr>
  </table>
</div>




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

### Step 4: Build Rust Bindings

1. Run `build.sh` to automatically build Rust bindings for both `pso` and `perlin` crates:

```bash
bash build.sh
```

### Step 5: Install Python Libraries

1. **Install PyTorch:** Follow the instructions on the [PyTorch official website](https://pytorch.org/get-started/locally/) to install PyTorch for your specific system.

2. **Install Additional Libraries:** Use the following command to install `numpy`, `opencv`, `tqdm`, and `scikit-learn`:

```bash
conda install numpy tqdm scikit-learn
```
```bash
pip install opencv-python
```

### Step 6: Download Datasets

1. **HMDB51 Dataset:** 
    - **Description:** The HMDB51 dataset is a large human motion database. 
    - **Download:** Access the dataset and download options from the [HMDB51 official page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads).

2. **UCF101 Dataset:** 
    - **Description:** UCF101 is an action recognition dataset consisting of realistic action videos. 
    - **Download:** You can download the dataset from the [UCF101 official website](https://www.crcv.ucf.edu/data/UCF101.php).

Download the datasets to your preferred directory for further processing.

### Step 7: Train the Models

#### **1. Set up your working directory**
First, navigate to the `video-classification` directory located under `python/video_classification`:
```bash
cd python/video_classification
```

#### **2. Preprocess the datasets**
Move to the `dataloaders` directory inside `video_classification`:
```bash
cd dataloaders
```
Now, run the dataset preprocessing script for the HMDB51 dataset:
```bash
python dataset.py --dataset h --root_dir [PATH/TO/HMDB51] --output_dir [PATH/TO/OUTPUT/DIRECTORY]
```

Then, run the dataset preprocessing script for the UCF101 dataset:
```bash
python dataset.py --dataset u --root_dir [PATH/TO/UCF101] --output_dir [PATH/TO/OUTPUT/DIRECTORY]
```


#### **3. Download Pretrained Model for Fine-Tuning**

To enable fine-tuning in our project, we require a pretrained C3D model from the Sports-1M dataset. Follow these steps:

1. **Install `gdown` Package**:
   
   Install the `gdown` package to simplify Google Drive file downloads:

   ```bash
   pip install gdown
   ```

2. **Download Model**:
   
   Use this command to download the model:

   ```bash
   gdown https://drive.google.com/u/0/uc?id=19NWziHWh1LgCcHU34geoKwYezAogv9fX&export=download
   ```

#### **4. Train the models**
Head back to the `video_classification` directory:
```bash
cd ..
```

Now, train the model on the HMDB51 dataset:
```bash
python train.py --dataset h --data_org [PATH/TO/HMDB51] --data_splits [PATH/TO/HMDB51/SPLITS] --pretrained [PATH/TO/PRETRAINED/MODEL]
```

Then, train the model on the UCF101 dataset:
```bash
python train.py --dataset u --data_org [PATH/TO/UCF101] --data_splits [PATH/TO/UCF101/SPLITS] --pretrained [PATH/TO/PRETRAINED/MODEL]
```

In the next steps, we will use the trained model on HMDB51 to optimize the attack and the trained model on UCF101 to evaluate the attack.

### Step 8: Run the Attack
Follow these instructions to prepare data for the attack and run the attack script.

1. Navigate back to the `python/` subdirectory and execute the following command to generate random videos for optimizing the attack:
```bash
python attack_data_prep.py -d [PATH/TO/HMDB51/SPLITS] -o [PATH/TO/OUTPUT/FOLDER] -n [NUMBER/OF/RANDOM/VIDEOS]
```
Please note that the actual number of generated random videos may slightly differ from your choice to maintain the original class distribution. Default is 500.

2. Run the attack script without optional arguments to use default settings (for reproducing the paper's results). Use the pretrained model on HMDB51 for the attack:

```bash
python u3d-attack-C3D.py -d [PATH/TO/RANDOM/VIDEOS] -m [PATH/TO/PRETRAINED/MODEL]
```

### Fine-tune the Attack (Optional)

If you wish to fine-tune the attack, you can adjust the optional arguments. Below is a guide to these arguments:

**Optional Arguments:**

- `--lb`: Lower bounds for optimization parameters
  - **Format**:
    ```
    --lb num_octaves (int) wavelength_x (float) wavelength_y (float) wavelength_t (float) color_period (float)
    ```
  - **Example**:
    ```
    --lb 1 2.0 2.0 2.0 1.0
    ```

- `--ub`: Upper bounds for optimization parameters
  - **Format**:
    ```
    --ub num_octaves (int) wavelength_x (float) wavelength_y (float) wavelength_t (float) color_period (float)
    ```
  - **Example**:
    ```
    --ub 5 180.0 180.0 180.0 60.0
    ```

- `--swarmsize`: Size of the swarm in particle swarm optimization (PSO)
- `--omega`: Inertia weight for particle swarm optimization (PSO)
- `--phip`: Scaling factor for personal best in PSO
- `--phig`: Scaling factor for global best in PSO
- `--maxiter`: Maximum number of iterations in PSO
- `--T`: Number of frames for Perlin noise generation
- `--epsilon`: Maximum perturbation allowed for U3D attack
- `--alpha`: Alpha parameter for power normalization
- `--I`: Number of iterations for sampling
- `--dataset`: Specifies the dataset name. Your options are:
  - 'u' for UCF101
  - 'h' for HMDB51 (default)

To use the optional arguments and customize the attack parameters, run the following command:

```bash
python u3d-attack-C3D.py --lb <lower_bounds> --ub <upper_bounds> --swarmsize <swarm_size> --omega <omega_value> --phip <phip_value> --phig <phig_value> --maxiter <max_iterations> --T <frames_for_perlin> --epsilon <max_perturbation> --alpha <alpha_value> --I <num_iterations> --dataset <dataset_type>
```

### Step 9: Evaluate the Attack
For the final step, follow these instructions to evaluate the attack and generate the results:

1. Navigate to the `noise_perturbation` directory and run `perlin_noise.py` to perform the noise perturbation on the entire UCF101 dataset using the optimized attack parameters from the previous step. The parameters are saved in `attack_params.txt`:

```bash 
cd noise_perturbation
python perlin_noise.py -b [PATH/TO/UCF101/SPLITS] -o [PATH/TO/OUTPUT/FOLDER]
```

2. Navigate back to the `python/` subdirectory and run the following command to evaluate the attack:

```bash
python evaluate_attack.py --dataset u --data_org [PATH/TO/UCF101/SPLITS] --data_prt [PATH/TO/PERTURBED/UCF101/SPLITS] --model [PATH/TO/PRETRAINED/MODEL]
```

If you followed the instructions correctly, you should achieve an **84.59%** success rate on the UCF101 dataset.

## License

U3D is dual-licensed under either of the following, at your discretion:

- **Apache License, Version 2.0**
  - [LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

- **MIT license**
  - [LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT)

Unless explicitly stated otherwise, any contribution intentionally submitted for inclusion in U3D by you shall be dual-licensed as above, without any additional terms or conditions.

## Citation

If you use this implementation or are inspired by the associated paper, please consider citing:

```bibtex
@inproceedings{xie2022universal,
  title={Universal 3-dimensional perturbations for black-box attacks on video recognition systems},
  author={Xie, Shangyu and Wang, Han and Kong, Yu and Hong, Yuan},
  booktitle={2022 IEEE Symposium on Security and Privacy (SP)},
  pages={1390--1407},
  year={2022},
  organization={IEEE}
}
