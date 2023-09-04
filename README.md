<div align="center">
  <table align="center">
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/e34a0895-0f79-4024-a641-059ee4a74fac"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/2634d4ee-16d1-4836-b3e0-d071b5e7074e"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/b7157d09-4f60-4e35-b1f3-80973856c498"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/663ac3f4-ea55-4416-8a69-55a6b91ba15b"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4fe5b5cf-0f6a-4cca-b670-6cd4eb856962"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/44596828-1749-4eea-a505-3f0c77db3329"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/714b4351-cdf6-4628-9a49-21290c94916d"></td>
    <tr>
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/8fe412e6-02dd-4c67-94dd-e23a90203acc"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/577afe3e-f89a-41dd-a1d9-088b561bdd6e"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/6eb7609b-c2dc-4e0f-bfd6-b5f2f4c33e6e"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/32f3d281-113b-47c6-80c9-cf772c42f4b7"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/d0f0801a-e77f-4661-824a-4d6d0f8e5429"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4b7cd996-e0b9-48be-9e64-6f4086afd6d2"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/9c38ce65-626b-45cb-8638-379437e27d86"></td>
    <tr>
    <tr>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/0735699e-7d84-42e5-8c60-001a08d0e865"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4cb9334f-0557-4f07-94ee-c0326816bb37"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/e03c209e-2a3c-406f-9e3b-005a92858f8b"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/fc5a42ce-079c-40d1-a4bd-8ff8ef19364f"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/4ccf9688-b889-4b56-9c4b-905c1819bd67"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/70f24f63-76eb-41e6-bc54-689b7a6ad05b"></td>
      <td><img src="https://github.com/alarst13/u3d/assets/49977238/80f2eb85-f541-4f77-84f1-e82773f7715c"></td>
    </tr>
    <tr>
      <td colspan="23">Example of a perturbed video with U3D. For greater imperceptibility, choose a smaller epsilon.</td>
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

1. Navigate back to the `python/` subdirectory and run the following command:
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
cd noise_perturbation python perlin_noise.py -b [PATH/TO/UCF101] -o [PATH/TO/OUTPUT/FOLDER]
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
