import torch
import numpy as np
from numpy import array
from video_classification.dataloaders.u3d_dataset import VideoClipsDataset
from torch.utils.data import DataLoader
from noise_perturbation.perlin_noise import generate_noise
from video_classification.network import C3D_model
from psolib import particle_swarm_optimization as pso
# from pyswarm import pso
import argparse
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Define the power normalization function P(z) = sign(z) * |z|^alpha
def power_normalization(z, alpha=0.5):
    return torch.sign(z) * torch.abs(z) ** alpha


# Extract intermediate features from the C3D model for the input clip
def intermediate_features(model, inputs):
    with torch.no_grad():
        logits, intermediate_dict = model.forward(inputs)

    return intermediate_dict


def temporal_transformation(perturbation, tau_shift):
    """
    Apply temporal transformation to the perturbation by shifting frames.

    Parameters
    ----------
    perturbation : numpy.ndarray
        Original U3D perturbation of shape (T, H, W), where T is the number of frames, H is the height, and W is the width.
    tau_shift : int
        The temporal shift value for the transformation.

    Returns
    -------
    numpy.ndarray
        Transformed U3D perturbation with frames shifted by tau_shift along the temporal axis.

    Notes
    -----
    This function performs a temporal shift on the input perturbation by rolling frames along the temporal axis.
    """
    tau_shift = int(tau_shift)  # Ensure tau_shift is an integer

    # Shift the entire frames along the T axis
    transformed_perturbation = np.roll(perturbation, tau_shift, axis=0)

    return transformed_perturbation


def calculate_video_distance_expectation(model, video_clip, noise_cycle, I, T, alpha):
    video_clip = video_clip.to(device)

    total_distance_expectation = 0.0

    # Sample I random temporal transformations
    for _ in range(I):
        tau = np.random.uniform(0, T)
        t_sum = 0.0  # Initialize the sum of distances for this iteration

        # Transform the noise values using the temporal transformation
        transformed_perturbation = temporal_transformation(
            noise_cycle, tau)

        # Add the noise cycle to the video clip
        transformed_perturbation = torch.tensor(
            transformed_perturbation, device=device)
        clip_perturbed = torch.clamp(
            video_clip + transformed_perturbation, 0, 255)
        clip_perturbed = clip_perturbed.float()

        # Calculate intermediate features for original and perturbed frames
        features_original = intermediate_features(model, video_clip)
        features_perturbed = intermediate_features(model, clip_perturbed)

        # Calculate the power normalized distances at each layer
        for layer_name in features_original.keys():
            distance = torch.norm(
                power_normalization(features_original[layer_name], alpha) -
                power_normalization(features_perturbed[layer_name], alpha),
                p=2
            )
            t_sum += distance.detach().cpu().item()

        # Calculate the expectation of the total distance as the sum of distances across iterations divided by the number of iterations (I),
        total_distance_expectation += t_sum / I

    return total_distance_expectation


def attack_objective(args, model, dataloader, params):
    """
    Computes the attack objective using Particle Swarm Optimization (PSO).

    The goal of this function is to optimize an objective that measures the distortion between
    original video frames and those modified by a perturbation. The perturbation is applied based 
    on certain parameters and a time step, and is subject to constraints on its magnitude.

    Parameters:
        args: Miscellaneous arguments.
        model (torch.nn.Module): The pre-trained DNN model.
        dataloader: DataLoader for accessing video frames.
        params (list): Parameters guiding the U3D perturbation.

    Returns:
        float: The attack objective value.

    Mathematical Formulation:
        max ξ: ∑_{video, time step} ∑_{dimension} Distortion(original_frame, perturbed_frame; dimension)
        s.t. ξ = Noise(T; strength), ||ξ||_{∞} ≤ ε

    Where:
    - ξ denotes the perturbation.
    - Distortion represents the differential measure between frames across various dimensions.
    - Noise(T; strength) generates noise influenced by the U3D parameters and time step.
    - ε defines the maximum allowable perturbation magnitude.
    """

    # Parameters for noise generation
    num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period = params
    T = args.T  # Number of frames for Perlin noise generation
    epsilon = args.epsilon  # Maximum perturbation allowed for the U3D attack
    alpha = args.alpha  # The alpha parameter for power normalization
    I = args.iteration  # Number of iterations for sampling.
    total_distance_expectation = 0.0

    # Generate the Perlin noise for the first T frames
    noise_values = generate_noise(T=T, frame_height=112, frame_width=112, num_octaves=num_octaves, wavelength_x=wavelength_x,
                                  wavelength_y=wavelength_y, wavelength_t=wavelength_t, color_period=color_period, epsilon=epsilon)

    # TODO: Handle the case when T is not 16
    for video_clip, _ in dataloader:
        total_distance_expectation += calculate_video_distance_expectation(
            model, video_clip, noise_values, I, T, alpha)

    total_distance_expectation = total_distance_expectation / dataloader.__len__()

    print("Params:", params)
    print("Total distance:", total_distance_expectation)

    return total_distance_expectation


def main(args):
    print('Device: {}'.format(device))

    dataset = 'ucf101' if args.dataset == 'u' else 'hmdb51'
    if dataset == 'ucf101':
        num_classes = 101
    elif dataset == 'hmdb51':
        num_classes = 51
    else:
        raise ValueError("Unsupported dataset name")

    # init model
    model = C3D_model.C3D(num_classes=num_classes).to(device)
    checkpoint = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    # model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = VideoClipsDataset(videos_dir=args.dataset_dir, clip_len=16)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    def objective_function(params):
        """
        Calculate the negative total distance for PSO optimization.

        Parameters
        ----------
        params : list
            List of parameters for the U3D attack.
            [num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period]

        Returns
        -------
        float
            Negative total distance (PSO minimizes, so we negate for maximization).
        """
        params = [round(p) if i == 0 else p for i, p in enumerate(params)]
        total_distance = attack_objective(args, model, dataloader, params)
        # Negate because PSO minimizes, and we want to maximize the distance
        return -total_distance

    best_params, _ = pso(
        objective_function,
        lb=array(args.lb),
        ub=array(args.ub),
        swarmsize=args.swarmsize,
        omega=args.omega,
        phip=args.phip,
        phig=args.phig,
        maxiter=args.maxiter,
        debug=True)

    # Save the best results in a text file
    with open('attack_params.txt', 'a') as f:
        f.write("PSO Settings:\n")
        f.write("swarmsize: {}\n".format(args.swarmsize))
        f.write("omega: {}\n".format(args.omega))
        f.write("phip: {}\n".format(args.phip))
        f.write("phig: {}\n".format(args.phig))
        f.write("maxiter: {}\n".format(args.maxiter))
        f.write("Best Parameters:\n")
        f.write("num_octaves: {}\n".format(best_params[0]))
        f.write("wavelength_x: {}\n".format(best_params[1]))
        f.write("wavelength_y: {}\n".format(best_params[2]))
        f.write("wavelength_t: {}\n".format(best_params[3]))
        f.write("color_period: {}\n".format(best_params[4]))
        f.write("\n\n")  # Add an empty line for separation

    print("Best Parameters and PSO settings saved in 'attack_params.txt'")


if __name__ == '__main__':
    # Default parameters are chosen based on the paper's settings (adjust as needed).
    parser = argparse.ArgumentParser(
        description='Universal 3-Dimensional Perturbations for Black-Box Attacks')
    parser.add_argument('--dataset_dir', '-d', type=str,
                        required=True, help='Path to the random video dataset')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the pretrained model')
    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], required=True, help='Dataset name: "u" for UCF101 or "h" for HMDB51')
    parser.add_argument('--lb', type=float, nargs=5, default=[
                        1, 2.0, 2.0, 2.0, 1.0], help='Lower bounds for optimization parameters [num_octaves (int), wavelength_x (float), wavelength_y (float), wavelength_t (float), color_period (float)]')
    parser.add_argument('--ub', type=float, nargs=5, default=[5, 180.0, 180.0, 180.0, 60.0],
                        help='Upper bounds for optimization parameters [num_octaves (int), wavelength_x (float), wavelength_y (float), wavelength_t (float), color_period (float)]')
    parser.add_argument('--swarmsize', type=int, default=20,
                        help='Size of the swarm in particle swarm optimization (PSO)')
    parser.add_argument('--omega', type=float, default=1.2,
                        help='Inertia weight for particle swarm optimization (PSO)')
    parser.add_argument('--phip', type=float, default=2.0,
                        help='Scaling factor for personal best in particle swarm optimization (PSO)')
    parser.add_argument('--phig', type=float, default=2.0,
                        help='Scaling factor for global best in particle swarm optimization (PSO)')
    parser.add_argument('--maxiter', type=int, default=40,
                        help='Maximum number of iterations in particle swarm optimization (PSO)')
    parser.add_argument('--T', type=int, default=16,
                        help='Number of frames for Perlin noise generation')
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help='Maximum perturbation allowed for the U3D attack')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The alpha parameter for power normalization')
    parser.add_argument('--iteration', '-I', type=int,
                        default=5, help='Number of iterations for sampling')

    args = parser.parse_args()
    main(args)
