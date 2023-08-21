import torch
import numpy as np
from numpy import array
import cv2
from noise_perturbation.perlin_noise import generate_noise
from video_classification.network import C3D_model
import os
from psolib import particle_swarm_optimization as pso
# from pyswarm import pso
import argparse
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def load_preprocessed_frames(dataset_dir):
    frame_files = [
        os.path.join(dirpath, filename)
        for dirpath, dirnames, filenames in os.walk(dataset_dir)
        for filename in filenames
        if filename.endswith('.jpg')
    ]

    return frame_files


def load_video_clips(video_frames, clip_size=16):
    num_frames = len(video_frames)
    num_clips = num_frames // clip_size

    video_frames_3d = []

    for i in range(num_clips):
        clip_indices = range(i * clip_size, (i + 1) * clip_size)
        clip_frames = [cv2.imread(video_frames[idx]) for idx in clip_indices]
        video_frames_3d.append(clip_frames)

    return np.array(video_frames_3d)


# Define the power normalization function P(z) = sign(z) * |z|^alpha
def power_normalization(z, alpha=0.5):
    return torch.sign(z) * torch.abs(z) ** alpha


# Extract intermediate features from the C3D model for the input clip
def intermediate_features(model, clip):
    inputs = np.array(clip).astype(np.float32)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
    inputs = torch.from_numpy(inputs)
    inputs = torch.autograd.Variable(
        inputs, requires_grad=False).to(device)

    inputs = inputs.to(device)

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
    total_distance_expectation = 0.0

    # Sample I random temporal transformations
    for _ in range(I):
        # Generate a random time index (tau) using a uniform distribution
        tau = np.random.uniform(0, T)
        t_sum = 0.0  # Initialize the sum of distances for this iteration

        # Transform the noise values using the temporal transformation
        transformed_perturbation = temporal_transformation(
            noise_cycle, tau)

        # Add the noise cycle to the video clip
        clip_perturbed = video_clip + transformed_perturbation

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


def attack_objective(args, model, video_clips, params):
    """
    Calculate the attack objective using Particle Swarm Optimization (PSO).

    Parameters:
        model (torch.nn.Module): The pre-trained DNN model.
        input_video_path (str): Path to the input video.
        params (list): List of U3D parameters for the attack.

    Returns:
        float: The attack objective value.

    The attack objective function maximizes the expectation over the input video frames and time steps,
    subject to a perturbation constraint.

    The optimization problem aims to maximize the sum of the distortion between the original frames and
    the frames modified by the perturbation. The perturbation is applied based on the U3D parameters and
    the time step. The perturbation is subject to a maximum perturbation constraint.

    Mathematically:
        max ξ: ∑_{video, time step} ∑_{dimension} Distortion(original_frame, perturbed_frame; dimension)
        s.t. ξ = Noise(T; strength), ||ξ||_{∞} ≤ ε

    Where:
    - ξ represents the perturbation.
    - Distortion measures the difference between frames along different dimensions.
    - Noise(T; strength) generates noise based on the U3D parameters and time step.
    - ε is the maximum allowed perturbation.
    """

    noise_values_3D = []

    # Parameters for noise generation
    num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period = params
    T = args.T  # Number of frames for Perlin noise generation
    epsilon = args.epsilon  # Maximum perturbation allowed for the U3D attack
    alpha = args.alpha  # The alpha parameter for power normalization
    I = args.iteration  # Number of iterations for sampling.

    # Generate the Perlin noise for the first T frames
    noise_values_3D = generate_noise(T=T, frame_height=112, frame_width=112, num_octaves=num_octaves, wavelength_x=wavelength_x,
                                     wavelength_y=wavelength_y, wavelength_t=wavelength_t, color_period=color_period, epsilon=epsilon)

    # TODO: Handle the case when the video clip length is less than T, e.g., where should the next video clip start?
    # Extend the channels first
    noise_reshaped = noise_values_3D[:, :, :,
                                     np.newaxis]  # add an extra dimension
    noise_cycle = noise_reshaped * np.ones((1, 1, 1, 3))

    # Then cyclically extend along the T axis
    num_frames = len(video_clips[0])
    replication_factor = num_frames // T
    noise_cycle = np.tile(noise_cycle, (replication_factor, 1, 1, 1))

    # If there are remaining frames after tiling, append them
    rm_frames = num_frames - T*replication_factor
    if rm_frames > 0:
        noise_cycle = np.concatenate(
            [noise_cycle, noise_cycle[:rm_frames]], axis=0)

    total_distance_expectation = 0.0
    for video_clip in video_clips:
        total_distance_expectation += calculate_video_distance_expectation(
            model, video_clip, noise_cycle, I, T, alpha)

    # Extract number of videos from directory name
    num_videos = int(os.path.basename(args.dataset_dir).split('_')[-1])

    total_distance_expectation = total_distance_expectation / num_videos

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
    model = C3D_model.C3D(num_classes=num_classes)
    checkpoint = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    frame_files = load_preprocessed_frames(args.dataset_dir)
    clip_size = 16
    video_clips = load_video_clips(frame_files, clip_size)

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
        total_distance = attack_objective(args, model, video_clips, params)
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
    with open('attack_params.txt', 'a') as f:  # Open the file in append mode
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
                        required=True, help='Path to the video dataset')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the pretrained model')
    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], default='h', help='Dataset name: "u" for UCF101 or "h" for HMDB51')
    parser.add_argument('--lb', type=float, nargs=5, default=[
                        1, 2.0, 2.0, 2.0, 1.0], help='Lower bounds for optimization parameters [num_octaves (int), wavelength_x (float), wavelength_y (float), wavelength_t (float), color_period (float)]')
    parser.add_argument('--ub', type=float, nargs=5, default=[5, 180.0, 180.0, 180.0, 60.0],
                        help='Upper bounds for optimization parameters [num_octaves (int), wavelength_x (float), wavelength_y (float), wavelength_t (float), color_period (float)]')
    parser.add_argument('--swarmsize', type=int, default=20,
                        help='Size of the swarm in particle swarm optimization (PSO)')
    parser.add_argument('--omega', type=float, default=0.4,
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
