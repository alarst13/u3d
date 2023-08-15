import torch
import numpy as np
from numpy import array
import cv2
from video_classification.mypath import Path
from noise_perturbation.perlin_noise import generate_noise
from noise_perturbation.perlin_noise import add_perlin_noise_to_frame
from video_classification.network import C3D_model
from psolib import particle_swarm_optimization as pso
import argparse
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Define the power normalization function P(z) = sign(z) * |z|^alpha
def power_normalization(z, alpha=0.5):
    return torch.sign(z) * torch.abs(z) ** alpha


def center_crop(frame):
    # In the C3D architecture, the input frames are 112x112
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


# Extract intermediate features from the C3D model for the input clip
def intermediate_features(model, clip):
    inputs = np.array(clip).astype(np.float32)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
    inputs = torch.from_numpy(inputs)
    inputs = torch.autograd.Variable(
        inputs, requires_grad=False).to(device)

    with torch.no_grad():
        logits, intermediate_dict = model.forward(inputs)

    return intermediate_dict


def temporal_transformation(perturbation, tau_shift):
    """
    Apply temporal transformation to the perturbation by shifting frames.

    Parameters
    ----------
    perturbation : numpy.ndarray
        Original U3D perturbation of shape (H, W, T), where H is the height, W is the width, and T is the number of frames.
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
    H, W, T = perturbation.shape
    tau_shift = int(tau_shift)  # Ensure tau_shift is an integer
    transformed_perturbation = np.zeros((H, W, T))
    for t in range(T):
        transformed_perturbation[:, :, t] = np.roll(
            perturbation[:, :, t], tau_shift, axis=0)
    return transformed_perturbation


def attack_objective(args, model, input_video_path, params):
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
    features_original = {}
    features_perturbed = {}
    noise_values_3D = []

    # Parameters for noise generation
    num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period = params
    T = args.T  # Number of frames for Perlin noise generation
    epsilon = args.epsilon  # Maximum perturbation allowed for the U3D attack
    alpha = args.alpha  # The alpha parameter for power normalization
    I = args.I  # Number of iterations for sampling.

    cap_original = cv2.VideoCapture(input_video_path)

    # Get the video properties
    frame_width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Generate the Perlin noise for the first T frames
    noise_values_3D = generate_noise(
        T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon)

    clip_original = []
    clip_perturbed = []
    distances = []
    frame_num = 0

    total_distance_expectation = 0.0
    # Sample I random temporal transformations
    for _ in range(I):
        # Generate a random time index (tau) using a uniform distribution
        tau = np.random.uniform(0, T)
        t_sum = 0.0  # Initialize the sum of distances for this iteration

        # Transform the noise values using the temporal transformation
        transformed_perturbation = temporal_transformation(
            noise_values_3D, tau)

        # Loop through the frames in the video
        while True:
            ret_original, frame_original = cap_original.read()

            if not ret_original:
                break

            # Add Perlin noise to the current frame using the precomputed noise values
            t = frame_num % T  # Loop through the noise values
            frame_perturbed = add_perlin_noise_to_frame(
                frame_original, transformed_perturbation[t])

            # Preprocess the frames by resizing and center cropping
            tmp_original = center_crop(cv2.resize(frame_original, (171, 128)))
            tmp_perturbed = center_crop(
                cv2.resize(frame_perturbed, (171, 128)))

            # Apply color normalization
            tmp_original = tmp_original - np.array([[[90.0, 98.0, 102.0]]])
            tmp_perturbed = tmp_perturbed - np.array([[[90.0, 98.0, 102.0]]])

            clip_original.append(tmp_original)
            clip_perturbed.append(tmp_perturbed)

            if len(clip_original) == 16:
                features_original = intermediate_features(
                    model, clip_original)
                features_perturbed = intermediate_features(
                    model, clip_perturbed)

                # Calculate the power normalized distances at each layer
                for layer_name in features_original.keys():
                    distance = torch.norm(power_normalization(
                        features_original[layer_name], alpha) - power_normalization(features_perturbed[layer_name], alpha), p=2)
                    distances.append(distance)

                # Remove the oldest frame from each clip
                clip_original.pop(0)
                clip_perturbed.pop(0)

                t_sum += sum(distances).detach().cpu().item()

            frame_num += 1
            distances = []  # Reset the distances for memory efficiency

        # Calculate the expectation of the total distance as the sum of distances across iterations divided by the number of iterations (I),
        total_distance_expectation += t_sum / I

    # Release video file handles
    cap_original.release()

    print("Params:", params)
    print("Total distance:", total_distance_expectation)

    return total_distance_expectation


def main(args):
    print('Device: {}'.format(device))

    model = C3D_model.C3D(num_classes=101, pretrained=True)
    model.to(device)
    model.eval()

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
        total_distance = attack_objective(args, model, Path.video(), params)
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


if __name__ == '__main__':
    # Default parameters are chosen based on the paper's settings (adjust as needed).
    parser = argparse.ArgumentParser(
        description='Universal 3-Dimensional Perturbations for Black-Box Attacks')
    parser.add_argument('--lb', type=float, nargs=5, default=[1, 2.0, 2.0, 2.0, 1.0],
                        help='Lower bounds for optimization parameters [num_octaves (int), wavelength_x (float), wavelength_y (float), wavelength_t (float), color_period (float)]')
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
    parser.add_argument('--T', type=int, default=5,
                        help='Number of frames for Perlin noise generation')
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help='Maximum perturbation allowed for the U3D attack')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The alpha parameter for power normalization')
    parser.add_argument('--I', type=int, default=5,
                        help='Number of iterations for sampling')

    args = parser.parse_args()
    main(args)
