import torch
import numpy as np
import cv2
from video_classification.mypath import Path
from noise_perturbation.perlin_noise import generate_noise
from noise_perturbation.perlin_noise import add_perlin_noise_to_frame
from video_classification.network import C3D_model
from pyswarm import pso
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Define the power normalization function P(z) = sign(z) * |z|^alpha
def power_normalization(z, alpha):
    return torch.sign(z) * torch.abs(z) ** alpha


def center_crop(frame):
    # In the C3D architecture, the input frames are 112x112
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


# Exctract intermediate features from the C3D model with the given clip
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


# Maximize the distance between original and perturbed videos
def attack_objective(model, alpha, input_video_path, params):
    features_original = {}
    features_perturbed = {}
    noise_values_3D = []

    # Parameters for noise generation
    num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period = params
    T = 5  # Number of frames for Perlin noise generation (adjust as needed)
    epsilon = 8.0  # Maximum perturbation allowed (epsilon) for the U3D attack

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

    # Loop through the frames in the video
    while True:
        ret_original, frame_original = cap_original.read()

        if not ret_original:
            break

        # Add Perlin noise to the current frame using the precomputed noise values
        t = frame_num % T  # Loop through the noise values
        frame_perturbed = add_perlin_noise_to_frame(
            frame_original, noise_values_3D[t])

        # Preprocess the frames by resizing and center cropping
        tmp_original = center_crop(cv2.resize(frame_original, (171, 128)))
        tmp_perturbed = center_crop(cv2.resize(frame_perturbed, (171, 128)))

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

        frame_num += 1

    # Release video file handles
    cap_original.release()

    total_distance = sum(distances)
    total_distance = total_distance.detach().cpu().item()

    print("Params:", params)
    print("Total distance:", total_distance)

    return total_distance


if __name__ == '__main__':
    print('Device: {}'.format(device))

    # Load the pre-trained C3D model
    model = C3D_model.C3D(num_classes=101, pretrained=True)
    model.to(device)
    model.eval()

    # lower bouns
    lb = [
        1,       # num_octaves
        2.0,     # wavelength_x
        2.0,     # wavelength_y
        2.0,     # wavelength_t
        1.0      # color_period
    ]
    # upper bounds
    ub = [
        5,       # num_octaves
        180.0,   # wavelength_x
        180.0,   # wavelength_y
        180.0,   # wavelength_t
        60.0     # color_period
    ]

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
        total_distance = attack_objective(
            model, 0.5, Path.video(), params)
        # Negate because PSO minimizes, and we want to maximize the distance
        return -total_distance

    # Run PSO optimization
    # Best after iteration 40: [  5.           2.         180.          49.35137797   1.        ] -3859360.25
    best_params, _ = pso(
        objective_function,
        lb,
        ub,
        swarmsize=20,
        omega=1.2,   # Inertia weight
        phip=2.0,    # Scaling factor for personal best
        phig=2.0,    # Scaling factor for global best
        maxiter=40,
        debug=True)