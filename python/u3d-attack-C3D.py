import torch
import os
import numpy as np
import cv2
from video_classification.mypath import Path
from noise_perturbation.perlin_noise import perturb_video
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
def attack_objective(model, alpha, input_video_path, output_video_path):
    features_original = {} 
    features_perturbed = {} 
    
    # Perturb the video if it has not been perturbed yet
    if not os.path.exists(output_video_path):
        # Parameters for noise generation
        num_octaves = 2
        wavelength_x = 16.0
        wavelength_y = 16.0
        wavelength_t = 8.0
        color_period = 2.0
        # Number of frames for Perlin noise generation (adjust as needed)
        T = 5
        perturbed_video = perturb_video(num_octaves, wavelength_x, wavelength_y,
                                        wavelength_t, color_period, T, input_video_path, output_video_path)
    else:
        perturbed_video = output_video_path

    cap_original = cv2.VideoCapture(input_video_path)
    cap_perturbed = cv2.VideoCapture(perturbed_video)

    clip_original = []
    clip_perturbed = []
    num = 0
    distances = []

    # Loop through frames in the videos
    while True:
        ret_original, frame_original = cap_original.read()
        ret_perturbed, frame_perturbed = cap_perturbed.read()

        if not ret_original or not ret_perturbed:
            break

        # Preprocess frames by resizing and center cropping
        tmp_original = center_crop(cv2.resize(frame_original, (171, 128)))
        tmp_perturbed = center_crop(cv2.resize(frame_perturbed, (171, 128)))

        # Apply color normalization
        tmp_original = tmp_original - np.array([[[90.0, 98.0, 102.0]]])
        tmp_perturbed = tmp_perturbed - np.array([[[90.0, 98.0, 102.0]]])

        clip_original.append(tmp_original)
        clip_perturbed.append(tmp_perturbed)

        if len(clip_original) == 16 and len(clip_perturbed) == 16:
            num += 1
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

    # Release video file handles
    cap_original.release()
    cap_perturbed.release()

    total_distance = sum(distances)

    return total_distance


if __name__ == '__main__':
    # Load the pre-trained C3D model
    print('Device: {}'.format(device))
    model = C3D_model.C3D(num_classes=101, pretrained=True)
    model.to(device)
    model.eval()

    print(attack_objective(model, 0.5, Path.video(), Path.perturbed_video()))
