from perlin import perlin_noise
import cv2
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from typing import Generator
from functools import partial
import argparse
import logging


def save_video(args):
    """Save frames to a video folder."""
    video_frames = args['video_frames']
    output_video_folder = args['output_video_folder']

    # Create output folder if it doesn't exist
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    # Save each frame to the specified folder
    for idx, frame in enumerate(video_frames):
        frame_filename = os.path.join(
            output_video_folder, f"frame_{idx:04d}.jpg")
        try:
            cv2.imwrite(frame_filename, frame)
        except Exception as e:
            logging.error(f"Error saving frame: {e}")


def get_video_frame_paths(video_folder_path):
    """Retrieve paths of all .jpg frames in a given folder."""
    return [os.path.join(video_folder_path, fname) for fname in sorted(os.listdir(video_folder_path)) if fname.endswith('.jpg')]


def load_video_frames(video_folder):
    """Load video frames from a folder."""
    frame_paths = get_video_frame_paths(video_folder)
    return [cv2.imread(frame_path) for frame_path in frame_paths]


def video_folder_paths(base_path: str) -> Generator[str, None, None]:
    """Generate paths to all video folders within the base path."""
    # Get folders representing each action category
    action_categories = [os.path.join(base_path, action) for action in os.listdir(
        base_path) if os.path.isdir(os.path.join(base_path, action))]

    # Yield individual video folder paths
    for action_category in action_categories:
        for video_folder in os.listdir(action_category):
            if os.path.isdir(os.path.join(action_category, video_folder)):
                yield os.path.join(action_category, video_folder)


def process_video_folder(video_folder: str, num_octaves: int, wavelength_x: float, wavelength_y: float,
                         wavelength_t: float, color_period: float, T: int, epsilon: float,
                         base_path: str, output_base_path: str) -> None:
    """Process a single video folder: load, perturb, and save the video."""
    frames = load_video_frames(video_folder)
    perturbed_video = perturb_single_video((frames, T, num_octaves, wavelength_x, wavelength_y,
                                            wavelength_t, color_period, epsilon))
    relative_path = os.path.relpath(video_folder, base_path)
    output_video_folder = os.path.join(output_base_path, relative_path)
    save_video({
        'video_frames': perturbed_video,
        'output_video_folder': output_video_folder
    })


# TODO: Optimize the Perlin noise generation to accept vectorized inputs
def generate_noise(T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon):
    """Generate 3D Perlin noise for a video."""
    noise_values_3D = np.zeros((T, frame_height, frame_width))

    # Populate the noise array with perlin noise values
    for frame_num in range(T):
        for y in range(frame_height):
            for x in range(frame_width):
                noise_value = perlin_noise(
                    x, y, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon)
                noise_values_3D[frame_num][y][x] = noise_value
    return noise_values_3D


def add_perlin_noise_to_frame(frame, noise_values):
    """Add Perlin noise to a single video frame."""
    noisy_frame = frame + noise_values[:, :, np.newaxis]
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)


def perturb_single_video(args):
    """Perturb a single video with Perlin noise."""
    video, T, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon = args
    perturbed_video = []

    frame_width = video[0].shape[1]
    frame_height = video[0].shape[0]
    noise_values_3D = generate_noise(
        T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon)
    for frame_num in range(len(video)):
        t = frame_num % T
        noisy_frame = add_perlin_noise_to_frame(
            video[frame_num], noise_values_3D[t])
        perturbed_video.append(noisy_frame)

    return perturbed_video


def perturb_videos(videos, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, T, epsilon):
    """Perturb all given videos with Perlin noise."""
    args_list = [(video, T, num_octaves, wavelength_x, wavelength_y,
                  wavelength_t, color_period, epsilon) for video in videos]

    # Use multiprocessing to process the videos in parallel
    with Pool(cpu_count()) as pool:
        perturbed_videos = list(pool.imap(perturb_single_video, args_list))

    return perturbed_videos


def main(args):
    """Main function to process video folders."""
    # Check if base path exists
    if not os.path.exists(args.base_path):
        print(f"Base path {args.base_path} does not exist!")
        exit(1)

    # Ensure the output directory exists
    if not os.path.exists(args.output_base_path):
        os.makedirs(args.output_base_path)

    # Get a count for progress bar tracking
    total_folders = sum(1 for _ in video_folder_paths(args.base_path))

    # Process and perturb all videos using multiprocessing
    with Pool(cpu_count()) as pool:
        list(pool.map(partial(process_video_folder,
                              num_octaves=args.num_octaves,
                              wavelength_x=args.wavelength_x,
                              wavelength_y=args.wavelength_y,
                              wavelength_t=args.wavelength_t,
                              color_period=args.color_period,
                              T=args.T,
                              epsilon=args.epsilon,
                              base_path=args.base_path,
                              output_base_path=args.output_base_path),
                      tqdm(video_folder_paths(args.base_path), total=total_folders, desc="Adding noise to video folders")))

    print("\nAll videos have been perturbed successfully!")


if __name__ == '__main__':
    # Parameters for noise generation
    parser = argparse.ArgumentParser(description='Add noise to video frames')

    parser.add_argument('-n', '--num_octaves', type=int,
                        default=5, help='Number of octaves for noise generation')
    parser.add_argument('-x', '--wavelength_x', type=float,
                        default=2.1327105628890695, help='Wavelength in x direction')
    parser.add_argument('-y', '--wavelength_y', type=float,
                        default=2.0, help='Wavelength in y direction')
    parser.add_argument('-t', '--wavelength_t', type=float,
                        default=180.0, help='Wavelength in time')
    parser.add_argument('-c', '--color_period', type=float,
                        default=1.0, help='Color period for noise')
    parser.add_argument('-T', '--T', type=int, default=5,
                        help='Number of frames for Perlin noise generation')
    parser.add_argument('-e', '--epsilon', type=float,
                        default=8.0, help='Epsilon value for noise')
    parser.add_argument('-b', '--base_path', type=str,
                        required=True, help='Base path for video folders')
    parser.add_argument('-o', '--output_base_path', type=str,
                        required=True, help='Output path for perturbed videos')

    args = parser.parse_args()

    main(args)
