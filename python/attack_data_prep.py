import os
import cv2
import random
import numpy as np
import argparse
from tqdm import tqdm  # Import tqdm for progress visualization


def center_crop(frame):
    # In the C3D architecture, the input frames are 112x112
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def preprocess_and_save_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame by resizing and cropping
        frame = center_crop(cv2.resize(frame, (171, 128)))  # Resize and crop

        # Apply color normalization
        frame = frame - np.array([[[90.0, 98.0, 102.0]]])

        # Generate a filename for the frame
        frame_filename = f"{frame_num:04d}.jpg"
        # Create the full path to save the frame image
        frame_path = os.path.join(output_folder, frame_filename)

        # Save the preprocessed frame image
        cv2.imwrite(frame_path, frame)
        frame_num += 1
    cap.release()


def randomly_choose_and_preprocess_videos(dataset_root, output_folder, num_videos=500):
    video_paths = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(dataset_root)
        for filename in filenames
        if filename.endswith('.avi')
    ]

    random_video_paths = random.sample(video_paths, num_videos)

    # Use tqdm to visualize the progress
    for video_path in tqdm(random_video_paths, desc="Processing videos", unit="video"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder_with_num = f"{output_folder}_{num_videos}"
        preprocess_and_save_frames(video_path, output_folder_with_num)


def main(args):
    dataset_root = args.dataset_root
    output_folder = args.output_folder
    num_videos = args.num_videos

    print("Starting video preprocessing...")
    randomly_choose_and_preprocess_videos(
        dataset_root, output_folder, num_videos)
    print("Video preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and save video frames")
    parser.add_argument("--dataset_root", "-d", type=str, required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--output_folder", "-o", type=str, required=True,
                        help="Path to the output folder for saving preprocessed frames")
    parser.add_argument("--num_videos", "-n", type=int, default=500,
                        help="Number of videos to randomly choose and process")

    args = parser.parse_args()

    print("Please note that video preprocessing might take some time, but it's a one-time process.")
    main(args)