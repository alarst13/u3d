import os
import cv2
import random
import numpy as np
import argparse
from tqdm import tqdm


def center_crop(frame):
    # In the C3D architecture, the input frames are 112x112
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def preprocess_and_save_frames(video_path, class_name, output_folder):
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    # Create a directory for this class
    class_output_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_output_folder, exist_ok=True)

    # Create a directory for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(class_output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame by resizing and cropping
        frame = center_crop(cv2.resize(frame, (171, 128)))  # Resize and crop

        # Apply color normalization
        frame = frame - np.array([[[90.0, 98.0, 102.0]]])

        # Generate a filename for the frame with the video number and class name prefixes
        frame_filename = f"{frame_num:04d}.jpg"
        # Create the full path to save the frame image
        frame_path = os.path.join(video_output_folder, frame_filename)

        # Save the preprocessed frame image
        cv2.imwrite(frame_path, frame)
        frame_num += 1

    cap.release()


def randomly_choose_and_preprocess_videos(dataset_root, output_folder, num_videos=500):
    video_paths = [
        (os.path.basename(dirpath), os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(dataset_root)
        for filename in filenames
        if filename.endswith('.avi')
    ]

    # Organize video paths by class
    class_to_videos = {}
    for c, video_path in video_paths:
        class_to_videos.setdefault(c, []).append(video_path)

    # Calculate the number of videos to select from each class
    total_videos = sum(len(videos) for videos in class_to_videos.values())
    class_to_num_videos = {c: int(len(videos)/total_videos * num_videos)
                           for c, videos in class_to_videos.items()}

    random_video_paths = []
    for c, num in class_to_num_videos.items():
        random_video_paths.extend((c, video)
                                  for video in random.sample(class_to_videos[c], num))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for class_name, video_path in tqdm(random_video_paths, desc="Processing videos", unit="video"):
        preprocess_and_save_frames(video_path, class_name, output_folder)


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
    parser.add_argument('--dataset_root', '-d', type=str, required=True,
                        help='Directory path with video splits (train, val, test)')
    parser.add_argument('--output_folder', '-o', type=str, required=True,
                        help='Path to the output folder for saving preprocessed frames')
    parser.add_argument('--num_videos', '-n', type=int, default=500,
                        help='Number of videos to randomly choose and process')

    args = parser.parse_args()

    print("Please note that video preprocessing might take some time, but it's a one-time process.")
    main(args)
