from perlin import perlin_noise
import cv2
import numpy as np

# Generate Perlin noise for the first T frames of the video
def generate_noise(T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period):
    # Create a 3D array to store the noise values for each (x, y, t) in the frame
    noise_values_3D = [[[0 for _ in range(frame_width)] for _ in range(
        frame_height)] for _ in range(T)]

    # Calculate the Perlin noise for each (x, y) in the frame
    for frame_num in range(T):
        for y in range(frame_height):
            for x in range(frame_width):
                noise_value = perlin_noise(
                    x, y, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period)
                noise_values_3D[frame_num][y][x] = noise_value
    return noise_values_3D


# Add Perlin noise to each pixel in the frame
def add_perlin_noise_to_frame(frame, frame_noise_values):
    frame_height, frame_width = frame.shape[:2]

    # Calculate the Perlin noise for each pixel in the frame
    for y in range(frame_height):
        for x in range(frame_width):
            noise_value = frame_noise_values[y][x]

            # Apply Perlin noise to each color channel (R, G, B)
            for c in range(frame.shape[2]):
                frame[y][x][c] = np.clip(frame[y][x][c] + noise_value, 0, 255)

    return frame


# Process the entire video and add Perlin noise to each frame
def process_video(num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, T, input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_path, fourcc,
                             frame_rate, (frame_width, frame_height))

    # Process each frame of the video and add Perlin noise
    frame_num = 0
    # Generate the Perlin noise for the first T frames
    noise_values_3D = generate_noise(
        T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period)
    while frame_num < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Add Perlin noise to the current frame using the precomputed noise values
        t = frame_num % T  # Loop through the noise values
        noisy_frame = add_perlin_noise_to_frame(frame, noise_values_3D[t])

        # Write the noisy frame to the output video using cv2
        writer.write(noisy_frame)

        frame_num += 1

    # Release video capture and writer objects
    cap.release()
    writer.release()

    print("Perlin noise added to video successfully!")


# T: Number of frames for Perlin noise generation (adjust as needed)
def perturb_video_perlin(num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, T, input_video_path, output_video_path):
    # Process the video and generate Perlin noise
    process_video(num_octaves, wavelength_x, wavelength_y, wavelength_t,
                  color_period, T, input_video_path, output_video_path)
