from perlin import perlin_noise
import cv2
import numpy as np


# TODO: Optimize the Perlin noise generation to accept vectorized inputs
# Generate Perlin noise for the first T frames of the video
def generate_noise(T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon):
    # Create a 3D array to store the noise values for each (x, y, t) in the frame
    noise_values_3D = np.zeros((T, frame_height, frame_width))

    # Calculate the Perlin noise for each (x, y) in the frame
    for frame_num in range(T):
        for y in range(frame_height):
            for x in range(frame_width):
                noise_value = perlin_noise(
                    x, y, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon)
                noise_values_3D[frame_num][y][x] = noise_value
    return noise_values_3D


# Add Perlin noise to a frame using precomputed noise values
def add_perlin_noise_to_frame(frame, noise_values):
    noisy_frame = frame + noise_values[:, :, np.newaxis]
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)


# Process the entire video and add Perlin noise to each frame
# T: Number of frames for Perlin noise generation (adjust as needed)
def perturb_video(num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, T, epsilon, input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_path, fourcc,
                             frame_rate, (frame_width, frame_height))

    # Process each frame of the video and add Perlin noise
    frame_num = 0
    # Generate the Perlin noise for the first T frames
    noise_values_3D = generate_noise(
        T, frame_height, frame_width, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period, epsilon)
    while True:
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

    return output_video_path


if __name__ == '__main__':
    # Parameters for noise generation
    num_octaves = 5
    wavelength_x = 2.0
    wavelength_y = 180.0
    wavelength_t = 49.35137797
    color_period = 1.0
    T = 5  # Number of frames for Perlin noise generation (adjust as needed)
    epsilon = 8.0  # Maximum perturbation allowed (epsilon) for the U3D attack
    input_video_path = '/mnt/data/UCF-101/Biking/v_Biking_g03_c04.avi'
    output_video_path = 'python/noise_perturbation/perturbed_video.avi'

    perturb_video(num_octaves, wavelength_x, wavelength_y,
                  wavelength_t, color_period, T, epsilon, input_video_path, output_video_path)
