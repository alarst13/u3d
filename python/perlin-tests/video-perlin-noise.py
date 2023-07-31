from perlin import perlin_noise
import cv2
import numpy as np
import imageio

# Function to add Perlin noise to each pixel in the frame
def add_perlin_noise_to_frame(frame, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period):
    frame_height, frame_width = frame.shape[:2]
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the Perlin noise for each pixel in the frame
    for y in range(frame_height):
        for x in range(frame_width):
            noise_value = perlin_noise(
                x, y, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period)
            # gray_frame[y][x] = np.clip(gray_frame[y][x] + noise_value, 0, 255)

            # Apply Perlin noise to each color channel (R, G, B)
            for c in range(frame.shape[2]):
                frame[y][x][c] = np.clip(frame[y][x][c] + noise_value, 0, 255)

    return frame
    # return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

# Function to process the entire video and add Perlin noise to each frame


def process_video(input_video_path, output_video_path, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure the frame dimensions are divisible by 16
    frame_width = (frame_width // 16) * 16
    frame_height = (frame_height // 16) * 16

    # Prepare the output video writer
    writer = imageio.get_writer(output_video_path, fps=frame_rate, quality=3)

    # Process each frame of the video and add Perlin noise
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to ensure dimensions are divisible by 16
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Add Perlin noise to the frame
        noisy_frame = add_perlin_noise_to_frame(
            frame, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t, color_period)

        # Write the noisy frame to the output video
        writer.append_data(noisy_frame)

        frame_num += 1

    # Release video capture and writer objects
    cap.release()
    writer.close()

    print("Perlin noise added to video successfully!")


if __name__ == '__main__':
    # Parameters for noise generation
    num_octaves = 2
    wavelength_x = 16.0
    wavelength_y = 16.0
    wavelength_t = 8.0
    color_period = 2.0

    # Input and output video paths
    input_video_path = '/home/ala22014/u3d/python/perlin-tests/nature.mp4'
    output_video_path = '/home/ala22014/u3d/python/perlin-tests/nature-noise.mp4'

    # Process the video and generate Perlin noise
    process_video(input_video_path, output_video_path, num_octaves,
                  wavelength_x, wavelength_y, wavelength_t, color_period)
