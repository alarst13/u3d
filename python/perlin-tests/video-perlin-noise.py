from perlin import perlin_noise
import cv2
import numpy as np
import imageio

# Function to process the entire video and generate Perlin noise
def process_video(input_video_path, output_video_path, num_octaves, wavelength_x, wavelength_y, wavelength_t):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    writer = imageio.get_writer(output_video_path, fps=frame_rate, quality=3)

    # Process each frame of the video and generate Perlin noise
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the Perlin noise for each pixel in the frame
        for y in range(frame_height):
            for x in range(frame_width):
                noise_value = perlin_noise(
                    x, y, frame_num, num_octaves, wavelength_x, wavelength_y, wavelength_t)
                gray_frame[y][x] = int(noise_value * 255)

        # Write the frame to the output video
        writer.append_data(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))

    # Release video capture and writer objects
    cap.release()
    writer.close()

    print("Pelin noise video generated successfully!")


if __name__ == '__main__':
    # Parameters for noise generation
    num_octaves = 2
    wavelength_x = 16.0
    wavelength_y = 16.0
    wavelength_t = 8.0

    # Input and output video paths
    input_video_path = '/home/ala22014/u3d/python/perlin-tests/nature.mp4'
    output_video_path = '/home/ala22014/u3d/python/perlin-tests/nature-noise.mp4'

    # Process the video and generate Perlin noise
    process_video(input_video_path, output_video_path, num_octaves,
                  wavelength_x, wavelength_y, wavelength_t)