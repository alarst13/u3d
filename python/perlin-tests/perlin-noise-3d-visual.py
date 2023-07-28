import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the Perlin noise data from the CSV file
x_coords, y_coords, z_coords, noise_values = [], [], [], []
with open('perlin-noise-3d-data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        x_coords.append(int(row[0]))
        y_coords.append(int(row[1]))
        z_coords.append(int(row[2]))
        noise_values.append(float(row[3]))

# Create a 3D scatter plot of the Perlin noise
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c=noise_values, cmap='gray')

# Add labels and title
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
ax.set_zlabel('Z-coordinate')
ax.set_title('3D Perlin Noise Visualization')

# Save the 3D Perlin noise visualization as a PNG image
plt.savefig('perlin-noise-visual.png')

# Display the 3D Perlin noise visualization
plt.show()
