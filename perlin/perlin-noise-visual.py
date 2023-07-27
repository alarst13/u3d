import csv
import numpy as np
from mayavi import mlab

# Read the Perlin noise data from the CSV file
x_coords, y_coords, z_coords, noise_values = [], [], [], []
with open('perlin-noise-data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        x_coords.append(int(row[0]))
        y_coords.append(int(row[1]))
        z_coords.append(int(row[2]))
        noise_values.append(float(row[3]))

# Create a 3D scalar field using the Perlin noise data
grid_size = 10
scalar_field = np.zeros((grid_size, grid_size, grid_size))
for x, y, z, noise in zip(x_coords, y_coords, z_coords, noise_values):
    scalar_field[x, y, z] = noise

# Visualize the 3D scalar field using Mayavi
mlab.figure()
mlab.contour3d(scalar_field, colormap='viridis')
mlab.axes()
mlab.show()
