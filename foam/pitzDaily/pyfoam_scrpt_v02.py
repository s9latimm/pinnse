import os
import numpy as np
import matplotlib.pyplot as plt

def load_U_file(file_path):
    # Load OpenFOAM U file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the internal field section
    internal_field_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('internalField'):
            internal_field_start = i + 1
            break

    if internal_field_start is None:
        raise ValueError("Could not find 'internalField' in the file.")

    # Find the number of vectors and the start of the data
    num_vectors_line = lines[internal_field_start].strip()
    num_vectors = int(num_vectors_line)
    
    # Extract the field data
    field_data = lines[internal_field_start + 1 : internal_field_start + 1 + num_vectors]
    
    # Parse the field data
    U = np.array([list(map(float, line.strip('() \n').split())) for line in field_data])
    
    return U

def plot_velocity_field(U, nx, ny):
    U_x = U[:, 0].reshape((ny, nx))  # Assumes (x, y) arrangement
    U_y = U[:, 1].reshape((ny, nx))

    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))  # Normalized coordinates

    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U_x, U_y, scale=5)
    plt.title("Velocity Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Define the path to your OpenFOAM case folder
case_path = os.path.dirname(os.path.abspath(__file__))  # Replace with your path

# Define mesh dimensions
nx, ny = 100, 100  # Replace with your mesh dimensions

# Path to the U file in the latest time directory
time_dirs = [d for d in os.listdir(case_path) if d.replace('.', '').isdigit()]
latest_time = sorted(time_dirs, key=float)[-1]
U_file_path = os.path.join(case_path, latest_time, 'U')

# Load U file and plot
U = load_U_file(U_file_path)
plot_velocity_field(U, nx, ny)
