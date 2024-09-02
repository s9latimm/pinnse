import numpy as np
import matplotlib.pyplot as plt
import os

def read_velocity_data(filename):
    """Read velocity data from OpenFOAM file."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    reading_data = False

    for line in lines:
        line = line.strip()
        if line == "vectorField":
            reading_data = True
            continue
        if reading_data:
            if line == ";":
                break
            data.append(line)

    # Convert data to numpy array
    data = np.array([list(map(float, d.split())) for d in data])
    print(f"Data shape: {data.shape}")  # Debug print to check the shape of the data
    return data

def plot_velocity(u_data, nx, ny):
    """Plot the velocity field."""
    # Ensure u_data has the correct shape
    if len(u_data.shape) != 2 or u_data.shape[1] != 2:
        raise ValueError("u_data must be a 2D array with 2 columns for velocity components.")
    
    # Check the total number of elements
    num_elements = u_data.shape[0]
    print(f"Number of data points: {num_elements}")  # Debug print to check the number of data points

    if num_elements != nx * ny:
        raise ValueError(f"Number of data points ({num_elements}) does not match grid size ({nx * ny}).")

    u_x = u_data[:, 0].reshape((ny, nx))
    u_y = u_data[:, 1].reshape((ny, nx))

    # Create a grid
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(14, 6))

    # Plot Ux
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u_x, cmap='viridis')
    plt.colorbar(label='Ux')
    plt.title('Velocity Ux')

    # Plot Uy
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u_y, cmap='viridis')
    plt.colorbar(label='Uy')
    plt.title('Velocity Uy')

    # Plot magnitude of velocity
    magnitude = np.sqrt(u_x**2 + u_y**2)
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, magnitude, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Velocity Magnitude')

    plt.tight_layout()
    plt.show()

# Path to the OpenFOAM results
data_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(data_path, '99/U')

# Read data
u_data = read_velocity_data(filename)

# Define grid size (example, adjust according to your mesh)
nx, ny = 50, 50  # Adjust these dimensions to match your grid

# Plot results
plot_velocity(u_data, nx, ny)
