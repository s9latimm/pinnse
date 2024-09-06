import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math

def read_velocity_data(filename):
    """Read velocity data from OpenFOAM file."""
    """
    Reads the velocity tuples from an OpenFOAM file and returns them as a list of tuples.
    
    Parameters:
        filename (str): Path to the OpenFOAM file.
        
    Returns:
        List[Tuple[float, float, float]]: List of velocity tuples.
    """
    data = []

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Flag to start collecting data
    collect_data = False

    for line in lines:
        line = line.strip()
        
        # Start of data section
        if line.startswith("internalField"):
            collect_data = True
            continue
        
        # End of data section
        if collect_data and line.startswith(";"):
            collect_data = False
            continue
        
        if collect_data:
            # Extract tuples using regular expression
            matches = re.findall(r'\(([^)]+)\)', line)
            for match in matches:
                # Convert list string to list of floats
                list_of_floats = list(map(float, match.split()))
                data.append(list_of_floats[:2])
    
    return data

def convert_data_to_model(data, nx, ny):
    parts = []
    points_per_part = nx * ny

    for i in range(5):
        parts.append(data[(points_per_part * i) : (points_per_part * (i+1))])
    empty_field = [[0.0, 0.0]] * points_per_part
    row1 = [empty_field, parts[1], parts[3]]
    row2 = [parts[0], parts[2], parts[4]]

    converted_data = []
    for i in range(ny):
        for part in row1:
            converted_data += part[(nx * i) : (nx * (i+1))]
    for i in range(ny):
        for part in row2:
            converted_data += part[(nx * i) : (nx * (i+1))]
    
    return converted_data
    


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

def plot_velocity_magnitude(u_data, nx, ny):
    """Plot the magnitude of the velocity field."""
    
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

    # Calculate the magnitude of velocity
    magnitude = np.sqrt(u_x**2 + u_y**2)

    # Plot the magnitude
    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, magnitude, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Velocity Magnitude')

    plt.tight_layout()
    plt.show()

# Path to the OpenFOAM results
data_path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(data_path, '186/U')

# Read data
u_data = read_velocity_data(filename)

# Define grid size (example, adjust according to your mesh)
nx, ny = 100, 100  # Adjust these dimensions to match your grid

conv_data = np.array(convert_data_to_model(u_data, nx, ny))

# Plot results
plot_velocity_magnitude(conv_data, 3*nx, 2*ny)
