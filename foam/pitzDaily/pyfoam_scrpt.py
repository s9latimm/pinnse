import os
import numpy as np
import matplotlib.pyplot as plt
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.BasicRunner import BasicRunner

# Step 1: Define paths and run simulation
case_path = os.path.dirname(os.path.abspath(__file__))  # Path to the current directory (case folder)
solver = "simpleFoam"  # Update with the solver you're using if different

# Run the simulation using BasicRunner
runner = BasicRunner(argv=[solver, '-case', case_path], silent=True, logname="log.simpleFoam")
runner.start()

# Step 2: Get the latest time directory manually
time_dirs = [d for d in os.listdir(case_path) if d.replace('.', '').isdigit()]
latest_time = sorted(time_dirs, key=float)[-1]  # Get the last time step as a string

# Step 3: Load the velocity field U
U_file = os.path.join(case_path, latest_time, "U")
U_data = ParsedParameterFile(U_file)
U_values = np.array(U_data["internalField"].getInternalField())

# Step 4: Reshape U data for plotting
# Assuming 2D simulation with a structured mesh
# Update nx, ny with the actual dimensions of your mesh
nx, ny = 100, 100  # Update with actual mesh dimensions
U_x = U_values[:, 0].reshape((nx, ny))
U_y = U_values[:, 1].reshape((nx, ny))

# Step 5: Plot the velocity field
X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))  # Assuming normalized coordinates

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U_x, U_y)
plt.title(f"Velocity Field at Time = {latest_time}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
