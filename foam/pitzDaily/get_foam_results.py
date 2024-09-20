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
    data_x = []
    data_y = []

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
                data_x.append(list_of_floats[0])
                data_y.append(list_of_floats[1])
    
    return data_x, data_y

def read_pressure_data(filename):
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
        
        # End of data section
        if collect_data and line.startswith(")"):
            collect_data = False
            continue
        
        if collect_data:
            '''
            match = []
            match = re.findall(r'-?\d+\.\d+', line)
            # Extract tuples using regular expression
            if match != []:
                data.append(float(match[0]))
            '''
            data.append(float(line))
        
        # Start of data section
        if line.startswith("("):
            collect_data = True
            continue
        
    return data

def convert_data_to_map(data, model):
    '''
    data: a list of data-ponts, wirten as a list of floats
    model: a list of tupels, discribes the parts of the model in the form (start_x, start_y, end_x, end_y)
    '''
    min_x = min(tup[0] for tup in model)
    min_y = min(tup[1] for tup in model)
    size_x = max(tup[2] for tup in model) - min_x
    size_y = max(tup[3] for tup in model) - min_y

    map = np.array([[0.0] * size_x] * size_y)
    
    index = 0
    for i in range(len(model)):
        partsize_x = model[i][2] - model[i][0]
        partsize_y = model[i][3] - model[i][1]

        new_index = index + partsize_x * partsize_y
        part = (data[index : new_index])
        index = new_index

        # now draw the data on the map
        for y in range(partsize_y):
            for x in range(partsize_x):
                map[model[i][1] + y - min_y][model[i][0] + x - min_x] = part[(partsize_y - 1 - y) * partsize_x + x]

    return map
    
def plot_single_map(array, cmap='viridis', colorbar=True, title="Array Image"):
    """
    Plots a 2D numpy array as an image.
    
    Parameters:
        array (np.ndarray): 2D numpy array of float values.
        cmap (str): Color map to use for the image. Default is 'viridis'.
        colorbar (bool): Whether to show a colorbar. Default is True.
        title (str): Title of the plot. Default is 'Array Image'.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    plt.figure(figsize=(15, 3))
    plt.imshow(array, cmap=cmap, aspect='auto')
    
    if colorbar:
        plt.colorbar(label='Intensity')
    
    plt.title(title)
    plt.axis('off')  # Turn off axis labels
    plt.show()

def plot_stacked_images(arrays, figsize=(18, 3), cmap='bwr', colorbar=True, titles=None):
    """
    Plots multiple 2D numpy arrays as images stacked vertically.
    
    Parameters:
        arrays (list of np.ndarray): List of 2D numpy arrays of float values.
        figsize (tuple): Figure size for each image. Default is (15, 3).
        cmap (str): Color map to use for the images. Default is 'viridis'.
        colorbar (bool): Whether to show a colorbar. Default is True.
        titles (list of str): Titles for each subplot. Default is None.
    """
    num_images = len(arrays)
    
    if titles is not None and len(titles) != num_images:
        raise ValueError("Length of titles list must match the number of images.")
    
    plt.figure(figsize=(figsize[0], figsize[1] * num_images))
    
    for i, array in enumerate(arrays):
        if array.ndim != 2:
            raise ValueError(f"Array at index {i} is not 2D.")
        
        ax = plt.subplot(num_images, 1, i + 1)
        img = ax.imshow(array, cmap=cmap, aspect='auto')
        
        if titles is not None:
            ax.set_title(titles[i])
        
        ax.axis('off')  # Turn off axis labels
        
        if colorbar:
            plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def get_highest_numbered_directory(data_path):
    """
    Returns the directory with the highest numerical name in the specified path.
    
    Parameters:
        data_path (str): The path to the directory containing the subdirectories.
        
    Returns:
        str: The path to the directory with the highest numerical name.
    """
    # List all entries in the data_path
    entries = os.listdir(data_path)
    
    # Filter out directories with numeric names
    numeric_dirs = []
    for entry in entries:
        full_path = os.path.join(data_path, entry)
        if os.path.isdir(full_path) and entry.isdigit():
            numeric_dirs.append(int(entry))
    
    # Check if there are any numeric directories
    if not numeric_dirs:
        return None  # Return None if there are no directories with numeric names
    
    # Find the maximum number
    highest_number = max(numeric_dirs)
    
    # Return the full path of the directory with the highest number
    return str(highest_number)

def scale_model(model, scale):
    # scales every value in every tupel
    return [(i * scale, ii * scale, iii * scale, iv * scale) for (i, ii, iii, iv) in model]

def get_constants():
    # a list of important constants
    nu = 0.08
    return [nu]

def get_normalized_maps(iteration = 0):

    # Path to the OpenFOAM results
    data_path = os.path.dirname(os.path.abspath(__file__))
    if iteration == 0:
        iteration = get_highest_numbered_directory(data_path)
    filename_u = os.path.join(data_path, str(iteration) + '/U')
    filename_p = os.path.join(data_path, str(iteration) + '/p')

    # Read data
    ux_data, uy_data = read_velocity_data(filename_u)
    p_data = read_pressure_data(filename_p)

    # transform the data into 2D arrays
    model = [(0,0,1,1),(1,1,10,2),(1,0,10,1)] # discribes, how the model is build, every block is a tupel
    scale = 10
    model = scale_model(model, scale)
    #print(model)
    ux_map = convert_data_to_map(ux_data, model) 
    uy_map = convert_data_to_map(uy_data, model) 
    p_map = convert_data_to_map(p_data, model) 

    return ux_map, uy_map, p_map

if __name__ == "__main__":
    ux_map, uy_map, p_map = get_normalized_maps()
    u_map = np.sqrt(ux_map**2 + uy_map**2)
    # Plot results
    plot_stacked_images([ux_map, uy_map, u_map, p_map], figsize=(9, 2), cmap='bwr', titles=['velocity X','velocity Y','velocity magnitude','pressure']) # plasma / bwr
