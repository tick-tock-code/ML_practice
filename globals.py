import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


def save_fullscreen_figure(figure, figure_name_string, path_to_save_folder):
    # Adjust the layout to make sure everything fits
    #figure.tight_layout()

    # Save the figure
    figure.savefig(f"{path_to_save_folder}/{figure_name_string}", bbox_inches='tight')

    plt.show(block=False)
    plt.pause(1)

    # Close the figure
    plt.close(figure)

def create_check_folder(folder_name, directory_path):
    # Create path
    folder_path = os.path.join(directory_path, folder_name)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    return 

def find_script_directory():
    # Get the directory of the current script
    return os.path.dirname(os.path.realpath(__file__))



# Helper to ensure each parameter is iterable (list)
def listify(x):
    """Return a list of scalar values for grid creation.

    - If x is None -> [None]
    - If x is a numpy array / pandas Series -> list(x) (each element becomes a separate value)
    - If x is a list/tuple -> list(x)
    - Otherwise -> [x] (wrap scalar)
    """
    if x is None:
        return [None]
    # flatten numpy arrays and pandas Series into lists of scalars
    if isinstance(x, (np.ndarray, pd.Series)):
        return list(x.tolist())
    if isinstance(x, (list, tuple)):
        # ensure nested arrays are flattened one level
        flattened = []
        for el in x:
            if isinstance(el, (np.ndarray, pd.Series)):
                flattened.extend(list(el.tolist()))
            else:
                flattened.append(el)
        return flattened
    return [x]