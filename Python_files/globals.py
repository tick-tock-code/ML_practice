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
