import os
import sys
import subprocess
import platform
import json

VENV_NAME = "karst_env"
PACKAGES = [
    "numpy",
    "pandas",
    "h5py",
    "hdf5plugin",       # Additional HDF5 compression support for H5Web
    "numpy-quaternion", # Correct package name for quaternion
    "plotly",
    "matplotlib",
    "ipykernel>=6",
    "nbformat>=4.2.0",  # Required for Plotly to display figures in Jupyter notebooks
    "ipywidgets",
    #"jupyter",          # Full Jupyter suite for notebook support
    "pyvista",
    "trame",
    "trame-client",
    "trame-vtk",
    "trame-vuetify",
]

def get_venv_paths(venv_name):
    """Gets the platform-specific paths for the virtual environment."""
    if platform.system() == "Windows":
        python_executable = os.path.join(venv_name, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_name, "Scripts", "pip.exe")
    else: # macOS and Linux
        python_executable = os.path.join(venv_name, "bin", "python")
        pip_executable = os.path.join(venv_name, "bin", "pip")
    return python_executable, pip_executable

def update_vscode_settings(venv_name):
    """
    Creates/updates .vscode/settings.json to default to the venv interpreter.
    """
    vscode_dir = ".vscode"
    settings_file = os.path.join(vscode_dir, "settings.json")
    
    # Get the relative path to the python executable
    if platform.system() == "Windows":
        python_path = os.path.join(venv_name, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_name, "bin", "python")

    # Normalize path for JSON (use forward slashes)
    python_path = python_path.replace('\\', '/')

    settings = {}
    
    # Create .vscode directory if it doesn't exist
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
    
    # Load existing settings if the file exists
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                print(f"\nWarning: Could not parse existing '{settings_file}'. It will be overwritten.")
                settings = {}

    # Update the python path
    settings['python.defaultInterpreterPath'] = python_path
    
    # Write the settings back
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
        
    print("\n======================================================")
    print("VS Code workspace configured to use the new environment.")
    print("======================================================")


def main():
    """
    Creates a virtual environment and installs necessary packages.
    """
    print("======================================================")
    print(f"Creating virtual environment '{VENV_NAME}'...")
    print("======================================================")

    try:
        # Create the virtual environment
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
    except subprocess.CalledProcessError:
        print("\nERROR: Failed to create the virtual environment.")
        print(f"Please ensure Python {sys.version_info.major}.{sys.version_info.minor} is correctly installed.")
        return
    except FileNotFoundError:
        print("\nERROR: `python` command not found.")
        print("Please ensure Python is installed and in your system's PATH.")
        return

    python_executable, pip_executable = get_venv_paths(VENV_NAME)

    # Check if the environment was created successfully
    if not os.path.exists(python_executable):
        print(f"\nERROR: Could not find Python executable in '{VENV_NAME}'.")
        print("Virtual environment creation failed.")
        return

    print("\n======================================================")
    print("Activating environment and installing packages...")
    print("======================================================")

    try:
        # Install packages using the venv's pip
        # Using the direct path to pip is more reliable than activating
        command = [pip_executable, "install"] + PACKAGES
        subprocess.run(command, check=True)

    except subprocess.CalledProcessError:
        print("\nERROR: Failed to install packages.")
        print("Please check your internet connection and try again.")
        return

    # Configure VS Code to use this environment
    update_vscode_settings(VENV_NAME)

    print("\n======================================================")
    print("Done!")
    print("======================================================")
    print(f"\nYour virtual environment '{VENV_NAME}' is ready.")
    print("\nVS Code has been configured to use this new environment automatically.")
    print("Please reload the VS Code window if prompted.")
    print("\nNext, open a notebook file (e.g., '1_DATA_LOAD.ipynb') and check the kernel in the top-right corner.")
    print(f"It should automatically select the interpreter from the '{VENV_NAME}' environment.")
    
    print("\nIf the correct kernel is NOT selected automatically, please follow these steps:")
    print("1. Click the 'Select Kernel' button in the top-right corner of the notebook.")
    print("2. A list of Python environments will appear.")
    print(f"3. Select the Python interpreter that is inside the '{VENV_NAME}' folder.")

    print("\nIf you want to activate the environment manually in a terminal, run:")
    if platform.system() == "Windows":
        print(f"  .\\{VENV_NAME}\\Scripts\\activate")
    else:
        print(f"  source {VENV_NAME}/bin/activate")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
