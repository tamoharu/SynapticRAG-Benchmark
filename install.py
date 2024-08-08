import os
import subprocess
import sys
import urllib.request
import zipfile

venv_dir = "venv"

def create_virtual_environment():
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    print(f"Virtual environment created in {venv_dir}")

def install_requirements():
    print("Installing packages from requirements.txt...")
    pip_executable = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip.exe")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
    print("Packages installed")

def download_dataset():
    dataset_url = "https://github.com/username/repo/releases/download/v1.0/dataset.zip"
    dataset_zip_path = "dataset.zip"
    dataset_extract_dir = "dataset"

    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_zip_path)
    print("Dataset downloaded")

    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_extract_dir)
    print(f"Dataset extracted to {dataset_extract_dir}")

    os.remove(dataset_zip_path)
    print("ZIP file removed")

def main():
    create_virtual_environment()
    install_requirements()
    download_dataset()
    print("Setup completed")

if __name__ == "__main__":
    main()
