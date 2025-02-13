import os
import subprocess
import sys
import urllib.request
import zipfile


venv_dir = "venv"
datasets = {
    "SMRCs": [
        'https://github.com/SynapticRAG/acl25/releases/download/dataset/SMRCs.zip'
    ],
    "PerLTQA": [
        'https://github.com/SynapticRAG/acl25/releases/download/dataset/PerLTQA.zip'
    ]
}


def create_virtual_environment():
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    print(f"Virtual environment created in {venv_dir}")


def install_requirements():
    print("Installing packages from requirements.txt...")
    pip_executable = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip.exe")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
    subprocess.check_call([pip_executable, "install", "faiss-cpu"])
    print("Packages installed")


def download_and_extract(url, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    filename = url.split("/")[-1]
    file_path = os.path.join(extract_to, filename)

    print(f"Downloading {filename} to {extract_to}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"{filename} downloaded")

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{filename} extracted to {extract_to}")

    os.remove(file_path)
    print(f"{filename} removed")


def download_datasets():
    for dataset, urls in datasets.items():
        if dataset == "SMRCs":
            extract_to = "./datasets/SMRCs/dataset/prepared"
        elif dataset == "PerLTQA":
            extract_to = "./datasets/PerLTQA/dataset/prepared"
        else:
            continue

        for url in urls:
            download_and_extract(url, extract_to)


def main():
    create_virtual_environment()
    install_requirements()
    download_datasets()
    print("Setup completed")


if __name__ == "__main__":
    main()
