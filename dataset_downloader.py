import os #library for interacting with the operating system
import requests #library for making http requests
import yaml #library for parsing and writing YAML files

from zipfile import ZipFile #library for working with ZIP files
from tqdm import tqdm #library for creating progress bars

class DatasetDownloader:
    def __init__(self, base_directory="datasets"):
        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)

    def download_file(self, url, target_directory, filename=None):
        os.makedirs(target_directory, exist_ok=True)
        if not filename:
            filename = os.path.basename(url)
        file_path = os.path.join(target_directory, filename)

        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            return file_path
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, 'wb') as file, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))

        return file_path

    def extract_zip(self, zip_path, extract_to=None):
        if extract_to is None:
            extract_to = os.path.splitext(zip_path)[0]

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to

    def download_and_extract(self, url, target_directory):
        zip_path = self.download_file(url, target_directory)
        extract_path = self.extract_zip(zip_path, target_directory)

        os.remove(zip_path)  # Delete the zip file after extraction

        print(f"Extracted to: {extract_path}")
        return extract_path

    def process_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        for challenge_name, machines in data.items():
            for machine_type, datasets in machines.items():
                for dataset_type, urls in datasets.items():
                    for url in urls:
                        target_directory = os.path.join(self.base_directory, challenge_name, machine_type, dataset_type)
                        self.download_and_extract(url, target_directory)


if __name__ == "__main__":
    dataset_downloader = DatasetDownloader()

    yaml_file_path = "datasets/path_files/download_paths_2025.yaml"
    dataset_downloader.process_yaml(yaml_file_path)