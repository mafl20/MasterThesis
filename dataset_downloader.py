import os #library for interacting with the operating system
import requests #library for making http requests
import yaml #library for parsing and writing YAML files

from zipfile import ZipFile #library for working with ZIP files
from tqdm import tqdm #library for creating progress bars

class DatasetDownloader:
    def __init__(self, base_directory="datasets"): #constructor
        self.base_directory = base_directory #directory where the datasets will be stored
        os.makedirs(self.base_directory, exist_ok=True) #create the base directory if it doesn't exist

    #> Downloads a file from the given URL and saves it to the target directory
    def download_file(self, url, target_directory, filename=None):
        os.makedirs(target_directory, exist_ok=True) #ensure the target directory exists
        
        #> if no filename is provided, extract it from the URL
        if not filename:
            filename = os.path.basename(url)

        file_path = os.path.join(target_directory, filename) #full path for the downloaded file

        #> if the file already exists, notify the user and return the file path
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            return file_path 
        
        response = requests.get(url, stream=True) #make an HTTP GET request to download the file
        total_size = int(response.headers.get('content-length', 0)) #get the total file size

        #> open the file for writing (in binary mode, wb) and show a progress bar
        with open(file_path, 'wb') as file, tqdm(
            desc=f"Downloading {filename}", #label for the progress bar
            total=total_size, #total size of the file, for the progress bar
            unit='B', #unit of measurement for the progress bar (B for bytes)
            unit_scale=True, #scale hte unit (e.g., KB, MB, GB)
            unit_divisor=1024, #divisor for scaling the unit (1024 for binary)
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024): #download in chunks of 1024 bytes
                file.write(chunk) #write each chunk to the file
                bar.update(len(chunk)) #update the progress bar

        return file_path #return the path of the downloaded file

    #> Extracts a ZIP file to the specified directory
    def extract_zip(self, zip_path, extract_to=None):
        #> if not extraction directory is specified, use the zip file name (without extension) as the directory
        if extract_to is None:
            extract_to = os.path.splitext(zip_path)[0]

        #> open the ZIP file and extract all its contents to the target directory
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        return extract_to #return the directory where the files were extracted to

    #> Downloads a ZIP file from the given URL and extracts it to the target directory
    def download_and_extract(self, url, target_directory):
        zip_path = self.download_file(url, target_directory) #download the ZIP file
        extract_path = self.extract_zip(zip_path, target_directory) #extract the contents of the ZIP file

        os.remove(zip_path)  # Delete the zip file after extraction

        print(f"Extracted to: {extract_path}") #notify the user where the files were extracted
        return extract_path #return extraction path

    #> Processes the YAML file and downloads the datasets into the appropriate directories
    def process_yaml(self, yaml_file):
        #> open the YAML file for reading (r) and store its contents in the variable 'data'
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        #> iterate through the YAML data hierarchy
        for challenge_name, machines in data.items(): #iterate through challenges
            for machine_type, datasets in machines.items(): #iterate through machine types
                for dataset_type, urls in datasets.items(): #iterate through dataset types
                    for url in urls: #iterate through URLs and build the target directory structure
                        target_directory = os.path.join(self.base_directory, challenge_name, machine_type, dataset_type)
                        self.download_and_extract(url, target_directory) #download and extract the dataset
    
    def download_datasets(self, yaml_file_path):
        while True:
            answer = input("Do you want to download the datasets? (yes/no): ").strip().lower()
            if answer in ("yes", "no"):
                break
            print("Please enter 'yes' or 'no'.")

        if answer == "yes":
            try:
                self.process_yaml(yaml_file_path)
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Exiting the program. No datasets were downloaded.")
            return