import os
import glob
import random
import numpy as np

from audio_converter import AudioConverter

class DataBundler:
    def __init__(self):
        print("\nData Loader Here!")
        self.audio_converter = AudioConverter()

    def load_dataset(self, dataset_path, percentage=1.0, shuffle=True):
        audio_files = glob.glob(os.path.join(dataset_path, '*.wav')) #load all audio files in list

        if shuffle:
            random.shuffle(audio_files)  # Shuffle the list if `shuffle` is True

        number_of_files = len(audio_files) #get number of files in dataset
        files_to_process = int(percentage * number_of_files) #get number of files to process
        audio_files = audio_files[:files_to_process]

        all_input_features = []
        filenames = []
        for i, file_path in enumerate(audio_files, 1):
            print(f"({i}/{files_to_process}) Processing '{os.path.basename(file_path)}'")
            input_features = self.audio_converter.wav_to_input(file_path)

            all_input_features.append(input_features)
            filenames.append(os.path.basename(file_path))

        dataset = np.vstack(all_input_features) #stack all input features into a single array
        print(f"\nDone loading!")
        print(f"Length of dataset: {len(dataset)}\n")

        return dataset, filenames
    
# data_loader = DataLoader()
# dataset = data_loader.load_dataset("datasets/DCASE2025T2/ToyCar/Development/ToyCar/train", 0.1)