import os
import glob
import numpy as np

from audio_converter import AudioConverter

class DataLoader:
    def __init__(self):
        print("Data Loader Here!")
        self.audio_converter = AudioConverter()

    def load_dataset(self, dataset_path):
        audio_files = glob.glob(os.path.join(dataset_path, '*.wav')) #load all audio files in list

        all_input_features = []

        number_of_files = len(audio_files)
        i = 1

        for file_path in audio_files:
            print(f"({i}/{number_of_files}) Processing '{os.path.basename(file_path)}'")
            input_features = self.audio_converter.wav_to_input(file_path) #convert from wav to input features
            all_input_features.append(input_features)
            i += 1

        dataset = np.vstack(all_input_features) #stack all input features into a single array
        return dataset
    
# data_loader = DataLoader()
# dataset = data_loader.load_dataset("datasets/DCASE2025T2/ToyCar/Development/ToyCar/train")
# print(len(dataset))