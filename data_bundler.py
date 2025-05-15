import os
import glob
import random
import numpy as np
from tqdm import tqdm

from audio_converter import AudioConverter

class DataBundler:
    def __init__(self, root_path="datasets/DCASE2025T2"):
        print("\nData Loader Here!")
        self.root_path = root_path


    def load_dataset(self, inclusion_string=None, percentage=1.0, shuffle=True, as_input=True, audio_type="all"):
        audio_files = []
        for directory_path, _, filenames in os.walk(self.root_path):
            if inclusion_string is None or inclusion_string in directory_path:
                files = glob.glob(os.path.join(directory_path, '*.wav'))
                
                if audio_type == "normal":
                    filtered_files = [file for file in files if "normal" in os.path.basename(file)]
                elif audio_type == "anomaly":
                    filtered_files = [file for file in files if "anomaly" in os.path.basename(file)]
                else:
                    filtered_files = files
                
                audio_files.extend(filtered_files)

        if shuffle:
            random.shuffle(audio_files)  # Shuffle the list if `shuffle` is True
            #print(f"Shuffled {len(audio_files)} files")

        number_of_files = len(audio_files) #get number of files in dataset
        files_to_process = int(percentage * number_of_files) #get number of files to process
        audio_files = audio_files[:files_to_process]

        audio_converter = AudioConverter()
        all_input_features = []
        all_clip_lengths = []
        filenames = []

        for i, file_path in enumerate(tqdm(audio_files, desc="Processing audio files", unit="file"), 1):
            #print(f"({i}/{files_to_process}) Processing '{os.path.basename(file_path)}'")
            if as_input:
                clip_lengths, input_features = audio_converter.wav_to_input(file_path)
            else:
                clip_lengths, input_features = audio_converter.wav_to_mel(file_path)

            all_input_features.append(input_features)
            all_clip_lengths.append(clip_lengths)
            filenames.append(os.path.basename(file_path))

        dataset = np.vstack(all_input_features) #stack all input features into a single array
        all_clip_lengths = np.array(all_clip_lengths)
        print(f"\nDone loading!")
        print(f"Length of dataset: {len(dataset)}\n")

        return all_clip_lengths, dataset, filenames

    
data_bundler = DataBundler()
clip_lengths, dataset, filenames = data_bundler.load_dataset('train', 0.01, True, True)
print(clip_lengths)