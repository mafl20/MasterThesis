import yaml
import librosa
import numpy as np

class AudioConverter:
    def __init__(self):
        print("Audio Converter Here!")

        self.hyper_parameters = self.load_hyper_parameters()
        self.acoustic_features = self.hyper_parameters['acoustic_features']

        #> setup general acoustic features
        self.frame_size_seconds = self.acoustic_features['frame_size_seconds']
        self.number_of_mels = self.acoustic_features['number_of_mels']


    def load_hyper_parameters(self):
        with open("hyper_parameters.yaml", 'r') as file:
            return yaml.safe_load(file)


    def wav_to_mel(self, wav_path):
        amplitude, sample_rate = librosa.load(wav_path, sr=None, mono=True) #load audio file

        #> setup audio clip specific acoustic features
        frame_size_samples = int(self.frame_size_seconds * sample_rate)
        hop_size_samples = frame_size_samples // 2
        #window_size_samples = frame_size_samples
        number_of_mels = self.number_of_mels

        #> print out some information
        # print(f"frame_size_samples: {frame_size_samples}")
        # print(f"hop_size_samples: {hop_size_samples}")
        # print(f"number_of_mels: {number_of_mels}")

        #> conversion to mel spectrogram
        mel_audio = librosa.feature.melspectrogram(
            y           = amplitude,
            sr          = sample_rate,
            n_fft       = frame_size_samples,
            hop_length  = hop_size_samples,
            n_mels      = number_of_mels
        )
        mel_audio = librosa.power_to_db(mel_audio, ref=np.max) # convert to decibels

        return mel_audio
    

    def mel_to_input(self, mel_audio):
        number_of_frames = mel_audio.shape[1]
        number_of_frames_to_concatenate = self.acoustic_features['number_of_frames_to_concatenate']

        input_features = []
        for i in range(number_of_frames - number_of_frames_to_concatenate + 1):
            # print(i)
            concatenated_segment = mel_audio[:, i:i + number_of_frames_to_concatenate].flatten()
            input_features.append(concatenated_segment)
        
        input_features = np.array(input_features)
        # print(f"input_features.shape: {input_features.shape}")
        return input_features
    

    def wav_to_input(self, wav_path):
        mel_audio = self.wav_to_mel(wav_path)
        return self.mel_to_input(mel_audio)
        


# audio_converter = AudioConverter()
# mel_audio = audio_converter.wav_to_mel("datasets/DCASE2025T2/ToyCar/Development/ToyCar/train/section_00_source_train_normal_0001_car_B1_spd_31V_mic_1.wav")
# input_features = audio_converter.mel_to_input(mel_audio)