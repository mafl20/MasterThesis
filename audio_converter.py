import yaml
import librosa
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader

class AudioConverter:
    def __init__(self):
        print("Audio Converter Here!")

        self.hyper_parameters = self.load_hyper_parameters()

        #> setup general acoustic features
        self.acoustic_features = self.hyper_parameters['acoustic_features']
        self.frame_size_seconds = self.acoustic_features['frame_size_seconds']
        self.number_of_mels = self.acoustic_features['number_of_mels']

        self.training_parameters = self.hyper_parameters['training_parameters']
        self.batch_size = self.training_parameters['batch_size']
        self.shuffle = self.training_parameters['shuffle']


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
        mel_audio = librosa.power_to_db(mel_audio, ref=np.max) #convert to decibels

        return mel_audio
    

    def mel_to_input(self, mel_audio):
        number_of_frames = mel_audio.shape[1]

        trimmed_frames = number_of_frames - (number_of_frames % 5) #376 - (376 % 5) => 376 - 1 = 375
        # print(f"trimmed_frames: {trimmed_frames}")
        mel_audio = mel_audio[:, :trimmed_frames] #trim the mel audio to be divisible by 5

        concatenated_mel_audio = mel_audio.reshape(mel_audio.shape[0], mel_audio.shape[1] // 5, 5) #reshape to (number_of_mels, number_of_frames // 5, 5) which is (128, 75, 5)
        concatenated_mel_audio = concatenated_mel_audio.transpose(1, 0, 2) #transpose to (number_of_frames // 5, number_of_mels, 5) which is (75, 128, 5)
        concatenated_mel_audio = concatenated_mel_audio.reshape(concatenated_mel_audio.shape[0], -1) #reshape to (number_of_frames // 5, number_of_mels * 5) which is (75, 640)

        return concatenated_mel_audio.shape[0], concatenated_mel_audio
    

    def wav_to_input(self, wav_path):
        mel_audio = self.wav_to_mel(wav_path)
        return self.mel_to_input(mel_audio)
        
    
    def output_to_mel(self, output, number_of_mels=128, number_of_frames_to_concatenate=5):
        output_reshaped = output.reshape(output.shape[0], number_of_mels, number_of_frames_to_concatenate)
        output_reshaped = output_reshaped.transpose(1, 0, 2)
        output_reshaped = output_reshaped.reshape(output_reshaped.shape[0], -1)

        return output_reshaped

    def mel_to_wav(self, mel_audio):
        linear_audio = librosa.db_to_power(mel_audio, ref=1.0) #convert back to linear scale
        
        mel_to_linear = librosa.feature.inverse.mel_to_audio(
            linear_audio,
            sr=16000,
            n_fft=1024,
            hop_length=512
        )

        return mel_to_linear
    
    def output_to_wav(self, output):
        mel_audio = self.output_to_mel(output)
        return self.mel_to_wav(mel_audio)


audio_converter = AudioConverter()
clip_length, audio = audio_converter.wav_to_input("datasets/DCASE2025T2/Development/ToyCar/train/section_00_source_train_normal_0002_car_B1_spd_31V_mic_1.wav")
# print(clip_length)


# output_path = f"reconstructions/RECONSTRUCTED_section_00_source_train_normal_0001_car_B1_spd_31V_mic_1.wav"
# sf.write(output_path, audio, 16000)
# reconstructed_audio_data, sampling_rate = librosa.load(output_path, sr=None)