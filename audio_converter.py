import yaml
import librosa
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader

class AudioConverter:
    def __init__(self):
        # print("Audio Converter Here!")

        self.hyper_parameters = self.load_hyper_parameters()

        #> setup general acoustic features
        self.acoustic_features = self.hyper_parameters['acoustic_features']
        self.number_of_frames_to_concatenate = self.acoustic_features['number_of_frames_to_concatenate']
        self.frame_size_seconds = self.acoustic_features['frame_size_seconds']
        self.frame_size_samples = self.acoustic_features['frame_size_samples']
        self.hop_size_samples = self.acoustic_features['hop_size_samples']
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
        number_of_mels = self.number_of_mels

        #> conversion to mel spectrogram
        mel_audio = librosa.feature.melspectrogram(
            y           = amplitude,
            sr          = sample_rate,
            n_fft       = frame_size_samples,
            hop_length  = hop_size_samples,
            n_mels      = number_of_mels
        )
        mel_audio = librosa.power_to_db(mel_audio, ref=np.max) #convert to decibels

        return mel_audio.shape[1], mel_audio
    
    def wav_to_input_with_freq(self, wav_path, n=10):
        clip_length, audio = self.wav_to_input(wav_path)
        top_frequencies = self.get_top_frequencies(wav_path, n=n)

        new_columns = np.tile(np.array(top_frequencies).reshape(1, -1), (clip_length, 1))

        result = np.concatenate((audio, new_columns), axis=1)

        return clip_length, result

    def get_top_frequencies(self, wav_path, n=10):
        amplitude, sampling_rate = librosa.load(wav_path, sr=None, mono=True)
        stft = librosa.stft(amplitude, n_fft=self.frame_size_samples, hop_length=self.hop_size_samples)
        
        magnitudes = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sampling_rate, n_fft=self.frame_size_samples)

        mean_magnitudes = np.mean(magnitudes, axis=1)
        top_indices = np.argsort(mean_magnitudes)[-n:][::-1]

        top_frequencies = frequencies[top_indices]

        return top_frequencies


    def mel_to_input(self, mel_audio):
        trimmed_audio = self.trim(mel_audio, mel_audio.shape[1])
        concatenated_audio = self.concatenate(trimmed_audio)
        return concatenated_audio.shape[0], concatenated_audio
        

    def trim(self, mel_audio, number_of_frames):
        trimmed_frames = number_of_frames - (number_of_frames % self.number_of_frames_to_concatenate) #e.g. 376 - (376 % 5) => 376 - 1 = 375
        trimmed_audio = mel_audio[:, :trimmed_frames] #trim the mel audio to be divisible by 5
        return trimmed_audio


    def concatenate(self, mel_audio):
        concatenated_audio = mel_audio.reshape(mel_audio.shape[0], mel_audio.shape[1] // self.number_of_frames_to_concatenate, self.number_of_frames_to_concatenate) #reshape to (number_of_mels, number_of_frames // 5, 5) which is (128, 75, 5)
        concatenated_audio = concatenated_audio.transpose(1, 0, 2) #transpose to (number_of_frames // 5, number_of_mels, 5) which is (75, 128, 5)
        concatenated_audio = concatenated_audio.reshape(concatenated_audio.shape[0], -1) #reshape to (number_of_frames // 5, number_of_mels * 5) which is (75, 640)
        return concatenated_audio


    def output_to_mel(self, output, number_of_mels=128):
        output_reshaped = output.reshape(output.shape[0], number_of_mels, self.number_of_frames_to_concatenate)
        output_reshaped = output_reshaped.transpose(1, 0, 2)
        output_reshaped = output_reshaped.reshape(number_of_mels, -1)

        return output_reshaped

    def mel_to_wav(self, mel_audio):
        linear_audio = librosa.db_to_power(mel_audio, ref=1.0) #convert back to linear scale
        
        mel_to_linear = librosa.feature.inverse.mel_to_audio(
            linear_audio,
            sr=16000,
            n_fft=self.frame_size_samples,
            hop_length=self.hop_size_samples
        )

        return mel_to_linear
    
    
    def wav_to_input(self, wav_path):
        lengths, mel_audio = self.wav_to_mel(wav_path)
        return self.mel_to_input(mel_audio)
    

    def output_to_wav(self, output):
        lengths, mel_audio = self.output_to_mel(output)
        return self.mel_to_wav(mel_audio)