import torch
import numpy as np

from scipy.stats import gamma
from torch.nn.functional import mse_loss

class Evaluator:
    def __init__(self):
        pass

    def reconstruct_clips(self, model, input_data, clip_lengths, device):
        output, mse = self.evaluate_model()
        output_features = np.vstack(output)
        clips = self.bundle(output_features, clip_lengths)

        return clips
    
    def evaluate_model(self, model, input_data, device):
        output_features = []
        mse_list = []

        model.eval()
        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for data in input_data:
                input = data.to(device)
                output = model(input)

                output_features.append(output.cpu().numpy())  # Store the reconstruction for later use

                mse = mse_loss(output, input, reduction='sum').item()
                mse_list.append(mse)
                total_mse += mse
                total_samples += input.numel()
        
        average_mse = total_mse / total_samples

        return output_features, average_mse

    def bundle(self, data, clip_lengths):
        clips = []

        start_index = 0

        for size in clip_lengths:
            end_index = start_index + size
            clips.append(data[start_index:end_index])
            start_index = end_index

        return clips
    
    def reconstruction_error(self, original, reconstructed):
        recon_err_per_clip = self.mse(original, reconstructed)

        return recon_err_per_clip

    def mse(self, original, reconstructed):
        error_array = []

        for i in range(len(original)):
            original_sample = original[i].reshape(-1)
            reconstructed_sample = reconstructed[i].reshape(-1)
            
            error_array.append(np.mean((original_sample - reconstructed_sample)**2))
        
        return error_array
        

    def gamma_distribution(self, error):
        shape, location, scale = gamma.fit(error)

        x = np.linspace(0, max(error), 1000)
        gamma_pdf = gamma.pdf(x, shape, loc=location, scale=scale)

        anomaly_threshold = gamma.ppf(0.9, shape, loc=location, scale=scale)

        return gamma_pdf, anomaly_threshold
    
    def confusion_matrix(self):
        pass

    def predictions(self, values, threshold):
        return (values > threshold).astype(int)

    def roc_auc(self):
        pass

    def performance_metrics(self):
        pass

    def compute_metrics(self):
        pass

    def plot_metrics(self):
        pass

    def save_results(self):
        pass

