import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from scipy.signal import find_peaks

from avl_models import SignalToImageCNN, ResNetSignalToImage, XformNetSignalToImage
from avl_urad_create_h5 import DepthDataset
import pandas as pd
import matplotlib.pyplot as plt

# Function to display predictions, ground truth, and loss
def visualize_predictions_vs_ground_truth(batch_signal_fft_vals, predictions, ground_truth, loss_fn):

    for result_idx in range(len(batch_signal_fft_vals)):
        
        plt.subplot(1,2,1)
        plt.imshow(predictions[0].cpu().numpy())
        plt.xlabel("Predicted")
        plt.subplot(1,2,2)
        plt.imshow(ground_truth[0].cpu().numpy())
        plt.xlabel("Ground Truth")
        # plt.title("Peak Detection Result")
        # plt.xlabel("Frequency (Index)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.xlim(2048,4095)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Path to saved weights
    # saved_weights_path = "weights_depths/model_2025-03-05_12-53-11_final.pth"
    # saved_weights_path = "weights_depths/model_2025-03-10_17-50-40_final.pth"
    # dataset_filename = "02-07-2025_15-06-16-737244.h5"

    saved_weights_path = "weights_depths/model_2025-04-16_12-40-20_final.pth"
    dataset_filename = "test_dataset.h5"

    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    peakfinder_dataset = DepthDataset(dataset_filename)
    data_loader = DataLoader(peakfinder_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    # Instantiate model, optimizer, and loss function
    input_shape = (15, 500)
    
    model = ResNetSignalToImage()
    # model = XformNetSignalToImage()
    loss_function = nn.MSELoss()

    print(model)
    # Load saved weights
    model.load_state_dict(torch.load(saved_weights_path, weights_only=True, map_location=torch.device(device)))
    model.eval()  # Set model to evaluation mode

    # Move model and data to device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=input_shape)

    # Perform inference
    with torch.no_grad():  # No gradient computation for inference
        for batch_data in data_loader:
            batch_signal_fft_vals, batch_peak_true_vals = batch_data

            input_data = batch_signal_fft_vals.to(device)
            ground_truth = batch_peak_true_vals.to(device)
            batch_y_pred = model(input_data)
            predicted_classes = torch.clamp(batch_y_pred, 0, 1)
            loss = loss_function(batch_y_pred, ground_truth)
            visualize_predictions_vs_ground_truth(batch_signal_fft_vals, predicted_classes, ground_truth, loss_function)
