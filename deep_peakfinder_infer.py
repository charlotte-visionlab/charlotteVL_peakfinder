import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

from deep_peakfinder_model import ResNet1D, PeakDetectionNet, PeakDetectionXformNet
from deep_peakfinder_loss import IoULoss, SparsePeakDetectionLoss, SparsePeakLoss
from deep_peakfinder_csv2h5 import PeaksDataset
from deep_peakfinder_utils import moving_average_filter_pytorch, moving_average_filter_numpy
import pandas as pd
import matplotlib.pyplot as plt


# Function to display predictions, ground truth, and loss
def display_predictions_vs_ground_truth(predictions, ground_truth, loss_fn, verbose=False):
    """
    Display predictions, ground truth values, and the loss value in a tabular format.

    Parameters:
        predictions (torch.Tensor): Predicted values from the model, shape (batch_size, 1).
        ground_truth (torch.Tensor): Ground truth values, shape (batch_size, 1).
        loss_fn (callable): Loss function to compute the loss between predictions and ground truth.

    Returns:
        None
    """
    # Ensure predictions and ground_truth are 1D tensors
    predictions = predictions.view(-1).cpu()
    ground_truth = ground_truth.view(-1).cpu()

    # Compute loss for each element
    losses = [loss_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item() for pred, gt in zip(predictions, ground_truth)]
    print(f"Max loss = {np.max(losses)}")
    print(f"Mean loss = {np.mean(losses)}")
    print(f"Std. loss = {np.std(losses)}")
    # Create a pandas DataFrame for tabular display
    data = {
        "Prediction": predictions.numpy(),
        "Ground Truth": ground_truth.numpy(),
        "Loss": losses
    }
    df = pd.DataFrame(data)
    # Ensure all rows are displayed
    if verbose:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df)
    else:
        print(df)


# Function to display predictions, ground truth, and loss
def visualize_predictions_vs_ground_truth(batch_signal_fft_vals, predictions, ground_truth, loss_fn):
    batch_sample_list = []
    for index in range(batch_signal_fft_vals.shape[0]):
        batch_sample = {
            "signal": batch_signal_fft_vals[index].numpy(),
            "signal_mean": torch.mean(batch_signal_fft_vals[index,:], dim=0).cpu().numpy(),
            "predictions": predictions[index].squeeze(0).cpu().numpy(),
            "ground_truth": ground_truth[index].squeeze(0).cpu().numpy(),
        }
        batch_sample_list.append(batch_sample)

    for result in batch_sample_list:
        # num_classifications = result["ground_truth"].shape[1]
        true_peak_indices = np.where(result["ground_truth"] == 1)
        predicted_peak_indices = np.where(result["predictions"] == 1)
        print(f"predicted indices = {predicted_peak_indices}" +
              f"true indices = {true_peak_indices}")
        # Plot the result
        plt.figure(figsize=(10, 6))
        plt.plot(result["signal"][0, :], "r--", label="X_real(w)")
        plt.plot(result["signal"][1, :], "b--", label="X_imag(w)")
        # plt.plot(np.squeeze(true_peak_indices),
        #          result["signal_mean"][true_peak_indices], 'bo', label="Detected Peaks")
        plt.plot(np.squeeze(predicted_peak_indices),
                 result["signal_mean"][predicted_peak_indices], 'ro', label="Detected Peaks")
        # plt.plot(labels, label="Labels", linestyle="--", alpha=0.7)
        plt.title("Peak Detection Result")
        plt.xlabel("Frequency (Index)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()



# Example usage
if __name__ == "__main__":
    # Path to saved weights
    # saved_weights_path = "weights_peaks/model_2025-01-07_16-12-07_epoch_100.pth"
    # saved_weights_path = "weights_peaks/model_2025-01-09_02-50-16_epoch_100.pth"
    saved_weights_path = "weights_peaks/model_2025-01-13_16-00-15_epoch_135.pth"
    # dataset_filename = "IQ-2_shifted.h5"
    dataset_filename = "IQ_synthetic_v03.h5"
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    peakfinder_dataset = PeaksDataset(dataset_filename)
    data_loader = DataLoader(peakfinder_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    # Instantiate model, optimizer, and loss function
    input_shape = (2, 400)
    # Example Configuration
    block_configs = [
        {"in_channels": 2, "out_channels": 16, "kernel_size": 3, "stride": 1},
        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 1},
        {"in_channels": 128, "out_channels": 32, "kernel_size": 3, "stride": 1},
        {"in_channels": 32, "out_channels": 16, "kernel_size": 3, "stride": 1},
        {"in_channels": 16, "out_channels": 4, "kernel_size": 3, "stride": 1},
    ]
    num_peaks = 6
    # model = ResNet1D(input_channels=input_shape, output_size=num_peaks, block_configs=block_configs).to(device)

    # model = PeakDetectionNet()
    # loss_function = nn.BCELoss().to(device)  # For binary classification

    model = PeakDetectionXformNet(input_channels=2, output_length=400, num_classes=1)
    # loss_function = nn.BCELoss(weight=torch.tensor(197.0/3.0)).to(device)  # For binary classification
    loss_function = SparsePeakDetectionLoss(alpha=1.0, gamma=2.0).to(device)

    print(model)
    # Load saved weights
    model.load_state_dict(torch.load(saved_weights_path, weights_only=True, map_location=torch.device(device)))
    model.eval()  # Set model to evaluation mode

    # Move model and data to device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=input_shape)

    N = 5
    # loss_function = nn.MSELoss().to(device)
    # loss_function = SparsePeakDetectionLoss().to(device)
    # Perform inference
    with torch.no_grad():  # No gradient computation for inference
        for batch_data in data_loader:
            batch_signal_fft_vals, batch_peak_true_vals = batch_data
            batch_signal_fft_vals = torch.abs(batch_signal_fft_vals)
            batch_signal_fft_vals = batch_signal_fft_vals / torch.max(batch_signal_fft_vals)
            # smoothed_batch_signal_fft_vals = moving_average_filter_numpy(batch_signal_fft_vals, N).astype(np.float32)
            # batch_signal_fft_vals = torch.tensor(smoothed_batch_signal_fft_vals)
            smoothed_batch_signal_fft_vals = moving_average_filter_pytorch(batch_signal_fft_vals, N)
            batch_signal_fft_vals = smoothed_batch_signal_fft_vals
            input_data = batch_signal_fft_vals.to(device)
            ground_truth = batch_peak_true_vals.to(device)
            batch_y_pred = model(input_data)
            predicted_classes = torch.sigmoid(batch_y_pred)  # Convert logits to probabilities
            predicted_classes = (predicted_classes > 0.5).float()  # Threshold probabilities to binary
            # predicted_classes = torch.round(batch_y_pred[:,1,:]).unsqueeze(1)
            loss = loss_function(batch_y_pred, ground_truth)
            # Display the predictions, ground truth, and losses
            # display_predictions_vs_ground_truth(predictions, ground_truth, loss_function, verbose=False)
            visualize_predictions_vs_ground_truth(batch_signal_fft_vals, predicted_classes, ground_truth, loss_function)
            # print("Predictions:", predictions)
