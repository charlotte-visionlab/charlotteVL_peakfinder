import numpy as np
import torch

def moving_average_filter_numpy(data, N):
    """
    Smooth a batch of data along the third dimension using an N-point moving average filter.

    Parameters:
        data (numpy.ndarray): Input data of shape (X, 2, 200).
        N (int): Number of points for the moving average.

    Returns:
        numpy.ndarray: Smoothed data of the same shape as input.
    """
    if N < 1 or N > data.shape[2]:
        raise ValueError("N must be between 1 and the size of the third dimension.")

    # Define the moving average kernel
    kernel = np.ones(N) / N

    # Apply the filter along the third dimension for each entry in the batch
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=2, arr=data)

    return smoothed_data

# Example usage:
# X, N = 5, 10  # Example batch size and filter size
# data = np.random.rand(X, 2, 200)  # Example input data
# smoothed_data = moving_average_filter(data, N)
# print(smoothed_data.shape)  # Should output (X, 2, 200)


def moving_average_filter_pytorch(data, N):
    """
    Smooth a batch of data along the last dimension using an N-point moving average filter in PyTorch.

    Parameters:
        data (torch.Tensor): Input data of shape (X, 2, 200). Must be on GPU for GPU computation.
        N (int): Number of points for the moving average.

    Returns:
        torch.Tensor: Smoothed data of the same shape as input.
    """
    if N < 1 or N > data.shape[-1]:
        raise ValueError("N must be between 1 and the size of the last dimension.")

    # Define the moving average kernel
    kernel = torch.ones(1, 1, N, device=data.device) / N  # Kernel for convolution

    # Reshape data to have a channels-like dimension for PyTorch's conv1d
    data = data.view(-1, 1, data.shape[-1])  # Shape: (X * 2, 1, 200)

    # Apply convolution using 'same' padding
    padding = (N - 1) // 2
    smoothed_data = torch.nn.functional.conv1d(data, kernel, padding=padding)

    # Reshape back to original dimensions
    smoothed_data = smoothed_data.view(-1, 2, data.shape[-1])  # Shape: (X, 2, 200)

    return smoothed_data


# Example usage:
# X, N = 5, 10  # Example batch size and filter size
# data = torch.rand(X, 2, 200, device='cuda')  # Example input data on GPU
# smoothed_data = moving_average_filter_torch(data, N)
# print(smoothed_data.shape)  # Should output (X, 2, 200)
