import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from deep_peakfinder_utils import moving_average_filter_numpy

class PeaksDataset(Dataset):
    def __init__(self, h5_file):
        """
        Dataset for correlation matrices.

        Parameters:
            correlation_matrices (torch.Tensor): Tensor of shape (num_samples, num_blocks, num_channels, num_channels)
                                                 containing correlation matrices.
        """
        self.h5_file = h5_file
        self.length = 0
        verbose = False
        with h5py.File(h5_file, 'r') as hdf:
            self.length = hdf['signal_fft'].shape[0]
            # List all groups in the file
            print(f"Keys in the file:")
            for key in hdf.keys():
                data = hdf[key][:]
                print(f"Data in dataset \"{key}\":")
                if verbose:
                    print(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hdf:
            channelized_iq_signal = torch.tensor(hdf['signal_fft'][idx], dtype=torch.float32)
            # signal_imag = torch.tensor(hdf['signal_fft_imag'][idx], dtype=torch.float32)
            # correlation_matrix = correlation_real + 1j * correlation_imag
            # channelized_iq_signal = torch.cat((signal_real, signal_imag))
            peaks_true = torch.tensor(hdf['peaks_true_binary_mask'][idx], dtype=torch.float32)
        return channelized_iq_signal, peaks_true


def create_peaks_dataset(inputfile, outputfile, num_samples=400, num_peaks=3):
    # with (h5py.File(outputfile, "w") as hdf):
    with h5py.File(outputfile, "w") as hdf:
        signal_fft_dataset = hdf.create_dataset("signal_fft",
                                                shape=(0, 2, num_samples),
                                                maxshape=(None, 2, num_samples), dtype='float32')
        peaks_true_index_list_dataset = hdf.create_dataset('peaks_true_index_list',
                                                           shape=(0, num_peaks),
                                                           maxshape=(None, num_peaks), dtype='float32')
        peaks_true_binary_mask_dataset = hdf.create_dataset('peaks_true_binary_mask',
                                                            shape=(0, 1, num_samples),
                                                            maxshape=(None, 1, num_samples), dtype='int32')

        with open(inputfile, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            rows = []
            for row in reader:
                rows.append(row)
            num_records = len(rows)
            # for index in np.arange(7000, 7001):
            for index in np.arange(num_records):
                row = rows[index]
                i_up_ramp = np.array(row[0:200], dtype=np.complex64)
                i_down_ramp = np.array(row[200:400], dtype=np.complex64)
                q_up_ramp = np.array(row[400:600], dtype=np.complex64)
                q_down_ramp = np.array(row[600:800], dtype=np.complex64)
                timestamp = row[800]
                up_ramp = i_up_ramp + 1j * q_up_ramp
                down_ramp = i_down_ramp + 1j * q_down_ramp
                signal = np.concatenate((up_ramp, down_ramp))
                signal -= np.mean(signal)
                signal_fft = np.fft.fftshift(np.fft.fft(signal))
                signal_fft_real = np.real(signal_fft)
                signal_fft_imag = np.imag(signal_fft)

                function_values = np.abs(signal_fft)
                peaks, _ = find_peaks(function_values, distance=10)
                # Get the indices of the top 3 peaks
                top_N_peaks = peaks[np.argsort(function_values[peaks])[-num_peaks:]]
                peaks_true_mask = np.zeros((1, num_samples))
                peaks_true_mask[0, top_N_peaks] = 1
                # Append to the HDF5 file incrementally
                num_records = signal_fft_dataset.shape[0] + 1
                new_signal_shape = (signal_fft_dataset.shape[0] + 1,) + signal_fft_dataset.shape[1:]
                new_peaks_index_shape = (peaks_true_index_list_dataset.shape[
                                             0] + 1,) + peaks_true_index_list_dataset.shape[1:]
                new_peaks_mask_shape = (peaks_true_binary_mask_dataset.shape[
                                            0] + 1,) + peaks_true_binary_mask_dataset.shape[1:]
                print(f"Expanding the dataset to {num_records} elements: " +
                      f"signal_fft.shape = {new_signal_shape} " +
                      f"peaks_true_index_list.shape = {new_peaks_index_shape} " +
                      f"peaks_true_binary_mask_dataset.shape = {new_peaks_mask_shape} ")
                channelized_signal = torch.tensor(np.stack((signal_fft_real, signal_fft_imag)))
                signal_fft_dataset.resize(new_signal_shape)
                signal_fft_dataset[-1:] = channelized_signal

                peaks_true_index_list_dataset.resize(new_peaks_index_shape)
                peaks_true_index_list_dataset[-1:] = torch.tensor(top_N_peaks)

                peaks_true_binary_mask_dataset.resize(new_peaks_mask_shape)
                peaks_true_binary_mask_dataset[-1:] = torch.tensor(peaks_true_mask)


def generate_peak_vector(vector_length=400, num_peaks=1, min_base_width=5, max_base_width=20, min_height=0.5,
                         max_height=1.0, min_spacing=5, noise_floor=0.3, verbose=False):
    """
    Generates a 400-element vector with triangular peaks, ensuring no peaks overlap or occur within `min_spacing` indices.

    Parameters:
    - vector_length: Length of the vector (default 400).
    - num_peaks: Number of peaks to insert (default 1).
    - min_base_width: Minimum base width of the peaks.
    - max_base_width: Maximum base width of the peaks.
    - min_height: Minimum height of the peaks.
    - max_height: Maximum height of the peaks.
    - min_spacing: Minimum spacing between peaks (default 5).

    Returns:
    - vector: The generated 400-element vector.
    - labels: A binary vector indicating peak locations.
    """
    # Initialize the vector and label
    # vector = np.zeros(vector_length)
    vector = noise_floor * np.ones(vector_length) + np.random.uniform(low=-0.05, high=0.05, size=vector_length)
    labels = np.zeros(vector_length)
    occupied_indices = set()  # Track indices already occupied by peaks

    for _ in range(num_peaks):
        attempts = 0
        while attempts < 100:  # Prevent infinite loops
            # Randomly select peak properties
            peak_center = np.random.randint(0, vector_length)
            base_width = np.random.randint(min_base_width, max_base_width)
            height = np.random.uniform(min_height, max_height)

            # Define the range of the triangle
            half_width = base_width // 2
            start = max(0, peak_center - half_width)
            end = min(vector_length, peak_center + half_width + 1)

            # Check if the peak overlaps with any existing peaks
            if all(idx not in occupied_indices for idx in range(start - min_spacing, end + min_spacing)):
                # Create the triangular peak
                for i in range(start, end):
                    distance = abs(i - peak_center)
                    vector[i] += height * (1 - distance / half_width)
                    # labels[i] = 1  # Mark as a peak location
                    if np.abs(peak_center - i) <= min_spacing:
                        occupied_indices.add(i)  # Mark indices as occupied
                labels[peak_center] = 1.0  # Mark as a peak location
                if vector[peak_center] < 0.3 and verbose:
                    print(f"low peak @ {peak_center} height = {height} (min,max)=({min_height},{max_height})")
                break  # Move to the next peak
            attempts += 1

        if attempts == 100 and verbose:
            print("Could not place all peaks due to spacing constraints.")

    return vector, labels


def generate_dataset(outputfile, num_samples=1000, show=False, verbose=False, **kwargs):
    """
    Generates a dataset of 400-element vectors with triangular peaks.

    Parameters:
    - num_samples: Number of samples to generate.
    - kwargs: Additional arguments for the generate_peak_vector function.

    Returns:
    - data: Array of shape (num_samples, 400) containing the vectors.
    - labels: Array of shape (num_samples, 400) containing the binary labels.
    """
    # data = []
    # labels = []
    peak_generation_config = {
        "vector_length": 400,
        "num_peaks": 25,
        "min_base_width": 5,
        "max_base_width": 30,
        "min_height": 0.5,
        "max_height": 1.0,
        "min_spacing": 3,
        "noise_floor": 0.1,
    }
    # print(peak_generation_config)

    # with (h5py.File(outputfile, "w") as hdf):
    with h5py.File(outputfile, "w") as hdf:
        signal_fft_dataset = hdf.create_dataset("signal_fft",
                                                shape=(0, 6, peak_generation_config["vector_length"]),
                                                maxshape=(None, 6, peak_generation_config["vector_length"]),
                                                dtype='float32')
        # peaks_true_index_list_dataset = hdf.create_dataset('peaks_true_index_list',
        #                                                    shape=(0,),
        #                                                    maxshape=(None,), dtype='float32')
        peaks_true_binary_mask_dataset = hdf.create_dataset('peaks_true_binary_mask',
                                                            shape=(0, 1, peak_generation_config["vector_length"]),
                                                            maxshape=(None, 1, peak_generation_config["vector_length"]),
                                                            dtype='int32')
        for sample_index in range(num_samples):
            if sample_index % 10000 == 0:
                print(f"Generating sample {sample_index} of {num_samples}...")
            peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
                                                                                                             high=0.2)
            peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            i_vector, i_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            n_1_vector = i_vector

            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            # i3_vector, i3_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # n_3_vector = moving_average_filter_numpy(i3_vector.reshape((1,1,400)), 3)

            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            # i5_vector, i5_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # n_5_vector = moving_average_filter_numpy(i5_vector.reshape((1,1,400)), 5)

            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            # i7_vector, i7_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # n_7_vector = moving_average_filter_numpy(i7_vector.reshape((1,1,400)), 7)

            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            # i9_vector, i9_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # n_9_vector = moving_average_filter_numpy(i9_vector.reshape((1,1,400)), 9)

            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=6))
            # i11_vector, i11_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # n_11_vector = moving_average_filter_numpy(i11_vector.reshape((1,1,400)), 11)

            n_3_vector = moving_average_filter_numpy(i_vector.copy().reshape((1,1,400)), 3)
            n_5_vector = moving_average_filter_numpy(i_vector.copy().reshape((1,1,400)), 5)
            n_7_vector = moving_average_filter_numpy(i_vector.copy().reshape((1,1,400)), 7)
            n_9_vector = moving_average_filter_numpy(i_vector.copy().reshape((1,1,400)), 9)
            n_11_vector = moving_average_filter_numpy(i_vector.copy().reshape((1,1,400)), 11)

            n_3_vector = n_3_vector.reshape((-1))
            n_5_vector = n_5_vector.reshape((-1))
            n_7_vector = n_7_vector.reshape((-1))
            n_9_vector = n_9_vector.reshape((-1))
            n_11_vector = n_11_vector.reshape((-1))
            # peak_generation_config["min_height"] = np.random.uniform(low=0.15, high=0.3)
            # peak_generation_config["noise_floor"] = peak_generation_config["min_height"] * np.random.uniform(low=0.1,
            #                                                                                                  high=0.2)
            # moving_average_filter_numpy(data, N)
            # peak_generation_config["num_peaks"] = int(np.random.uniform(low=2, high=18))
            # peak_generation_config["min_spacing"] = int(np.random.uniform(low=3, high=5))
            # q_vector, q_peaks_true_mask = generate_peak_vector(**peak_generation_config)
            # peaks_true_mask = i_peaks_true_mask + q_peaks_true_mask
            peaks_true_mask = i_peaks_true_mask # + i3_peaks_true_mask + i5_peaks_true_mask + i7_peaks_true_mask + i9_peaks_true_mask + i11_peaks_true_mask
            peaks_true_mask = np.clip(peaks_true_mask, a_min=None, a_max=1)
            # # peaks_true_mask =
            # # scale_factor = np.random.uniform(low=0.3, high=0.6, size=400)
            # scale_factor = 1.0
            # signal_fft_real = scale_factor * i_vector
            # signal_fft_imag = (1 - scale_factor) * q_vector

            i_true_peak_indices = np.where(i_peaks_true_mask)[0]
            # q_true_peak_indices = np.where(q_peaks_true_mask)[0]
            if show:
                # Plot data on each subplot
                plt.plot(np.arange(400), n_1_vector, 'r--')
                plt.plot(np.arange(400), n_3_vector, 'b--')
                plt.plot(np.arange(400), n_5_vector, 'g--')
                plt.plot(np.arange(400), n_7_vector, 'c--')
                plt.plot(np.arange(400), n_9_vector, 'm--')
                plt.plot(np.arange(400), n_11_vector, 'k--')
                plt.plot(i_true_peak_indices, i_vector[i_true_peak_indices], 'r*')
                plt.title("Example of Peak Vector Sample")
                # plt.plot(np.arange(400), i_vector, 'r--')
                # plt.plot(i_true_peak_indices, i_vector[i_true_peak_indices], 'r*')
                # plt.plot(np.arange(400), q_vector, 'b--')
                # plt.plot(q_true_peak_indices, q_vector[q_true_peak_indices], 'bo')
                plt.show()
            num_records = signal_fft_dataset.shape[0] + 1
            new_signal_shape = (signal_fft_dataset.shape[0] + 1,) + signal_fft_dataset.shape[1:]
            # new_peaks_index_shape = (peaks_true_index_list_dataset.shape[
            #                              0] + 1,) + peaks_true_index_list_dataset.shape[1:]
            new_peaks_mask_shape = (peaks_true_binary_mask_dataset.shape[
                                        0] + 1,) + peaks_true_binary_mask_dataset.shape[1:]
            if verbose:
                print(f"Expanding the dataset to {num_records} elements: " +
                      f"signal_fft.shape = {new_signal_shape} " +
                      # f"peaks_true_index_list.shape = {new_peaks_index_shape} " +
                      f"peaks_true_binary_mask_dataset.shape = {new_peaks_mask_shape} ")
            # channelized_signal = torch.tensor(np.stack((signal_fft_real, signal_fft_imag)))
            channelized_signal = torch.tensor(np.stack((n_1_vector, n_3_vector, n_5_vector, n_7_vector, n_9_vector, n_11_vector)))
            signal_fft_dataset.resize(new_signal_shape)
            signal_fft_dataset[-1:] = channelized_signal

            # peaks_true_index_list_dataset.resize(new_peaks_index_shape)
            # peaks_true_index_list_dataset[-1:] = torch.tensor(true_peak_indices)

            peaks_true_binary_mask_dataset.resize(new_peaks_mask_shape)
            peaks_true_binary_mask_dataset[-1:] = torch.tensor(peaks_true_mask)
            # data.append(vector)
            # labels.append(label)

    # return np.array(data), np.array(labels)


if __name__ == "__main__":
    outputfile = 'IQ_synthetic_v04.h5'
    show = True
    num_samples = 250000
    generate_dataset(outputfile, num_samples, show=show)
    if True:
        exit()
    num_peaks = 2
    create_peaks_dataset('IQ-2.csv', 'IQ-2_shifted.h5', num_samples=400, num_peaks=num_peaks)

    with open('IQ-2.csv', 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        rows = []
        for row in reader:
            rows.append(row)
        for index in np.arange(7000, 7001):
            # for row in rows:
            row = rows[index]
            i_up_ramp = np.array(row[0:200], dtype=np.complex64)
            i_down_ramp = np.array(row[200:400], dtype=np.complex64)
            q_up_ramp = np.array(row[400:600], dtype=np.complex64)
            q_down_ramp = np.array(row[600:800], dtype=np.complex64)
            timestamp = row[800]
            up_ramp = i_up_ramp + 1j * q_up_ramp
            down_ramp = i_down_ramp + 1j * q_down_ramp
            signal = np.concatenate((up_ramp, down_ramp))
            signal -= np.mean(signal)
            signal_fft = np.fft.fft(signal)
            # Find peaks
            function_values = np.abs(signal_fft)
            peaks, _ = find_peaks(function_values, distance=10)
            # Get the indices of the top 3 peaks
            top_N_peaks = peaks[np.argsort(function_values[peaks])[-num_peaks:]]
            # plt.plot(np.arange(200), np.abs(up_ramp), 'r--')
            # plt.plot(np.arange(200,400), np.abs(down_ramp), 'b--')
            # plt.plot(np.arange(400), np.abs(signal_fft), 'r--')
            # plt.plot(np.arange(400), 20*np.log10(np.abs(signal_fft)), 'r--')
            # Create the figure and subplots
            fig, axs = plt.subplots(4, 1, figsize=(8, 8))

            # Plot data on each subplot
            axs[0].plot(np.arange(400), function_values, 'r--')
            axs[0].plot(top_N_peaks, function_values[top_N_peaks], 'bo')
            axs[0].set_title('Signal')

            axs[1].plot(np.arange(200), function_values[0:200], 'r--')
            axs[1].set_title('UpRamp')

            axs[2].plot(np.arange(200, 400), function_values[200:400], 'r--')
            axs[2].set_title('DownRamp')

            # axs[3].plot(x, np.exp(x))
            # axs[3].set_title('Exponential')

            # Adjust layout to prevent overlapping
            plt.tight_layout()

            # Show the plot
            plt.show()
            # plt.plot(np.arange(200,400), np.abs(down_ramp), 'b--')
            # plt.axis((0, 6, 0, 20))
            break
