import numpy as np
import pandas as pd
import cv2
import os
import torch
import h5py
import scipy

Fs = 200000
Fs_CW = 25000
max_voltage = 3.3
ADC_bits = 12
ADC_intervals = 2^ADC_bits
max_fd = 12500
mode = 3
f0 = 5
BW = 240
Ns = 200
Ntar = 5
Rmax = 100
MTI = 0
Mth = 1
N_FFT = 4096
c = 299792458
RampTimeReal = 0.001
RampTimeReal2 = 0.00075
factorPresencia_CW = 40
factorPresencia_FMCW = 22.58

BW_actual = BW * 1000000
f0_v = f0*1000000000 + BW_actual/2
max_velocity = c/(2*f0_v) * max_fd
max_distance = c/(2*BW_actual) * Fs/2 * RampTimeReal

class DepthDataset(torch.utils.data.Dataset):
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
            peaks_true = torch.tensor(hdf['images'][idx], dtype=torch.float32)
        return channelized_iq_signal, peaks_true

def test_deleteBadSamples(I, Q):
    # Outputs: I, Q
    # Appears to be adding the average to the final samples if they're zero
    # I say appears to be because looking at the average value it's close but different
    # So far I and Q bad samples appear to be in the same spots at the end (as in I and Q doesn't have different number of bad samples)
    SumI = np.sum(I)
    SumQ = np.sum(Q)
    count = 0
    index = -1
    while I[index] == 0:
        count = count + 1
        index = index - 1
    AvgI = SumI / (len(I) - count)
    AvgQ = SumQ / (len(Q) - count)
    index = -1
    while I[index] == 0:
        I[index] = AvgI
        Q[index] = AvgQ
        index = index - 1
    return I, Q

def convert_IQ_to_FFT(test_i,test_q):
    test_i, test_q = test_deleteBadSamples(test_i, test_q)
    test_i = np.subtract(np.multiply(test_i, max_voltage/ADC_intervals), np.mean(np.multiply(test_i, max_voltage/ADC_intervals)))
    test_q = np.subtract(np.multiply(test_q, max_voltage/ADC_intervals), np.mean(np.multiply(test_q, max_voltage/ADC_intervals)))

    ComplexVector = test_i + 1j*test_q

    ComplexVector = ComplexVector * np.hanning(Ns) * 2 / 3.3

    FrequencyDomain = 2*np.absolute(np.fft.fftshift(np.fft.fft(ComplexVector/Ns, N_FFT)))
    start = int(N_FFT/2)
    FrequencyDomain[start] = FrequencyDomain[start - 1]
    FrequencyDomain = 20 * np.log10(FrequencyDomain)

    return FrequencyDomain

def get_dataset_fft_max_min(I,Q):
    max_value = -np.Inf
    min_value = np.Inf

    for idx in range(len(I)):
        test_i = I[idx,:]
        test_q = Q[idx,:]

        FrequencyDomain = convert_IQ_to_FFT(test_i, test_q)

        max_value = np.max([max_value, np.max(FrequencyDomain)])
        min_value = np.min([min_value, np.min(FrequencyDomain)])

    return max_value, min_value

def get_dataset_images_depth_values(images, folder_path):
    max_depth = -np.Inf
    mean_depth = 0
    mean_Q = np.array([0,0,0])

    for idx in range(len(images)):
        depth_I = cv2.imread(folder_path + "/" + images[idx], cv2.IMREAD_UNCHANGED)
        depth_I = depth_I[:,25:]
        mean_depth = mean_depth + np.mean(depth_I)
        max_depth = np.max([max_depth, np.max(depth_I)])
        sort_I = np.sort(depth_I, axis=None)
        Q1_idx = int(len(sort_I) * 0.25)
        Q2_idx = int(len(sort_I) * 0.5)
        Q3_idx = int(len(sort_I) * 0.75)

        mean_Q = mean_Q + np.array([sort_I[Q1_idx], sort_I[Q2_idx], sort_I[Q3_idx]])

    mean_depth = mean_depth / len(images)
    mean_Q = mean_Q / len(images)

    return max_depth, mean_depth, mean_Q

def create_h5_dataset(I, Q, sync_idx, images, output_file, folder_path):
    with h5py.File(output_file, "w") as hdf:
        signal_fft_dataset = hdf.create_dataset("signal_fft",
                                                shape=(0,15, 500),
                                                maxshape=(None, 15, 500),
                                                dtype='float32')
        images_dataset = hdf.create_dataset("images",
                                            shape=(0, 64, 64),
                                            maxshape=(None, 64, 64),
                                            dtype='float32')
        
        fft_max, fft_min = get_dataset_fft_max_min(I, Q)
        depth_max, depth_mean, depth_Qrange = get_dataset_images_depth_values(images, folder_path)

        filter_size = 7
        filter = np.ones((filter_size, filter_size)) / (filter_size**2)

        for idx in np.arange(7, len(I)-7):
            fft_holder = torch.zeros((15,500))
            fft_counter = 0
            start = int(N_FFT/2)
            for fft_idx in np.arange(idx-7, idx+8):
                temp_fft = convert_IQ_to_FFT(I[fft_idx,:], Q[fft_idx,:])
                temp_fft = torch.from_numpy(temp_fft)
                fft_holder[fft_counter,:] = (temp_fft[start:start+500] - fft_min) / (fft_max - fft_min)
                fft_counter = fft_counter + 1
        
            depth_I = cv2.imread(folder_path + "/" + images[int(sync_idx[idx,2])], cv2.IMREAD_UNCHANGED)
            depth_I = depth_I[:,25:]
            depth_I[depth_I < depth_Qrange[0]*0.5] = 2200.0
            depth_I[depth_I > depth_Qrange[2]] = 2200.0
            depth_I = scipy.signal.convolve2d(depth_I, filter, mode="same")
            depth_I = depth_I / depth_max
            depth_I = depth_I[31:95, 51:115]
            
            num_records = signal_fft_dataset.shape[0] + 1
            new_signal_shape = (signal_fft_dataset.shape[0] + 1,) + signal_fft_dataset.shape[1:]
            new_image_shape = (images_dataset.shape[0] + 1,) + images_dataset.shape[1:]

            print(f"Expanding the dataset to {num_records} elements: " + 
                  f"signal_fft.shape = {new_signal_shape} " +
                  f"images.shape = {new_image_shape} ")
            
            signal_fft_dataset.resize(new_signal_shape)
            signal_fft_dataset[-1:] = fft_holder

            images_dataset.resize(new_image_shape)
            images_dataset[-1:] = torch.from_numpy(depth_I)

if __name__ == "__main__":
    folder_name = "02-07-2025_15-06-16-737244"
    folder_path = "./datasets/"+folder_name
    output_file = folder_name+".h5"

    I = pd.read_csv(folder_path+"/I.csv", header=None)
    I = I.to_numpy()
    I = I[:,1:]
    # print(len(I))
    Q = pd.read_csv(folder_path+"/Q.csv", header=None)
    Q = Q.to_numpy()
    Q = Q[:,1:]

    sync_idx = pd.read_csv(folder_path+"/synced/synced_idx.csv", header=None)
    sync_idx = sync_idx.to_numpy()

    images_folder_path = folder_path+"/images/depth/"
    images = os.listdir(images_folder_path)

    create_h5_dataset(I, Q, sync_idx, images, output_file, images_folder_path)