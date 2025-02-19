import numpy as np
import pandas as pd
import cv2
import os
import torch
import h5py
import scipy
import matplotlib.pyplot as pyplot
from avl_urad_create_h5 import DepthDataset

if __name__ == "__main__":
    dataset_name = "02-07-2025_15-06-16-737244.h5"

    with h5py.File(dataset_name, 'r') as hdf:
            idx = 0
            channelized_iq_signal = torch.tensor(hdf['signal_fft'][idx], dtype=torch.float32)
            # signal_imag = torch.tensor(hdf['signal_fft_imag'][idx], dtype=torch.float32)
            # correlation_matrix = correlation_real + 1j * correlation_imag
            # channelized_iq_signal = torch.cat((signal_real, signal_imag))
            peaks_true = torch.tensor(hdf['images'][idx], dtype=torch.float32)

            pyplot.plot(channelized_iq_signal[6,:])
            pyplot.show()
            cv2.imshow("test",peaks_true.cpu().numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()