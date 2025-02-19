import os
import pandas as pd
import csv
import numpy as np

import re

def use_regex(input_text):
    pattern = re.compile(r"depth_([0-9]+)\.tiff", re.IGNORECASE)
    return pattern.findall(input_text)


folder_path = "./02-07-2025_15-06-16-737244/"

imu_file = "Raw_IMU.csv"
i_file = "I.csv"
q_file = "Q.csv"
pose_file = "Pose.csv"
depth_images_folder = "images/depth/"

synced_file = "synced/synced_idx.csv"
synced_timestamps = "synced/synced_timestamps.csv"

I = pd.read_csv(folder_path + i_file)
imu = pd.read_csv(folder_path + imu_file)
depth_images_folder = os.listdir(folder_path + depth_images_folder)

timestamps = I[I.columns[0]].to_list()
I_len = len(timestamps)
imu_timestamps = imu[imu.columns[0]].to_list()

depth_timestamp = []
for images in depth_images_folder:
    temp = use_regex(images)
    depth_timestamp.append(int(temp[0]))
# print(depth_timestamp)

timestamps = np.array(timestamps)
imu_timestamps = np.array(imu_timestamps)
depth_timestamp = np.array(depth_timestamp)

with open(folder_path + synced_file, "w") as file:
    with open(folder_path + synced_timestamps, "w") as file2:
        for idx in range(I_len):
            imu_idx = np.argmin(np.abs(imu_timestamps - timestamps[idx]))
            depth_idx = np.argmin(np.abs(depth_timestamp - timestamps[idx]))
            file.write(f"{idx},{imu_idx},{depth_idx},\n")
            file2.write(f"{timestamps[idx]},{imu_timestamps[imu_idx]},{depth_timestamp[depth_idx]},\n")
            # print([timestamps[idx], imu_timestamps[imu_idx], depth_timestamp[depth_idx]])
            # print([idx, imu_idx, depth_idx])
            print([timestamps[idx], (timestamps[idx]-imu_timestamps[imu_idx])*1e-9, (timestamps[idx]-depth_timestamp[depth_idx])*1e-9])
