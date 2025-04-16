import os
import pandas as pd
import csv
import numpy as np

import re

def use_regex(input_text):
    pattern = re.compile(r"depth_([0-9]+)\.tiff", re.IGNORECASE)
    return pattern.findall(input_text)

def main(folder_path):
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
        # print(images)
        temp = use_regex(images)
        # print(images,int(temp[0]))
        depth_timestamp.append(int(temp[0]))
    depth_timestamp = sorted(depth_timestamp)
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

# folder_path = "./datasets/02-07-2025_15-06-16-737244/"
folder_names = ["02-05-2025_15-18-32-003539", "02-05-2025_16-17-48-719931", "02-07-2025_10-34-15-586727", "02-07-2025_11-03-23-147284", "02-07-2025_11-16-03-585918", "02-07-2025_11-29-43-359675", "02-07-2025_11-53-49-554597", "02-07-2025_13-04-56-680734", "02-07-2025_13-20-12-624401", "02-07-2025_13-36-06-814749", "02-07-2025_15-20-45-566250", "02-07-2025_15-35-54-028218"]
for folder_name in folder_names:
    folder_path = "./datasets/" + folder_name + "/"
    main(folder_path)
