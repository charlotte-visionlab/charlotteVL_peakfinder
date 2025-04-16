from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary
import os
import optuna

from avl_models import SignalToImageCNN, ComplexSignalToImageCNN, ResNetSignalToImage, XformNetSignalToImage
from avl_urad_create_h5 import DepthDataset
from sklearn.metrics import f1_score
from ssim import SSIM, ssim

def ssim_loss(X,Y):
    loss_func = SSIM(data_range=1., channel=1)
    return 1 - loss_func(X,Y)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.ssim = ssim_loss
        self.l1 = nn.SmoothL1Loss()
        self.alpha = alpha
    def forward(self, img1, img2):
        return self.alpha * self.ssim(img1, img2) + (1 - self.alpha) * self.l1(img1, img2)

def create_dataloaders(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, batch_size=32, shuffle=True):
    """
    Create DataLoaders for training, validation, and testing using a single dataset.

    Parameters:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        batch_size (int): Batch size for the DataLoaders.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        dict: A dictionary containing the DataLoaders for 'train', 'validate', and 'test'.
    """
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return {"train": train_loader, "validate": val_loader, "test": test_loader}


dataset_filename = "training_dataset.h5"
def objective(trial):
    batch_size = trial.suggest_int("batch_size", 32, 1024, step=32)#128 #5096  # 5096 ==> 13898 MB GPU memory
    num_epochs = 50

    # Hyperparameters to tune
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2)#1.0e-3

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    correlation_matrix_dataset = DepthDataset(dataset_filename)

    # Create DataLoaders
    train_ratio, val_ratio, test_ratio = (0.7, 0.2, 0.1)
    dataloaders = create_dataloaders(correlation_matrix_dataset,
                                        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                                        batch_size=batch_size)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validate"]
    test_loader = dataloaders["test"]

    model = ResNetSignalToImage()
    # model = XformNetSignalToImage()
    # loss_function = nn.MSELoss()
    # loss_function = torch.nn.SmoothL1Loss().to(device)
    loss_function = ssim_loss
    # loss_function = CombinedLoss().to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    # Training loop
    global_train_batch_index = 0
    global_val_batch_index = 0

    for epoch in range(num_epochs):

        # Training
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            batch_signal_fft_vals, batch_peak_true_vals = batch_data
            batch_X = batch_signal_fft_vals.to(device)
            batch_y_true = batch_peak_true_vals.to(device)
            batch_y_pred = model(batch_X)

            optimizer.zero_grad()
            batch_y_pred = batch_y_pred.reshape((-1,1,64,64))
            batch_y_true = batch_y_true.reshape((-1,1,64,64))
            # print(batch_y_pred.shape)
            train_loss = loss_function(batch_y_pred, batch_y_true)
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            global_train_batch_index += 1

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        dataset_validation_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_signal_fft_vals, batch_peak_true_vals = batch_data
                batch_X = batch_signal_fft_vals.to(device)
                batch_y_true = batch_peak_true_vals.to(device)
                batch_y_pred = model(batch_X)
                batch_y_pred = batch_y_pred.reshape((-1,1,64,64))
                batch_y_true = batch_y_true.reshape((-1,1,64,64))
                validation_loss = loss_function(batch_y_pred, batch_y_true).item()
                dataset_validation_loss += validation_loss
                global_val_batch_index += train_ratio / val_ratio

        avg_val_loss = dataset_validation_loss / len(test_loader)
    return avg_val_loss

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_value)
print(study.best_params)