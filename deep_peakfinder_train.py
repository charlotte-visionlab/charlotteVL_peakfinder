from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary
import os

from deep_peakfinder_model import ResNet1D, PeakDetectionNet, PeakDetectionXformNet
from deep_peakfinder_loss import IoULoss, SparsePeakDetectionLoss, SparsePeakLoss, WeightedIoULoss
from deep_peakfinder_csv2h5 import PeaksDataset
from sklearn.metrics import f1_score


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


if __name__ == "__main__":
    # dataset_filename = "/home/arwillis/AF241/af241_deep_doa/IQ-2.h5"
    dataset_filename = "IQ_synthetic_v04.h5"

    batch_size = 20000  # 65535 ==> 1128 MB GPU memory
    num_epochs = 2000

    # Hyperparameters to tune
    learning_rate = 1.0e-3
    weight_decay = 1.0e-5
    epsilon = 1.0e-6

    log_dir = "runs_peaks"
    save_dir = "weights_peaks"

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    correlation_matrix_dataset = PeaksDataset(dataset_filename)

    # Create DataLoaders
    train_ratio, val_ratio, test_ratio = (0.7, 0.2, 0.1)
    dataloaders = create_dataloaders(correlation_matrix_dataset,
                                     train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                                     batch_size=batch_size)

    # Iterate over DataLoaders
    # for phase in ["train", "validate", "test"]:
    #     print(f"Processing {phase} dataset:")
    #     for batch in dataloaders[phase]:
    #         batch_corr_matrices, batch_theta_music, batch_theta_true = batch
    #         # corr_matrix_real = batch["correlation_matrices_real"]
    #         # corr_matrix_imag = batch["correlation_matrices_imag"]
    #         # theta_music = batch["theta"]
    #         # theta_true = batch["theta_true"]
    #         print(f"  Batch size: {batch_corr_matrices.size(0)}")
    #         break  # Process only the first batch for demonstration

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validate"]
    test_loader = dataloaders["test"]

    # Set up TensorBoard writer
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs_peaks/training_logs_{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    input_size = (6, 400)

    # Example Configuration
    block_configs = [
        {"in_channels": 6, "out_channels": 64, "kernel_size": 5, "stride": 1},
        {"in_channels": 64, "out_channels": 256, "kernel_size": 3, "stride": 1},
        {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 1},
        {"in_channels": 512, "out_channels": 1024, "kernel_size": 3, "stride": 1},
        {"in_channels": 1024, "out_channels": 512, "kernel_size": 3, "stride": 1},
        {"in_channels": 512, "out_channels": 128, "kernel_size": 3, "stride": 1},
        {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "stride": 1},
    ]
    # Instantiate model, optimizer, and loss function
    # num_peaks = 2
    # model = ResNet1D(input_channels=(2, 400), output_size=num_peaks, block_configs=block_configs)
    # loss_function = nn.MSELoss().to(device)

    # model = PeakDetectionNet()
    model = PeakDetectionXformNet(input_channels=6, output_length=400, num_classes=1)
    # model = nn.DataParallel(model)
    # loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 400.0])) .to(device) # Higher weight for class 1
    # 01-09_19_33
    # loss_function = SparsePeakDetectionLoss(alpha=1.0, gamma=2.0).to(device)
    # loss_function = SparsePeakLoss().to(device)
    loss_function = WeightedIoULoss(pos_weight=390.0,neg_weight=1.0).to(device)
    # loss_function = IoULoss().to(device)
    # loss_function = nn.BCELoss(weight=torch.tensor(394.0 / 6.0)).to(device)  # For binary classification
    # loss_function = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    print(model)
    # summary(model, input_size=input_size)

    layout = {
        "PeakResNet1D": {
            "iteration loss": ["Multiline", ["loss/train", "loss/validation"]],
            "epoch loss": ["Multiline", ["epoch loss/train", "epoch loss/validation"]],
            "learning rate": ["Multiline", ["learning rate/lr"]]
            # "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
        },
    }

    writer = SummaryWriter(log_dir)
    writer.add_custom_scalars(layout)

    # Training loop
    global_train_batch_index = 0
    global_val_batch_index = 0

    for epoch in range(num_epochs):

        # Training
        model.train()
        epoch_loss = 0.0
        # epoch_mpe = 0
        # epoch_mrpd = 0
        for batch_data in train_loader:
            batch_signal_fft_vals, batch_peak_true_vals = batch_data
            batch_X = batch_signal_fft_vals.to(device)
            batch_y_true = batch_peak_true_vals.to(device)
            batch_y_pred = model(batch_X)
            # predicted_classes = torch.round(batch_y_pred[:, 1, :]).unsqueeze(1)
            # predicted_classes = torch.argmax(probabilities, dim=1)
            # predicted_classes = torch.argmax(probabilities, dim=1)
            # print("Softmax Output Shape:", softmax_output.shape)
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss = loss_function(batch_y_pred, batch_y_true)
            train_loss.backward()
            optimizer.step()

            # true_peak_indices = np.where(batch_y_true.detach().cpu().numpy() == 1)
            # predicted_peak_indices = np.where(predicted_classes.detach().cpu().numpy() == 1)
            # print(f"train predicted indices shape = {(predicted_peak_indices)}"
            #       + f" true indices shape = {(true_peak_indices)}")

            # Compute Mean Percent Error (MPE)
            # mpe = torch.mean(torch.abs(outputs - batch_y) / (batch_y + epsilon)) * 100
            # Mean Relative Percent Difference https://en.wikipedia.org/wiki/Relative_change
            # mrpd = compute_mprd(outputs, batch_y)
            # Log batch metrics to TensorBoard
            writer.add_scalar("loss/train", train_loss, global_train_batch_index)
            writer.add_scalar("learning rate/lr", optimizer.param_groups[0]["lr"], global_train_batch_index)
            # F1 Score
            # batch_y_pred = batch_y_pred.detach().cpu().numpy()
            # batch_y_true = batch_y_true.detach().cpu().numpy()
            # f1 = f1_score(batch_y_true, batch_y_pred)
            # print("F1 Score:", f1)
            # Accuracy
            # accuracy = (batch_y_true == batch_y_pred).float().mean()
            # print("Accuracy:", accuracy.item())
            epoch_loss += train_loss.item()
            # epoch_mpe += loss.item()
            # epoch_mrpd += mrpd.item()
            global_train_batch_index += 1

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        # avg_epoch_mpe = epoch_mpe / len(train_loader)
        # avg_epoch_mrpd = epoch_mrpd / len(train_loader)
        # TensorBoard logging
        writer.add_scalar("epoch loss/train", avg_epoch_loss, epoch + 1)
        # writer.add_scalar("Epoch MPE/Validation", avg_val_mpe, epoch + 1)
        # writer.add_scalar("Epoch MRPD/Train", avg_epoch_mrpd, epoch + 1)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_epoch_loss:.4f},  MRPD: {avg_epoch_mrpd:.2f}%")

        # Validation
        model.eval()
        dataset_validation_loss = 0.0
        # val_mpe = 0.0
        # val_mrpd = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_signal_fft_vals, batch_peak_true_vals = batch_data
                batch_X = batch_signal_fft_vals.to(device)
                batch_y_true = batch_peak_true_vals.to(device)
                batch_y_pred = model(batch_X)
                # predicted_classes = torch.round(batch_y_pred[:, 1, :]).unsqueeze(1)
                # true_peak_indices = np.where(batch_y_true.cpu().numpy() == 1)
                # predicted_peak_indices = np.where(predicted_classes.cpu().numpy() == 1)
                # print(f"val predicted indices shape = {predicted_peak_indices.shape}"
                #       + f" true indices shape = {true_peak_indices.shape}")
                validation_loss = loss_function(batch_y_pred, batch_y_true).item()
                writer.add_scalar("loss/validation", validation_loss, int(global_val_batch_index))
                dataset_validation_loss += validation_loss
                # Compute Mean Percent Error (MPE)
                # mpe = torch.mean(torch.abs(outputs - batch_y) / (batch_y + epsilon)) * 100
                # Mean Relative Percent Difference https://en.wikipedia.org/wiki/Relative_change
                # mrpd = compute_mprd(outputs, batch_y)
                # val_mpe += mpe.item()
                # val_mrpd += mrpd.item()
                global_val_batch_index += train_ratio / val_ratio

        avg_val_loss = dataset_validation_loss / len(test_loader)
        # avg_val_mpe = val_mpe / len(test_loader)
        # avg_val_mrpd = val_mrpd / len(test_loader)
        # TensorBoard logging
        writer.add_scalar("epoch loss/validation", avg_val_loss, epoch + 1)
        # writer.add_scalar("Epoch MPE/Validation", avg_val_mpe, epoch + 1)
        # writer.add_scalar("Epoch MRPD/Validation", avg_val_mrpd, epoch + 1)
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f},  MRPD: {avg_val_mrpd:.2f}%")

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, ")
        # f"Train MRPD: {avg_epoch_mrpd:.4f}% Val MRPD: {avg_val_mrpd:.4f}%")

        # Save weights every 100 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint_path = os.path.join(save_dir, f"model_{current_time}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # TODO: Add cost to perform inference on the test dataset

    # Save final weights
    final_path = os.path.join(save_dir, f"model_{current_time}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved at {final_path}")
    writer.close()
