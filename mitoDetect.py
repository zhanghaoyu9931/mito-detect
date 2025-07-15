import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from copy import deepcopy

import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Uncomment this line to use the corresponding GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

### Functions ###
def gaussian_kernel(size, sigma):
    """Generate a 1D Gaussian kernel.

    Args:
        size (int): The size of the kernel (odd number).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: 1D Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)  # Normalize the kernel to sum to 1.
    return kernel

def gaussian_filter(signal, kernel_size, sigma):
    """Apply Gaussian filtering to a 1D signal.

    Args:
        signal (numpy.ndarray): 1D input signal.
        kernel_size (int): The size of the Gaussian kernel (odd number).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: Filtered signal.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    filtered_signal = np.convolve(signal, kernel, mode='same')
    return filtered_signal

# Gaussian filter parameters
kernel_size = 21  # Choose an odd kernel size (e.g., 3, 5, 7, ...)
sigma = 5      # Standard deviation for the Gaussian kernel

def load_xy_file_out(dir_nm):
    """Load test data without labels for inference.
    
    Args:
        dir_nm (str): Directory name containing the test data file.
        
    Returns:
        numpy.ndarray: Preprocessed test features.
    """
    train_x = scio.loadmat(f'{dir_nm}/Fall.mat')
    F_train = train_x['F']
    x_train = np.zeros(shape=(F_train.shape[0], 3000)) # default: 3000 timepoints
    # x_train = np.zeros(shape=F_train.shape)
    
    ## Data preprocessing
    for i in range(F_train.shape[0]):
        sig_i = F_train[i, :]
        filtered_signal = gaussian_filter(sig_i, kernel_size, sigma)
        # filtered_signal[0: 10] = filtered_signal[10]
        # filtered_signal[-10:] = filtered_signal[-10]
        
        # Dataset normalization
        if np.std(filtered_signal) == 0:
            filtered_signal = np.zeros(filtered_signal.shape) + 0.001
        
        else:
            filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        
        # Data truncation to 3000 timepoints
        filter_num = filtered_signal.shape[0] - 3000
        filtered_signal = filtered_signal[filter_num: ]

        x_train[i, :] = filtered_signal
    
    return x_train

# Build CNN-LSTM model
class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for mitochondrial signal classification.
    
    This model combines convolutional layers for feature extraction with LSTM layers
    for temporal sequence modeling to classify mitochondrial calcium signals.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate, seq_len_afterTrans = 197):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, stride=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 1, 1024), #seq_len_afterTrans
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(2)  # Add dimension to adapt to CNN input requirements
        x = x.permute(0, 2, 1)  # Transform input shape to adapt to CNN input requirements
        out = self.cnn(x)
        out = out.permute(0, 2, 1)  # Transform back to LSTM input shape
        out, _ = self.lstm(out)
        # out = self.transformer(out)
        
        out = self.fc(out[:, -1, :]).squeeze(1)  # Take the last time step output from LSTM as prediction
        # out = self.fc(out.reshape((out.shape[0], -1))).squeeze(1)  # Take the last time step output from LSTM as prediction
        return out

### Main ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--visual_num', type=int, default=3)
    args = parser.parse_args()

    # Get the arguments
    data_dir = args.data_dir
    visual_num = args.visual_num

    # Load the data
    x_test = load_xy_file_out(data_dir)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    # Define the model
    input_size = 1
    hidden_size = 128
    num_layers = 3
    num_classes = 2
    dropout_rate = 0.2  # Add Dropout to prevent overfitting

    # Load the model
    model = CNNLSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)
    model.load_state_dict(torch.load(
        'model_weights.pth', map_location=torch.device('cpu')))
    model = model.to(device)

    # Prepare the data loader
    y_test = np.zeros(x_test.shape[0])
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test)) # TensorDataset(torch.tensor(x_test), torch.tensor(y_test)) # TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Output the results
    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = (outputs > 0.95).int()
            val_predictions.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            
    # Save the results
    root_dir = f'{data_dir}/'
    os.makedirs(root_dir, exist_ok=True)
    checked_fig_dir = f'{data_dir}/checked_figs'
    os.makedirs(checked_fig_dir, exist_ok=True)
    print(f'File dir: {data_dir}\n')

    # Convert predictions to label format (0 for non-mitochondrial, 2 for mitochondrial)
    val_predictions = [x * 2 for x in val_predictions]
    scio.savemat(f'{root_dir}/Label.mat', {'predicted_label': val_predictions})
    print('The first 10 predictions: ', val_predictions[: 10])

    # Save the sample visualization plots
    fig_nm = 0
    print('The total number of signals: ', len(val_predictions), 'The number of mitochondrial signals: ', np.sum(val_predictions)/ 2)

    # Generate sample plots for detected mitochondrial signals
    for i in range(min(visual_num, len(val_predictions))):
        
        if val_predictions[i] == 2:
            fig_nm += 1
            plt.figure(figsize=(20, 5))
            plt.plot(x_test[i, :])
            plt.title(f'Mitochondrial signal {fig_nm}, original idx: {i}')
            plt.savefig(f'{checked_fig_dir}/{i}.png', dpi = 300)
            plt.close()