import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        # Define the LSTM architecture
        self.lstm = nn.LSTM(input_dim, hidden_dims, num_layers=num_layers,
                            dropout=dropout_prob, batch_first=True)
        self.output_layer = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        # Pass the input through the LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last hidden state for the last time step
        last_time_step = lstm_out[:, -1, :]
        # Pass the last hidden state of the last time step through the output layer
        x = self.output_layer(last_time_step)
        return x