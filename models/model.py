import torch
import torch.nn as nn
import torch.optim as optim
import plotly

class ForecastModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, dropout=0.5, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4 * horizon)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        x = self.leaky_relu(self.fc1(last_hidden))
        self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        self.dropout(x)
        x = self.fc3(x)
        return x