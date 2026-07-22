import torch.nn as nn


class ForecastModel(nn.Module):
    def __init__(
        self, input_size=6, hidden_size=256, num_layers=6, num_month_features=12
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        x = self.relu(self.fc1(last_hidden))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_size=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.relu(self.fc3(x))
