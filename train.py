import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from models import *


def create_batches(data: torch.Tensor, size: int):
    X, y = [], []
    for i in range(len(data) - size):
        X.append(data[i : i + size - 1])
        y.append(data[i + size][2:])

    shapes = set(np.array(x).shape for x in X)
    print("Unique X shapes:", shapes)
    y_shapes = set(np.array(yy).shape for yy in y)
    print("Unique y shapes:", y_shapes)

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def normalize(data: torch.Tensor, col_indices: list):
    min_vals = data[:, col_indices].min(dim=0).values
    max_vals = data[:, col_indices].max(dim=0).values
    data[:, col_indices] = (data[:, col_indices] - min_vals) / (max_vals - min_vals)
    return data, min_vals, max_vals


path = "data/processed/train_data.csv"
df = pd.read_csv(path)

cols = [
    "month_sin",
    "month_cos",
    "housing_new_construction",
    "housing_extensions",
    "non_residential_icef",
    "non_residential_services",
]
data = torch.tensor(df[cols].values, dtype=torch.float32)

norm_cols = [2, 3, 4, 5]
scaler = MinMaxScaler()

ratio = 0.8
batch_size = 5
train_data = data[: int(ratio * len(data))]
test_data = data[int(ratio * len(data)) :]

train_data, train_max, train_min = normalize(train_data, norm_cols)
test_data, test_max, test_min = normalize(test_data, norm_cols)


X_train, y_train = create_batches(train_data, batch_size)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = create_batches(test_data, batch_size)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


model = ForecastModel()
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    train_loss = criterion(output, y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss.item():.6f} | "
            f"Test Loss: {test_loss.item():.6f}"
        )
