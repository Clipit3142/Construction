import os
import random
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.subplots
import torch
import torch.nn as nn
import torch.optim as optim
from plotly import graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import ForecastModel


def moving_average(data: torch.Tensor, window: int, cols: list):
    ma = data.clone()
    for i in range(len(data)):
        start = max(0, i - window + 1)
        ma[i, cols] = data[start : i + 1, cols].mean(dim=0)
    return ma


def create_batches(data: torch.Tensor, size: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - size - horizon):
        X.append(data[i : i + size])
        y.append(data[i + size : i + size + horizon].flatten())
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def normalize(data: torch.Tensor, col_indices: list, min_vals=None, max_vals=None):
    if min_vals is None or max_vals is None:
        min_vals = data[:, col_indices].min(dim=0).values
        max_vals = data[:, col_indices].max(dim=0).values
    data = data.clone()
    data[:, col_indices] = (data[:, col_indices] - min_vals) / (max_vals - min_vals)
    return data, min_vals, max_vals


def train_lstm():

    path = "data/processed/train_data.csv"
    df = pd.read_csv(path)

    cols = [
        "housing_new_construction",
        "housing_extensions",
        "non_residential_icef",
        "non_residential_services",
    ]
    data = torch.tensor(df[cols].values, dtype=torch.float32)

    norm_cols = [0, 1, 2, 3]

    ratio = 0.8
    horizon = 2
    window_size = 20
    train_data = data[: int(ratio * len(data))]
    test_data = data[int(ratio * len(data)) :]

    train_data, train_min, train_max = normalize(train_data, norm_cols)
    test_data, _, _ = normalize(
        test_data, norm_cols, min_vals=train_min, max_vals=train_max
    )

    X_train, y_train = create_batches(train_data, window_size, horizon)
    X_test, y_test = create_batches(test_data, window_size, horizon)

    model = ForecastModel(dropout=0.5, horizon=horizon)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    epochs = 500

    train_losses = []
    test_losses = []

    for epoch in range(epochs + 1):
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

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Test Loss: {test_loss.item():.6f}"
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(y=test_losses, mode="lines", name="Test Loss"))
    fig.update_layout(
        title="Training vs Test Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        template="plotly_dark",
    )

    fig.show()

    test_start = random.randint(0, len(test_data) - window_size - horizon)
    test = test_data[test_start : test_start + window_size + horizon]
    test_input = test[:window_size]
    test_target = test[window_size:]

    model.eval()
    with torch.no_grad():
        test_output = model(test_input.unsqueeze(0)).squeeze(0).view(horizon, 4)

    fig = plotly.subplots.make_subplots(rows=4, cols=1, subplot_titles=cols)

    x_input = list(range(window_size + horizon))
    x_future = list(range(window_size, window_size + horizon))

    for i, name in enumerate(cols):
        full_series = (
            test[:, i] * (train_max[i] - train_min[i]) + train_min[i]
        ).numpy()
        pred_series = (
            test_output[:, i] * (train_max[i] - train_min[i]) + train_min[i]
        ).numpy()

        fig.add_trace(
            go.Scatter(
                x=x_input,
                y=full_series,
                mode="lines",
                name="Full",
                line=dict(color="gray"),
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_future,
                y=pred_series,
                mode="lines",
                name="Pred",
                line=dict(color="blue"),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(title="Test Forecast")
    fig.show()
