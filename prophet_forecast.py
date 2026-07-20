import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.subplots
import plotly.graph_objects as go
import random
import os
from sklearn.metrics import mean_squared_error
from prophet.serialize import model_to_json

path = "data/processed/train_data.csv"
df = pd.read_csv(path)

target_cols = ["housing_new_construction", "housing_extensions", "non_residential_icef", "non_residential_services"]
feature_cols = ["interest_rate", "currency_exchange_rate", "economic_perception_index", "economic_activity_indicator"]

df["date"] = pd.date_range(start="2002-3-01", periods=len(df), freq="MS")


ratio = 0.8
split = int(ratio * len(df))
train_df = df.iloc[:split]
test_df = df.iloc[split:]

results = {}
forecasts = {}
models = {}

for target in target_cols:
    model = Prophet(
        changepoint_prior_scale=1,
        seasonality_prior_scale=12,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for reg in feature_cols:
        model.add_regressor(reg)

    prophet_train = train_df[["date", target] + feature_cols].rename(columns={"date": "ds", target: "y"})
    model.fit(prophet_train)

    prophet_test = test_df[["date"] + feature_cols].rename(columns={"date": "ds"})
    forecast = model.predict(prophet_test)

    forecasts[target] = forecast
    models[target] = model 
    results[target] = mean_squared_error(test_df[target].array, forecast["yhat"].array)
    print(f"{target} MAE: {results[target]:.2f}")

fig = plotly.subplots.make_subplots(rows=4, cols=1, subplot_titles=target_cols)

window_size = 20
start = random.randint(0, len(test_df) - window_size)

for i, target in enumerate(target_cols):
    forecast = forecasts[target]

    history_dates = train_df["date"].iloc[-window_size:]
    history_actual = train_df[target].iloc[-window_size:]
    test_dates = test_df["date"]
    test_actual = test_df[target]

    fig.add_trace(
        go.Scatter(x=pd.concat([history_dates, test_dates]),
                   y=pd.concat([history_actual, test_actual]),
                   mode="lines", name=f"{target} (actual)", line=dict(color="blue")),
        row=i+1, col=1
    )

    fig.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                   mode="lines+markers", name=f"{target} (predicted)", line=dict(color="red", dash="dash")),
        row=i+1, col=1
    )

    fig.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                   mode="lines", line=dict(width=0), showlegend=False),
        row=i+1, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                   mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(255,0,0,0.1)", showlegend=False),
        row=i+1, col=1
    )

fig.update_layout(title="Prophet Forecast", height=900)
fig.show()

os.makedirs("models", exist_ok=True)

for target in target_cols:
    with open(f"models/prophet_{target}.json", "w") as f:
        f.write(model_to_json(models[target]))