import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def build_features_for_target(df, target, lag_dict):
    feature_df = pd.DataFrame(index=df.index)
    for feature, lag in lag_dict[target].items():
        col_name = f"{feature}_lag{lag}" if lag > 0 else feature
        feature_df[col_name] = df[feature].shift(lag)
    return feature_df


def train_xgb():

    df = pd.read_csv("data/processed/train_data.csv")

    target_cols = [
        "housing_new_construction",
        "housing_extensions",
        "non_residential_icef",
        "non_residential_services",
    ]
    feature_cols = [
        "interest_rate",
        "currency_exchange_rate",
        "economic_perception_index",
        "economic_activity_indicator",
    ]

    with open("models/lags.json", "r") as f:
        lags = json.load(f)["template 1"]

    target_feature_sets = {}
    for target in target_cols:
        target_feature_sets[target] = build_features_for_target(df, target, lags)

    df = df.dropna().reset_index(drop=True)

    ratio = 0.8

    xgb_models = {}
    xgb_results = {}
    xgb_preds = {}

    target_train_test = {}

    for target in target_cols:
        features = target_feature_sets[target]
        combined = features.copy()
        combined[target] = df[target]
        combined["date"] = df["date"]
        combined = combined.dropna().reset_index(drop=True)

        split = int(ratio * len(combined))
        train_df = combined.iloc[:split]
        test_df = combined.iloc[split:]

        target_train_test[target] = (train_df, test_df)

        X_cols = list(features.columns)

        model = xgb.XGBRegressor(
            n_estimators=128,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=109234231,
        )
        model.fit(train_df[X_cols], train_df[target])
        preds = model.predict(test_df[X_cols])

        xgb_models[target] = model
        xgb_preds[target] = preds
        xgb_results[target] = mean_squared_error(test_df[target], preds)
        print(f"{target} MSE: {xgb_results[target]:.2f}")

    fig = plotly.subplots.make_subplots(rows=4, cols=1, subplot_titles=target_cols)
    window_size = 40

    for i, target in enumerate(target_cols):
        train_df, test_df = target_train_test[target]
        preds = xgb_preds[target]

        history_dates = train_df["date"].iloc[-window_size:]
        history_actual = train_df[target].iloc[-window_size:]
        test_dates = test_df["date"]
        test_actual = test_df[target]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([history_dates, test_dates]),
                y=pd.concat([history_actual, test_actual]),
                mode="lines",
                name=f"{target} (actual)",
                line=dict(color="blue"),
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=preds,
                mode="lines+markers",
                name=f"{target} (predicted)",
                line=dict(color="red", dash="dash"),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(title="XGBoost Forecast", height=900)
    fig.show()

    os.makedirs("models", exist_ok=True)
    for target in target_cols:
        xgb_models[target].save_model(f"models/xgb_{target}.json")
