import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.subplots
import xgboost as xgb
from plotly.subplots import make_subplots
from prophet.serialize import model_from_json
from statsmodels.tsa.seasonal import STL


def lagged_coorelations(df, feature_cols, target_cols, max_lag=10, diff=True):
    results = {}
    for target in target_cols:
        results[target] = {}
        for feature in feature_cols:
            corrs = []
            for lag in range(max_lag + 1):
                a = (
                    df[target].diff(1 if feature != target else 12)
                    if diff
                    else df[target]
                )
                corr = a.corr(
                    df[feature].diff(1 if feature != target else 12).shift(lag)
                    if diff
                    else df[feature].shift(lag)
                )
                corrs.append(corr)
            results[target][feature] = corrs
    return results


def heatmap(coor_dict, max_lag, target_cols, feature_cols):
    fig = make_subplots(rows=len(target_cols) + 1, cols=1, subplot_titles=target_cols)

    for i, target in enumerate(target_cols):
        z = [coor_dict[target][feature] for feature in feature_cols]
        fig.add_trace(
            go.Heatmap(
                z=np.abs(z),
                x=[f"lag {l}" for l in range(max_lag + 1)],
                y=[feature if feature != target else None for feature in feature_cols],
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
            ),
            row=i + 1,
            col=1,
        )
    fig.add_trace(
        go.Heatmap(
            z=[coor_dict[target][target] for target in target_cols],
            x=[f"lag {l}" for l in range(max_lag + 1)],
            y=target_cols,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
        ),
        row=len(target_cols) + 1,
        col=1,
    )

    fig.update_layout(title="Lagged Coorelations", height=300 * len(target_cols))
    fig.show()


def bar(corr_dict, max_lag, target, feature):
    fig = go.Figure(
        [
            go.Bar(
                x=[f"lag {l}" for l in range(max_lag + 1)], y=corr_dict[target][feature]
            )
        ]
    )
    fig.update_yaxes(
        range=[-1, 1], zeroline=True, zerolinewidth=2, zerolinecolor="black"
    )
    fig.update_layout(title=f"{feature} vs {target} — Correlation by Lag")
    fig.show()


def stl(df, cols):
    n_targets = len(cols)
    rows_per_group = 5
    n_rows = rows_per_group * n_targets

    titles = [
        item for col in cols for item in (col, "Trend", "Seasonal", "Residual", "")
    ]

    fig = plotly.subplots.make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        row_heights=[1, 1, 1, 1, 0.3] * n_targets,
        vertical_spacing=0.3 / n_rows,
    )
    stls = {}
    for i, row in enumerate(cols):
        result = STL(df[row], seasonal=13, period=12).fit()
        stls[row] = {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
        fig.add_trace(
            go.Scatter(x=df.index, y=result.observed, mode="lines"),
            row=rows_per_group * i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=result.trend, mode="lines"),
            row=rows_per_group * i + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=result.seasonal, mode="lines"),
            row=rows_per_group * i + 3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=result.resid, mode="lines"),
            row=rows_per_group * i + 4,
            col=1,
        )

    fig.update_layout(title="STL Decomposition", height=220 * n_rows, showlegend=False)
    fig.show()
    return stls


def run_analysis():
    df = pd.read_csv("data/processed/train_data.csv")
    df = df.set_index("date")
    df.index.freq = "MS"

    feature_cols = [
        "housing_new_construction",
        "housing_extensions",
        "non_residential_icef",
        "non_residential_services",
        "interest_rate",
        "currency_exchange_rate",
        "economic_perception_index",
        "economic_activity_indicator",
    ]
    target_cols = [
        "housing_new_construction",
        "housing_extensions",
        "non_residential_icef",
        "non_residential_services",
    ]

    xgb_models = {}
    for target in target_cols:
        model = xgb.XGBRegressor()
        model.load_model(f"models/xgb_{target}.json")
        xgb_models[target] = model

    prophet_models = {}
    for target in target_cols:
        with open(f"models/prophet_{target}.json", "r") as f:
            prophet_models[target] = model_from_json(f.read())

    max_lag = 12
    coor_dict = lagged_coorelations(df, feature_cols, target_cols, max_lag)

    heatmap(coor_dict, max_lag, target_cols, feature_cols)
    # bar(coor_dict, max_lag, target="non_residential_services", feature="non_residential_services")
    stls = stl(df, cols=feature_cols)
    resids = {row: stls[row]["resid"] for row in stls}
    coor_resid = lagged_coorelations(
        resids, feature_cols, target_cols, max_lag, diff=False
    )
    heatmap(coor_resid, max_lag, target_cols, feature_cols)
    print(coor_resid)
