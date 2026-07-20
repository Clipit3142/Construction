import pandas as pd
import csv
import numpy as np
import datetime

MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4,
    "may": 5, "jun": 6, "jul": 7, "ago": 8,
    "sep": 9, "oct": 10, "nov": 11, "dic": 12
}

def convert_date(value):
    if pd.isna(value):
        return (np.nan, np.nan)
    if type(value) == datetime.datetime:
        month = value.month
        month_sin = np.sin(np.pi * month / 6)
        month_cos = np.cos(np.pi * month / 6)
        return (month_sin, month_cos)
    try:
        month_str, _ = value.split("-")
        month = MONTHS[month_str]
        month_sin = np.sin(np.pi * int(month) / 6)
        month_cos = np.cos(np.pi * int(month) / 6)
        return (month_sin, month_cos)
    except Exception as e:
        print(e,":",value)
        return (np.nan, np.nan)

df = pd.read_excel("data/raw/download.xls", sheet_name=2, header=138, usecols="B,D:G")
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

df.columns = [
    "month_year",
    "housing_new_construction",
    "housing_extensions",
    "non_residential_icef",
    "non_residential_services"
    ]

extra = [
    "interest_rate",
    "currency_exchange_rate",
    "economic_perception_index",
    "economic_activity_indicator"
    ]



df["interest_rate"] = pd.read_excel("data/raw/interest_rates.xlsx", sheet_name="Chart", header=76, usecols="B")
df["currency_exchange_rate"] = pd.read_excel("data/raw/exchange_rates.xlsx", sheet_name="Chart", header=508, usecols="B")
df["economic_perception_index"] = pd.read_excel("data/raw/IPEC.xlsx", sheet_name="Chart", header=2, usecols="B")
df["economic_activity_indicator"] = pd.read_excel("data/raw/economic_activity.xlsx", sheet_name="Chart", header=88, usecols="B")

cols = ["housing_new_construction", "housing_extensions", "non_residential_icef", "non_residential_services"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
#df[["month_sin", "month_cos"]] = df["month_year"].apply(lambda v: pd.Series(convert_date(v)))

df["date"] = pd.date_range(start="2002-03-01", periods=len(df), freq="MS")
df = df.drop(columns=["month_year"])

df = df.dropna(how="any")

df.to_csv("data/processed/train_data.csv", index=False)



print(df)