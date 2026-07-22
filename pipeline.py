import pandas as pd
import csv
import numpy as np
import datetime

MONTHS = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
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
        print(e, ":", value)
        return (np.nan, np.nan)


path = "data/raw/download.xls"
df = pd.read_excel(path, sheet_name=2, header=4, usecols="B,D:G")
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

df.columns = [
    "month_year",
    "housing_new_construction",
    "housing_extensions",
    "non_residential_icef",
    "non_residential_services",
]

cols = [
    "housing_new_construction",
    "housing_extensions",
    "non_residential_icef",
    "non_residential_services",
]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df[["month_sin", "month_cos"]] = df["month_year"].apply(
    lambda v: pd.Series(convert_date(v))
)
df = df.drop(columns=["month_year"])

df.to_csv("data/processed/train_data.csv", index=False)
print(df)
