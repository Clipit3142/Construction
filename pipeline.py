import pandas as pd
import csv

MONTHS = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "may": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12"
}

def convert_date(value):
    if pd.isna(value):
        return pd.NaT
    try:
        month_str, year_str = value.split("-")
        month = MONTHS[month_str]
        year = int(year_str)
        year = 1900 + year if year > 90 else 2000 + year
        print(year)
        return pd.to_datetime(f"{year}-{month}-01")
    except Exception as e:
        print(e)
        return pd.NaT

url = "data/raw/series-históricas-a-mayo-2026.xls"
df = pd.read_excel(url, sheet_name=2, header=4, usecols="B,D:G")
df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")
df.columns = [
    "month_year",
    "housing_new_construction",
    "housing_extensions",
    "non_residential_icef",
    "non_residential_services",
]

numeric_cols = ["housing_new_construction", "housing_extensions", "non_residential_icef", "non_residential_services"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df["month_year"] = df["month_year"].apply(convert_date)

df.to_csv("data/processed/train_data.csv", index=False)

print(df.head(12))


