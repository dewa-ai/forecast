import pandas as pd

IN_CSV = "data/usd_idr_raw.csv"
OUT_CSV = "data/usd_idr.csv"

df = pd.read_csv(IN_CSV)

# Rename kolom
df = df.rename(columns={
    "Date": "date",
    "Price": "value"
})

# Parse date (MM/DD/YYYY → YYYY-MM-DD)
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

# Clean numeric values: "16,090.00" → 16090.00
df["value"] = (
    df["value"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Sort ascending by date (WAJIB untuk time series)
df = df.sort_values("date")

# Simpan CSV final
df[["date", "value"]].to_csv(OUT_CSV, index=False)

print(f"Saved clean dataset to {OUT_CSV}")
print(df.head())

