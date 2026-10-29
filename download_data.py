import yfinance as yf

df = yf.download(
    "AAPL",
    start="2025-10-10",
    end="2026-10-10",
    interval="1d",
    auto_adjust=False,
    multi_level_index=False,
    progress=False
)

if df.empty:
    raise RuntimeError("No data received. Try again later.")

df = df.reset_index()
df = df[
    ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
]
df = df.sort_values("Date")

df.to_csv("apple_stock.csv", index=False)

print(f"Downloaded {len(df)} trading sessions.")
print(df.head())