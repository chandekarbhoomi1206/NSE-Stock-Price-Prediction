"""
NIFTY 50 DATASET DOWNLOADER — START-TO-FINISH VERSION

Downloads about five years of daily OHLCV + Adjusted Close data from Yahoo Finance
and saves one CSV per company, including the latest available financial metrics.

Install in VS Code terminal:
    python -m pip install yfinance pandas python-dateutil

Run:
    python nifty50_dataset.py

Output:
    Nifty50_Individual_CSVs/RELIANCE.csv
    Nifty50_Individual_CSVs/TCS.csv
    ...
    Nifty50_Individual_CSVs/download_summary.csv

Important:
- Yahoo Finance data is not official NSE exchange data and may be delayed/incomplete.
- Market_Cap, PE_Ratio, ROE, Debt_to_Equity are the latest available metrics,
  repeated across each historical date. They are NOT historical values for each date.
- Some metrics can be blank and some ticker downloads may fail.
- Nifty 50 constituents change over time; this fixed list is not historical index membership.
"""

from datetime import date
from pathlib import Path
import time

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

# Five years ending today; Yahoo's end date is exclusive.
TODAY = date.today()
START_DATE = (TODAY - relativedelta(years=5)).isoformat()
END_DATE = (TODAY + relativedelta(days=1)).isoformat()

OUTPUT_DIR = Path("Nifty50_Individual_CSVs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed basket of 50 NSE-listed symbols available on Yahoo Finance.
TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "BAJAJHLDNG.NS"
]

def fetch_company_metrics(ticker):
    """Fetch latest available company metrics. Missing values are left blank."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "Market_Cap": info.get("marketCap"),
            "PE_Ratio": info.get("trailingPE"),
            "ROE": info.get("returnOnEquity"),
            "Debt_to_Equity": info.get("debtToEquity"),
        }
    except Exception as exc:
        print(f"    Warning: metrics unavailable: {exc}")
        return {
            "Market_Cap": None,
            "PE_Ratio": None,
            "ROE": None,
            "Debt_to_Equity": None,
        }

def download_company(ticker):
    symbol = ticker.removesuffix(".NS")

    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError("No historical data returned by Yahoo Finance.")

    # Handle yfinance versions that return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    price_columns = required + (["Adj Close"] if "Adj Close" in data.columns else [])
    data = data[price_columns].copy()

    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.reset_index()
    data = data.rename(columns={data.columns[0]: "Date"})
    data["Symbol"] = symbol

    # Latest available snapshot; not a date-by-date historical financial series.
    metrics = fetch_company_metrics(ticker)
    for metric_name, metric_value in metrics.items():
        data[metric_name] = metric_value

    desired_columns = [
        "Date", "Symbol", "Open", "High", "Low", "Close", "Volume",
        "Adj Close", "Market_Cap", "PE_Ratio", "ROE", "Debt_to_Equity"
    ]
    data = data[[column for column in desired_columns if column in data.columns]]
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    output_file = OUTPUT_DIR / f"{symbol}.csv"
    data.to_csv(output_file, index=False)
    return len(data), data["Date"].min(), data["Date"].max(), output_file.name

def main():
    print("=" * 65)
    print("NIFTY 50 DATASET DOWNLOAD")
    print(f"Requested period: {START_DATE} to {TODAY.isoformat()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    summary_rows = []
    for index, ticker in enumerate(TICKERS, start=1):
        symbol = ticker.removesuffix(".NS")
        print(f"\n[{index}/{len(TICKERS)}] {symbol}")
        try:
            row_count, first_date, last_date, filename = download_company(ticker)
            summary_rows.append({
                "Symbol": symbol,
                "Status": "Success",
                "Rows": row_count,
                "First_Date": first_date,
                "Last_Date": last_date,
                "File": filename,
                "Error": "",
            })
            print(f"    Saved {filename} ({row_count} daily rows)")
        except Exception as exc:
            summary_rows.append({
                "Symbol": symbol,
                "Status": "Failed",
                "Rows": 0,
                "First_Date": "",
                "Last_Date": "",
                "File": "",
                "Error": str(exc),
            })
            print(f"    FAILED: {exc}")
        time.sleep(0.3)  # modest pause between requests

    summary_path = OUTPUT_DIR / "download_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    success_count = sum(row["Status"] == "Success" for row in summary_rows)
    print("\n" + "=" * 65)
    print("FINISHED")
    print(f"Successful company files: {success_count}/{len(TICKERS)}")
    print(f"Folder: {OUTPUT_DIR.resolve()}")
    print(f"Summary file: {summary_path.resolve()}")
    print("=" * 65)
    print("If any symbols failed, check download_summary.csv and retry later.")
    print("Financial metrics are latest available snapshots, not historical daily values.")

if __name__ == "__main__":
    main()
