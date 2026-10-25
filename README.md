# NSE Stock Intelligence

An offline-capable Flask dashboard that reads the CSV files in `Nifty50_Individual_CSVs/`. It makes no stock-data requests and does not fetch online prices. The sample CSVs contain OHLCV and repeated latest-snapshot financial metrics; those snapshot values are shown once and excluded from daily model features.

## Run on Windows / VS Code

Open this folder in VS Code, then run in its PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The prediction models use pandas, NumPy and scikit-learn. FFANN is an MLPRegressor feed-forward network; ANFIS predicts next-session percentage returns using a compact Gaussian rule base and first-order Sugeno linear consequents. It tunes membership count, width and consequent regularization against chronological validation data. Both use chronological 70/15/15 splits, training-only preprocessing, and held-out test metrics.

## CSV format

At minimum, each company CSV needs `Date` and `Close`. `Open`, `High`, `Low`, `Volume`, `Adj Close`, `Market_Cap`, `PE_Ratio`, `ROE` and `Debt_to_Equity` are used when present. Files are discovered dynamically by scanning the dataset folder. Values missing in a CSV display as N/A. Historical targets refer to the next actual CSV observation, while the final forecast is an estimate for the next available trading session; no future date is claimed.

The current sample collection was originally generated using Yahoo Finance by `nifty50_dataset.py`. The dashboard only reads these existing local CSVs. Do not run that downloader for offline operation; it calls an external data provider.

Each successful prediction or comparison appends its held-out evaluation metrics to `prediction_metrics.csv` in this project folder. A comparison writes one row for each model. The file opens directly in VS Code or Excel.

Displayed price accuracy is defined here as `100% − MAPE`; regression has no single standard accuracy percentage. RMSE (%) is the held-out RMSE divided by the mean absolute actual test price, then multiplied by 100. The metrics CSV also retains raw RMSE in INR.

To train and evaluate both models for every company from the VS Code terminal, activate the virtual environment and run:

```powershell
python train_all.py
```

Or, without activating the environment:

```powershell
.\.venv\Scripts\python.exe train_all.py
```

The script prints progress, continues if an individual company fails, appends per-model metrics to `prediction_metrics.csv`, and writes one overall row per company to `training_batch_summary.csv`.

## Limitations

Financial metrics have no reporting date field in the supplied data, so the dashboard labels them “Latest Available Financial Snapshot” without inventing a snapshot date. Some fields, including ROE, may be absent. Prediction requires adequate history after indicator warm-up; shorter files return a clear insufficient-data error. MAPE omits zero-valued actual targets. Forecasts are estimates, not trading advice or guaranteed outcomes.

