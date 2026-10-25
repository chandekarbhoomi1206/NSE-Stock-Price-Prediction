"""Train and evaluate ANFIS and FFANN for every valid local company CSV."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path

from app import evaluate, symbols

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / 'training_batch_summary.csv'

def main():
    companies = symbols()
    if not companies:
        print('No valid company CSV files found.')
        return 1
    columns = ['run_utc','symbol','status','error','anfis_mae','anfis_rmse','anfis_rmse_percent','anfis_mape_percent','anfis_accuracy_percent','anfis_directional_accuracy_percent','anfis_rse','anfis_r2','ffann_mae','ffann_rmse','ffann_rmse_percent','ffann_mape_percent','ffann_accuracy_percent','ffann_directional_accuracy_percent','ffann_rse','ffann_r2','training_rows','validation_rows','testing_rows']
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    completed = failed = 0
    with SUMMARY.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        total = len(companies)
        for i, company in enumerate(companies, start=1):
            symbol = company['symbol']
            print(f'[{i}/{total}] Training ANFIS and FFANN for {symbol}...', flush=True)
            row = {'run_utc': stamp, 'symbol': symbol, 'status': 'failed', 'error': ''}
            try:
                result = evaluate(symbol, 'Compare Both Models')
                for model, prefix in [('ANFIS','anfis'), ('FFANN','ffann')]:
                    item = result[model]
                    row.update({f'{prefix}_mae':item['mae'],f'{prefix}_rmse':item['rmse'],f'{prefix}_rmse_percent':item['rmse_percent'],
                                f'{prefix}_mape_percent':item['mape'],f'{prefix}_accuracy_percent':item['accuracy'],
                                f'{prefix}_directional_accuracy_percent':item['directional_accuracy'],
                                f'{prefix}_rse':item['rse'],f'{prefix}_r2':item['r2']})
                row.update(status='completed', training_rows=result['ANFIS']['train_rows'],
                           validation_rows=result['ANFIS']['validation_rows'],
                           testing_rows=result['ANFIS']['test_rows'])
                completed += 1
                print(f"    Done | ANFIS MAE {result['ANFIS']['mae']:.3f} | FFANN MAE {result['FFANN']['mae']:.3f}", flush=True)
            except Exception as exc:
                row['error'] = str(exc).replace('\n',' ')[:500]
                failed += 1
                print(f'    Failed: {row["error"]}', flush=True)
            writer.writerow(row)
            f.flush()
    print(f'\nFinished: {completed} companies completed, {failed} failed.')
    print(f'Model metric history: {ROOT / "prediction_metrics.csv"}')
    print(f'Batch summary: {SUMMARY}')
    return 0 if completed else 1

if __name__ == '__main__':
    raise SystemExit(main())
