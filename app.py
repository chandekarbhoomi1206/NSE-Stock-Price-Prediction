"""Offline CSV-backed NSE analytics dashboard."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import csv, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, Response, abort
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Nifty50_Individual_CSVs'
app = Flask(__name__)

ALIASES = {'ADJ CLOSE':'Adj Close','ADJUSTED CLOSE':'Adj Close','MARKET CAP':'Market_Cap','MARKET_CAP':'Market_Cap','PE RATIO':'PE_Ratio','PE_RATIO':'PE_Ratio','DEBT TO EQUITY':'Debt_to_Equity','DEBT_TO_EQUITY':'Debt_to_Equity'}

@lru_cache(maxsize=128)
def load_symbol(symbol, mtime):
    path = DATA / f'{symbol}.csv'
    df = pd.read_csv(path)
    df.columns = [ALIASES.get(str(c).strip().upper(), str(c).strip()) for c in df.columns]
    if not {'Date','Close'}.issubset(df.columns): raise ValueError('CSV needs Date and Close columns')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['Open','High','Low','Close','Adj Close','Volume','Market_Cap','PE_Ratio','ROE','Debt_to_Equity']:
        if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date','Close']).sort_values('Date').drop_duplicates('Date', keep='last').reset_index(drop=True)
    return df

def symbols():
    found=[]
    for p in DATA.glob('*.csv'):
        try:
            df=pd.read_csv(p,nrows=3); cols={ALIASES.get(str(c).strip().upper(),str(c).strip()) for c in df.columns}
            if {'Date','Close'} <= cols:
                display_names={'HDFCLIFE':'HDFC Life','HDFCBANK':'HDFC Bank'}
                found.append({'symbol':p.stem,'name':display_names.get(p.stem,p.stem.replace('-',' & '))})
        except Exception: pass
    return sorted(found,key=lambda x:x['symbol'])

def get_df(sym):
    if not sym or not sym.replace('-','').replace('&','').isalnum(): abort(404)
    p=DATA/f'{sym}.csv'
    if not p.is_file(): abort(404)
    try: return load_symbol(sym,p.stat().st_mtime_ns)
    except Exception as e: abort(Response(str(e),status=422))

def json_records(df):
    z=df.copy(); z['Date']=z.Date.dt.strftime('%Y-%m-%d'); z=z.replace({np.nan:None})
    return z.to_dict(orient='records')

def rsi(series, period=14):
    d=series.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/period,min_periods=period,adjust=False).mean(); al=loss.ewm(alpha=1/period,min_periods=period,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))

def features(df):
    c=df.Close.astype(float); out=pd.DataFrame(index=df.index)
    out['close_lag1']=c; out['close_lag2']=c.shift(1); out['close_lag3']=c.shift(2)
    out['return1']=c.pct_change(); out['return2']=c.pct_change(2)
    out['ma7']=c.rolling(7).mean(); out['ma20']=c.rolling(20).mean(); out['ma50']=c.rolling(50).mean()
    out['rsi14']=rsi(c)
    if 'Volume' in df: out['volume_change']=df.Volume.pct_change().replace([np.inf,-np.inf],np.nan)
    else: out['volume_change']=0.0
    y=c.shift(-1); valid=out.replace([np.inf,-np.inf],np.nan).notna().all(axis=1)&y.notna()
    return out[valid],y[valid],df.loc[valid,'Date']

def anfis_fit_predict(Xtr,ytr,Xv,yv,Xtest,Xfuture):
    # First-order Sugeno ANFIS. Tune a compact Gaussian rule base and ridge penalty
    # using only the chronological validation block. Three inputs cap the largest
    # candidate at 27 rules / 108 consequent parameters.
    selected = [0, 3, 8]  # latest close, one-session return, RSI(14)
    Xtr, Xv, Xtest, Xfuture = (x[:, selected] for x in (Xtr, Xv, Xtest, Xfuture))
    means=Xtr.mean(0); std=Xtr.std(0); std=np.where(std<1e-9,1,std)
    a=(Xtr-means)/std; av=(Xv-means)/std; at=(Xtest-means)/std; af=(Xfuture-means)/std
    import itertools
    best=None
    for membership_count in (2,3):
        levels=np.linspace(.2,.8,membership_count)
        centers=np.quantile(a,levels,axis=0)
        # Width candidates span narrower/local to smoother memberships.
        base_width=np.maximum(np.std(a,axis=0),.25)
        rules=list(itertools.product(range(membership_count),repeat=a.shape[1]))
        for width_factor in (.45,.7,.75,1.0):
            widths=base_width*width_factor
            def design(q):
                mu=np.exp(-.5*((q[:,None,:]-centers[None,:,:])/widths[None,None,:])**2)
                w=np.ones((len(q),len(rules)))
                for j,rule in enumerate(rules):
                    for k,m in enumerate(rule): w[:,j]*=mu[:,m,k]
                w/=np.maximum(w.sum(1,keepdims=True),1e-15)
                terms=np.concatenate([q,np.ones((len(q),1))],axis=1)
                return (w[:,:,None]*terms[:,None,:]).reshape(len(q),-1)
            d=design(a); dv=design(av)
            # Ridge grid covers weak through strong consequent regularization.
            for alpha in (1e-5,1e-4,1e-3,.01,.03,.1,.3,1.0,3.0,10.0):
                reg=np.eye(d.shape[1])*alpha; reg[-1,-1]=0
                try: beta=np.linalg.solve(d.T@d+reg,d.T@ytr)
                except np.linalg.LinAlgError: beta=np.linalg.lstsq(d.T@d+reg,d.T@ytr,rcond=None)[0]
                val_prediction=dv@beta
                err=float(np.mean((val_prediction-yv)**2))
                val_accuracy=float(np.mean(np.sign(val_prediction)==np.sign(yv)))
                # Keep price-return fit as the primary validation objective; use
                # direction as a tie-breaker so tuning does not sacrifice the
                # regression task to noisy up/down labels.
                if best is None or err<best[0]-1e-12 or (abs(err-best[0])<=1e-12 and val_accuracy>best[1]):
                    best=(err,val_accuracy,beta,membership_count,width_factor,alpha)
    _,_,beta,membership_count,width_factor,alpha=best
    levels=np.linspace(.2,.8,membership_count); centers=np.quantile(a,levels,axis=0)
    widths=np.maximum(np.std(a,axis=0),.25)*width_factor
    rules=list(itertools.product(range(membership_count),repeat=a.shape[1]))
    def final_design(q):
        mu=np.exp(-.5*((q[:,None,:]-centers[None,:,:])/widths[None,None,:])**2)
        w=np.ones((len(q),len(rules)))
        for j,rule in enumerate(rules):
            for k,m in enumerate(rule): w[:,j]*=mu[:,m,k]
        w/=np.maximum(w.sum(1,keepdims=True),1e-15)
        terms=np.concatenate([q,np.ones((len(q),1))],axis=1)
        return (w[:,:,None]*terms[:,None,:]).reshape(len(q),-1)
    # Refit selected configuration using training rows only before test/forecast.
    d=final_design(a); reg=np.eye(d.shape[1])*alpha; reg[-1,-1]=0
    beta=np.linalg.solve(d.T@d+reg,d.T@ytr)
    return final_design(at)@beta,final_design(af)@beta

def evaluate(sym, only_model=None):
    df=get_df(sym); X,y,dates=features(df)
    n=len(X)
    if n<80: raise ValueError(f'Insufficient usable history ({n} rows); at least 80 processed rows are required.')
    nt=int(n*.70); nv=int(n*.15); ntest=n-nt-nv
    if min(nt,nv,ntest)<5: raise ValueError('Dataset is too small for chronological train/validation/test splits.')
    Xtr,Xv,Xte=X.iloc[:nt].values,X.iloc[nt:nt+nv].values,X.iloc[nt+nv:].values
    ytr,yv,yte=y.iloc[:nt].values,y.iloc[nt:nt+nv].values,y.iloc[nt+nv:].values
    scaler=StandardScaler().fit(Xtr); A=scaler.transform(Xtr); B=scaler.transform(Xv); C=scaler.transform(Xte); F=scaler.transform(X.iloc[[-1]].values)
    preds={}
    selected=['ANFIS','FFANN'] if only_model in (None,'Compare Both Models') else [only_model]
    for model in selected:
        if model=='ANFIS':
            # Learn the next-session percentage return, then map it back to INR.
            # This keeps the consequent scale comparable across share-price levels.
            prior_train=X.iloc[:nt]['close_lag1'].values
            prior_val=X.iloc[nt:nt+nv]['close_lag1'].values
            prior_test=X.iloc[nt+nv:]['close_lag1'].values
            prior_latest=X.iloc[[-1]]['close_lag1'].values
            train_return=ytr/prior_train-1
            val_return=yv/prior_val-1
            test_return,forecast_return=anfis_fit_predict(A,train_return,B,val_return,C,F)
            p=prior_test*(1+test_return); f=prior_latest*(1+forecast_return)
        elif model=='FFANN':
            # Learn next-session returns so validation selection can score the
            # same up/down objective used by the displayed directional accuracy.
            prior_train=X.iloc[:nt]['close_lag1'].values
            prior_val=X.iloc[nt:nt+nv]['close_lag1'].values
            prior_test=X.iloc[nt+nv:]['close_lag1'].values
            prior_latest=X.iloc[[-1]]['close_lag1'].values
            ytr_return=ytr/prior_train-1; yv_return=yv/prior_val-1
            # Scale return targets from training rows only for stable optimization.
            y_mean=float(np.mean(ytr_return)); y_scale=float(np.std(ytr_return)) or 1.0
            ytr_scaled=(ytr_return-y_mean)/y_scale; yv_scaled=(yv_return-y_mean)/y_scale
            best=None
            for hidden in [(32,16),(48,24)]:
                net=MLPRegressor(hidden_layer_sizes=hidden,activation='relu',solver='adam',
                    alpha=.001,learning_rate_init=.001,max_iter=1,warm_start=True,
                    shuffle=False,early_stopping=False,random_state=42,tol=0.0,
                    batch_size=min(32,len(A)))
                best_accuracy=-1.0; best_loss=float('inf'); best_weights=None; stale=0; epochs=0
                # Train one epoch at a time and use only the next chronological
                # validation block for stopping and architecture selection.
                for epoch in range(1,801):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', ConvergenceWarning)
                        net.fit(A,ytr_scaled)
                    val_pred=net.predict(B)
                    loss=float(np.mean((val_pred-yv_scaled)**2))
                    accuracy=float(np.mean(np.sign(val_pred)==np.sign(yv_scaled)))
                    epochs=epoch
                    if accuracy>best_accuracy or (accuracy==best_accuracy and loss<best_loss-1e-6):
                        best_accuracy=accuracy; best_loss=loss
                        best_weights=( [w.copy() for w in net.coefs_], [b.copy() for b in net.intercepts_] ); stale=0
                    else: stale+=1
                    if epoch>=100 and stale>=35: break
                net.coefs_,net.intercepts_=best_weights
                if best is None or best_accuracy>best[0] or (best_accuracy==best[0] and best_loss<best[1]): best=(best_accuracy,best_loss,net,epochs)
            net=best[2]
            p_return=net.predict(C)*y_scale+y_mean
            f_return=net.predict(F)*y_scale+y_mean
            p=prior_test*(1+p_return); f=prior_latest*(1+f_return)
        else: raise ValueError('Choose ANFIS, FFANN, or Compare Both Models.')
        err=yte-p; mae=float(np.mean(np.abs(err))); nz=np.abs(yte)>1e-12
        denom=float(np.sum((yte-yte.mean())**2)); rse=float(np.sum(err**2)/denom) if denom else None
        prior_close=X.iloc[nt+nv:]['close_lag1'].values
        directional_accuracy=float(np.mean(np.sign(yte-prior_close)==np.sign(p-prior_close))*100)
        price_accuracy=100.0-float(np.mean(np.abs(err[nz]/yte[nz]))*100) if nz.any() else None
        rmse=float(np.sqrt(np.mean(err**2)))
        rmse_percent=float(rmse/np.mean(np.abs(yte))*100) if np.mean(np.abs(yte))>1e-12 else None
        r2=float(1-np.sum(err**2)/denom) if denom else None
        mape=float(np.mean(np.abs(err[nz]/yte[nz]))*100) if nz.any() else None
        preds[model]={'prediction':float(f[0]),'mae':mae,'mape':mape,'accuracy':price_accuracy,'directional_accuracy':directional_accuracy,'rmse':rmse,'rmse_percent':rmse_percent,'rse':rse,'r2':r2,'test_dates':pd.to_datetime(dates.iloc[nt+nv:]).dt.strftime('%Y-%m-%d').tolist(),'actual':yte.tolist(),'test_prediction':p.tolist(),'train_rows':nt,'validation_rows':nv,'test_rows':ntest,'status':'Trained'}
    preds['baseline']={'mae':float(np.mean(np.abs(yte-X.iloc[nt+nv:].close_lag1.values)))}
    latest=float(df.Close.iloc[-1]); latest_date=df.Date.iloc[-1].strftime('%Y-%m-%d')
    for model in selected:
        p=preds[model]['prediction']; preds[model].update(latest_close=latest,change=p-latest,change_pct=(p-latest)/latest*100 if latest else None,latest_date=latest_date,rsi=float(rsi(df.Close).iloc[-1]) if pd.notna(rsi(df.Close).iloc[-1]) else None)
    preds.update(symbol=sym,model=only_model or 'Compare Both Models',latest_close=latest,latest_date=latest_date)
    record_metrics(preds, selected)
    return preds

def record_metrics(results, models):
    """Append each completed model evaluation to a local, analysis-ready CSV."""
    path = ROOT / 'prediction_metrics.csv'
    columns = ['run_utc','symbol','model','latest_data_date','latest_close','predicted_next_close',
               'mae','rmse','rmse_percent','mape_percent','accuracy_percent','directional_accuracy_percent','rse','r2',
               'training_rows','validation_rows','testing_rows','persistence_baseline_mae']
    new_file = not path.exists()
    old_rows=[]
    if not new_file:
        with path.open('r',newline='',encoding='utf-8-sig') as old:
            reader=csv.DictReader(old)
            if reader.fieldnames != columns: old_rows=list(reader)
    if old_rows:
        with path.open('w',newline='',encoding='utf-8') as migrated:
            writer=csv.DictWriter(migrated,fieldnames=columns); writer.writeheader()
            for old_row in old_rows:
                # Historical runs did not store RMSE or this price-accuracy metric.
                writer.writerow({k:old_row.get(k,'') for k in columns})
        new_file=True
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file: writer.writeheader()
        for model in models:
            item = results[model]
            writer.writerow({
                'run_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'symbol': results['symbol'], 'model': model,
                'latest_data_date': item['latest_date'], 'latest_close': item['latest_close'],
                'predicted_next_close': item['prediction'], 'mae': item['mae'], 'rmse': item['rmse'], 'rmse_percent': item['rmse_percent'],
                'mape_percent': item['mape'], 'accuracy_percent': item['accuracy'],
                'directional_accuracy_percent': item['directional_accuracy'], 'rse': item['rse'], 'r2': item['r2'],
                'training_rows': item['train_rows'], 'validation_rows': item['validation_rows'],
                'testing_rows': item['test_rows'],
                'persistence_baseline_mae': results['baseline']['mae']})

@app.get('/')
def index(): return render_template('index.html')
@app.get('/api/companies')
def api_companies(): return jsonify({'companies':symbols()})
@app.get('/api/company/<symbol>/history')
def history(symbol):
    df=get_df(symbol); return jsonify({'symbol':symbol,'records':json_records(df)})
@app.get('/api/company/<symbol>/summary')
def summary(symbol):
    d=get_df(symbol); c=d.Close; latest=d.iloc[-1]; prev=d.iloc[-2] if len(d)>1 else None
    return jsonify({'symbol':symbol,'latest_close':float(c.iloc[-1]),'previous_close':float(prev.Close) if prev is not None else None,'change':float(c.iloc[-1]-prev.Close) if prev is not None else None,'change_pct':float((c.iloc[-1]/prev.Close-1)*100) if prev is not None and prev.Close else None,'high_52w':float(d.tail(252).High.max()) if 'High' in d else None,'low_52w':float(d.tail(252).Low.min()) if 'Low' in d else None,'latest_date':latest.Date.strftime('%Y-%m-%d'),'rows':len(d)})
@app.get('/api/company/<symbol>/financials')
def financials(symbol):
    d=get_df(symbol); row=d.iloc[-1]; fields=['Market_Cap','PE_Ratio','Debt_to_Equity']
    return jsonify({'snapshot_label':'Latest Available Financial Snapshot','snapshot_date':None,'metrics':{f:(float(row[f]) if f in d and pd.notna(row[f]) else None) for f in fields}})
@app.get('/api/company/<symbol>/indicators')
def indicators(symbol):
    d=get_df(symbol); c=d.Close; val=rsi(c).iloc[-1]
    return jsonify({'rsi':float(val) if pd.notna(val) else None,'rsi_status':'Overbought' if pd.notna(val) and val>70 else ('Oversold' if pd.notna(val) and val<30 else ('Neutral' if pd.notna(val) else 'N/A')),'ma7':float(c.tail(7).mean()) if len(c)>=7 else None,'ma20':float(c.tail(20).mean()) if len(c)>=20 else None,'ma50':float(c.tail(50).mean()) if len(c)>=50 else None,'change_pct':float((c.iloc[-1]/c.iloc[-2]-1)*100) if len(c)>1 else None})
@app.post('/api/predict')
def predict():
    body=request.get_json(silent=True) or {}; model=body.get('model'); sym=body.get('symbol')
    if model not in ['ANFIS','FFANN']: return jsonify(error='Select ANFIS or FFANN.'),400
    try: return jsonify(evaluate(sym,model))
    except Exception as e: return jsonify(error=str(e)),422
@app.post('/api/model-comparison')
def compare():
    body=request.get_json(silent=True) or {}
    try: return jsonify(evaluate(body.get('symbol'),'Compare Both Models'))
    except Exception as e: return jsonify(error=str(e)),422
@app.get('/api/company/<symbol>/download')
def download(symbol):
    df=get_df(symbol); p=DATA/f'{symbol}.csv'
    return Response(p.read_text(encoding='utf-8-sig'),mimetype='text/csv',headers={'Content-Disposition':f'attachment; filename="{symbol}_history.csv"'})

if __name__=='__main__': app.run(debug=True,port=5000)
