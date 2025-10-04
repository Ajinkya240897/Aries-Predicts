\
# Aries Predicts - Advanced version (Streamlit-compatible)
# Added indicators: SuperTrend, Ichimoku, StochRSI, KAMA, CMF, rolling skew/kurtosis, realized vol
# Added classifier for direction, Brier score, bootstrap residual intervals, backtest metrics
# Dependencies: streamlit, yfinance, pandas, numpy, scikit-learn, requests, joblib
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, csv, os, re, math
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="Aries Predicts", layout="wide")
st.title("Aries Predicts")

# Sidebar inputs (unchanged)
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (no suffix)", value="")
    fmp_key = st.text_input("FMP API key (optional)", type="password")
    horizons = st.multiselect("Horizon(s)", ["3 days","15 days","1 month","3 months","6 months","1 year"], default=["3 days"])
    run = st.button("Run Aries Predicts")
    st.markdown("---")
    st.write("Advanced options (defaults are beginner-friendly)")
    use_saved_models = st.checkbox("Prefer pre-saved models (models/ folder)", value=True)
    show_debug = st.checkbox("Show diagnostics (developer view)", value=False)
    st.markdown("---")
    st.write("Notes: App uses ensemble models + calibrated ranges; for better fundamentals add FMP API key.")

# Lightweight lexicons
_POS = set("good great positive outperform beat strong upgrade profits growth excellent gain upward bull rally recovery improve robust".split())
_NEG = set("bad poor negative underperform loss downgrade weak decline fall downward bear risk scandal fraud halt".split())

def safe_str(x):
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

def sentiment_score(text):
    try:
        if not text: return 50, "Neutral"
        txt = safe_str(text).lower()
        tokens = re.findall(r"\\b\\w+\\b", txt)
        pos = sum(1 for t in tokens if t in _POS)
        neg = sum(1 for t in tokens if t in _NEG)
        if (pos + neg) > 0:
            score = int(50 + 40 * (pos - neg) / (pos + neg))
            score = max(0, min(100, score))
            label = "Positive" if score >= 65 else ("Negative" if score <= 35 else "Neutral")
            return score, label
        return 50, "Neutral"
    except Exception:
        return 50, "Neutral"

# Fetch helpers
@st.cache_data(ttl=600)
def fetch_history_try(ticker, period='5y'):
    t = ticker.strip().upper()
    candidates = [t, t + '.NS', t + '.BO']
    last_err = None
    for s in candidates:
        try:
            tk = yf.Ticker(s)
            hist = tk.history(period=period, interval='1d', auto_adjust=False)
            if hist is not None and not hist.empty:
                hist.index = pd.to_datetime(hist.index)
                return hist, s, None
        except Exception as e:
            last_err = str(e)
            continue
    return pd.DataFrame(), None, last_err

@st.cache_data(ttl=3600)
def fetch_profile_fmp(ticker, apikey):
    if not apikey: return {}
    trials = [ticker.strip().upper(), ticker.strip().upper() + ".NS"]
    for s in trials:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{s}?apikey={apikey}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and j:
                    return j[0]
                if isinstance(j, dict):
                    return j
        except Exception:
            continue
    return {}

@st.cache_data(ttl=3600)
def fetch_yf_info(ticker):
    t = ticker.strip().upper()
    for s in [t, t + ".NS", t + ".BO"]:
        try:
            tk = yf.Ticker(s)
            info = getattr(tk, "info", {}) or {}
            if info:
                return info, s
        except Exception:
            continue
    return {}, None

# ------------------
# Advanced indicators
# ------------------
def stoch_rsi(close, period=14, smooth_k=3, smooth_d=3):
    delta = close.diff()
    up = np.maximum(delta, 0)
    down = -np.minimum(delta, 0)
    rs = up.rolling(period).mean() / (down.rolling(period).mean() + 1e-9)
    rsi = 100 - 100/(1+rs)
    low_rsi = rsi.rolling(period).min()
    high_rsi = rsi.rolling(period).max()
    stoch = (rsi - low_rsi) / (high_rsi - low_rsi + 1e-9)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.fillna(0), d.fillna(0)

def ichimoku(df):
    high = df['High']; low = df['Low']; close = df['Close']
    high9 = high.rolling(9).max(); low9 = low.rolling(9).min()
    tenkan = (high9 + low9)/2
    high26 = high.rolling(26).max(); low26 = low.rolling(26).min()
    kijun = (high26 + low26)/2
    senkou_a = ((tenkan + kijun)/2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min())/2).shift(26)
    chikou = close.shift(-26)
    return {"tenkan":tenkan,"kijun":kijun,"senkou_a":senkou_a,"senkou_b":senkou_b,"chikou":chikou}

def kama(series, window=10, pow1=2, pow2=30):
    # simplified KAMA (adaptive smoothing)
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    er = change / (volatility + 1e-9)
    sc = (er*(2/(pow1+1)-2/(pow2+1)) + 2/(pow2+1))**2
    kama = series.copy()
    kama.iloc[:window] = series.iloc[:window]
    for i in range(window, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(series.iloc[i]-kama.iloc[i-1])
    return kama.fillna(method='ffill').fillna(method='bfill')

def chaikin_money_flow(df, n=20):
    adl = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9) * df['Volume']
    cmf = adl.rolling(n).sum() / (df['Volume'].rolling(n).sum() + 1e-9)
    return cmf.fillna(0)

def rolling_stats(series, window=20):
    return {"skew": series.pct_change().rolling(window).skew().fillna(0), "kurt": series.pct_change().rolling(window).kurt().fillna(0), "rv": series.pct_change().rolling(window).std().fillna(0)}

def compute_supertrend(df, period=10, multiplier=3.0):
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low)/2.0
    upperband = hl2 + multiplier*atr; lowerband = hl2 - multiplier*atr
    final_upper = upperband.copy(); final_lower = lowerband.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = 1
    for i in range(len(df)):
        if i==0:
            final_upper.iloc[i] = upperband.iloc[i]; final_lower.iloc[i] = lowerband.iloc[i]; supertrend.iloc[i] = close.iloc[i]; continue
        final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i-1]) if close.iloc[i-1] > final_upper.iloc[i-1] else upperband.iloc[i]
        final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i-1]) if close.iloc[i-1] < final_lower.iloc[i-1] else lowerband.iloc[i]
        if close.iloc[i] > final_upper.iloc[i-1]: direction = 1
        elif close.iloc[i] < final_lower.iloc[i-1]: direction = -1
        supertrend.iloc[i] = final_lower.iloc[i] if direction==1 else final_upper.iloc[i]
    last_dir = 1 if close.iloc[-1] > supertrend.iloc[-1] else -1
    return supertrend, last_dir

# base indicators compute
def compute_indicators(df):
    df = df.copy()
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    df['ret1'] = close.pct_change()
    df['ma7'] = close.rolling(7).mean(); df['ma21'] = close.rolling(21).mean()
    df['ema12'] = close.ewm(span=12, adjust=False).mean(); df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    delta = close.diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
    df['rsi14'] = 100.0 - (100.0 / (1.0 + (up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9))))
    prev_close = close.shift(1)
    tr1 = high - low; tr2 = (high - prev_close).abs(); tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); df['atr14'] = tr.rolling(14).mean()
    df['vol30'] = df['ret1'].rolling(30).std().fillna(0)
    df['obv'] = (vol * np.sign(df['ret1'].fillna(0))).cumsum()
    # advanced
    df['stoch_k'], df['stoch_d'] = stoch_rsi(close)
    ich = ichimoku(df)
    df['tenkan'] = ich['tenkan']; df['kijun'] = ich['kijun']
    df['senkou_a'] = ich['senkou_a']; df['senkou_b'] = ich['senkou_b']
    df['kama21'] = kama(close, window=10)
    df['cmf20'] = chaikin_money_flow(df, n=20)
    rs = rolling_stats(close, window=20)
    df['rskew20'] = rs['skew']; df['rkurt20'] = rs['kurt']; df['rv20'] = rs['rv']
    st, st_dir = compute_supertrend(df)
    df['supertrend'] = st; df['supertrend_dir'] = st_dir
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

# features
def make_X_y(df, days):
    df = df.copy()
    df['target'] = df['Close'].shift(-days) / df['Close'] - 1
    df['lag1'] = df['ret1'].shift(1); df['lag2'] = df['ret1'].shift(2)
    df['ma_diff'] = (df['ma7'] - df['ma21'])/(df['ma21']+1e-9)
    features = ['lag1','lag2','ma_diff','macd','rsi14','atr14','vol30','obv','stoch_k','stoch_d','kama21','cmf20','rskew20','rv20']
    df = df.dropna()
    if df.empty: return None, None, None
    X = df[features].copy(); y = df['target'].copy()
    return X, y, df

# load/train models with classifier + HGB
@st.cache_resource
def load_or_train_models(ticker, hist_df, prefer_saved=True):
    info = {"loaded": False, "trained": False, "errors": []}
    models_dir = Path("models"); tdir = models_dir / ticker.upper(); general_dir = models_dir / "general"
    try:
        if prefer_saved and tdir.exists():
            rf = joblib.load(tdir / "rf.joblib"); et = joblib.load(tdir / "et.joblib"); gbr = joblib.load(tdir / "gbr.joblib")
            hgb = joblib.load(tdir / "hgb.joblib") if (tdir / "hgb.joblib").exists() else None
            meta = joblib.load(tdir / "meta.joblib"); clf = joblib.load(tdir / "clf.joblib") if (tdir / "clf.joblib").exists() else None
            info["loaded"] = True; return {"rf":rf,"et":et,"gbr":gbr,"hgb":hgb,"meta":meta,"clf":clf,"info":info}
        if prefer_saved and general_dir.exists():
            rf = joblib.load(general_dir / "rf.joblib"); et = joblib.load(general_dir / "et.joblib"); gbr = joblib.load(general_dir / "gbr.joblib")
            hgb = joblib.load(general_dir / "hgb.joblib") if (general_dir / "hgb.joblib").exists() else None
            meta = joblib.load(general_dir / "meta.joblib"); clf = joblib.load(general_dir / "clf.joblib") if (general_dir / "clf.joblib").exists() else None
            info["loaded"] = True; return {"rf":rf,"et":et,"gbr":gbr,"hgb":hgb,"meta":meta,"clf":clf,"info":info}
    except Exception as e:
        info["errors"].append(str(e))
    try:
        X,y,dframe = make_X_y(hist_df, days=15) if hist_df is not None and not hist_df.empty else (None,None,None)
    except Exception:
        X,y,dframe = (None,None,None)
    try:
        if X is None or len(X) < 120:
            rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=1)
            et = ExtraTreesRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=1)
            gbr = GradientBoostingRegressor(n_estimators=80)
            hgb = HistGradientBoostingRegressor(max_iter=150) if hasattr(HistGradientBoostingRegressor, 'fit') else None
            meta = RidgeCV(alphas=[0.1,1.0,10.0])
            clf = None
            if X is not None and len(X) >= 30:
                rf.fit(X,y); et.fit(X,y); gbr.fit(X,y)
                if hgb is not None:
                    try: hgb.fit(X,y)
                    except Exception: hgb = None
                oof = np.column_stack([rf.predict(X), et.predict(X), gbr.predict(X)])
                meta.fit(oof, y)
                # direction classifier
                try:
                    y_dir = (y>0).astype(int)
                    base_clf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=1)
                    base_clf.fit(X, y_dir)
                    clf = CalibratedClassifierCV(base_clf, cv=3).fit(X, y_dir)
                except Exception:
                    clf = None
            info["trained"] = True; return {"rf":rf,"et":et,"gbr":gbr,"hgb":hgb,"meta":meta,"clf":clf,"info":info}
        # full training
        rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=1)
        et = ExtraTreesRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=1)
        gbr = GradientBoostingRegressor(n_estimators=160)
        hgb = HistGradientBoostingRegressor(max_iter=200)
        tscv = TimeSeriesSplit(n_splits=4)
        oof = np.zeros((len(X),3))
        for i,(name,model) in enumerate([("rf",rf),("et",et),("gbr",gbr)]):
            col = np.zeros(len(X))
            for tr,va in tscv.split(X):
                model.fit(X.iloc[tr], y.iloc[tr]); col[va] = model.predict(X.iloc[va])
            oof[:,i] = col
        meta = RidgeCV(alphas=[0.1,1.0,10.0]); valid = ~np.any(np.isnan(oof), axis=1)
        if valid.sum() > 10: meta.fit(oof[valid], y.iloc[valid])
        rf.fit(X,y); et.fit(X,y); gbr.fit(X,y)
        try: hgb.fit(X,y)
        except Exception: hgb = None
        # direction classifier
        clf = None
        try:
            y_dir = (y>0).astype(int)
            base_clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=1)
            base_clf.fit(X, y_dir)
            clf = CalibratedClassifierCV(base_clf, cv=3).fit(X, y_dir)
        except Exception:
            clf = None
        info["trained"] = True
        return {"rf":rf,"et":et,"gbr":gbr,"hgb":hgb,"meta":meta,"clf":clf,"info":info}
    except Exception as e:
        info["errors"].append(str(e))
        rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=1)
        et = ExtraTreesRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=1)
        gbr = GradientBoostingRegressor(n_estimators=80)
        meta = RidgeCV(alphas=[0.1,1.0,10.0]); clf = None; hgb = None
        try:
            if X is not None and len(X) >= 10:
                rf.fit(X,y); et.fit(X,y); gbr.fit(X,y); meta.fit(np.column_stack([rf.predict(X), et.predict(X), gbr.predict(X)]), y)
        except Exception:
            pass
        return {"rf":rf,"et":et,"gbr":gbr,"hgb":hgb,"meta":meta,"clf":clf,"info":info}

# bootstrap residual interval helper
def bootstrap_residual_interval(residuals, final_pred, alpha=0.88, n_boot=400):
    if residuals is None or len(residuals) < 8:
        q = np.quantile(np.abs(residuals) if residuals is not None and len(residuals)>0 else np.array([0.02]), alpha)
        return final_pred - q, final_pred + q
    boot_qs = []
    n = len(residuals)
    for _ in range(n_boot):
        sample = np.random.choice(residuals, size=n, replace=True)
        boot_qs.append(np.quantile(np.abs(sample), alpha))
    q = float(np.quantile(boot_qs, 0.9))
    return final_pred - q, final_pred + q

# backtest metrics (MAPE, MAE, directional accuracy)
def backtest_metrics(models_dict, X, y):
    preds = []
    for mdl in [models_dict.get("rf"), models_dict.get("et"), models_dict.get("gbr"), models_dict.get("hgb")]:
        try:
            if mdl is not None:
                preds.append(mdl.predict(X))
        except Exception:
            pass
    if not preds: return {}
    stack = np.vstack(preds)
    ens = np.nanmean(stack, axis=0)
    mape = np.mean(np.abs((y - ens) / (y + 1e-9))) * 100
    mae = np.mean(np.abs(y - ens))
    dir_actual = (y > 0).astype(int); dir_pred = (ens > 0).astype(int)
    dir_acc = float((dir_actual == dir_pred).mean())
    return {"MAPE": round(float(mape),3), "MAE": round(float(mae),6), "dir_acc": round(float(dir_acc),3)}

# predict wrapper (uses classifier probability and bootstrapped intervals)
def predict_horizon(models_dict, hist_df, days):
    X,y,df = make_X_y(hist_df, days)
    if X is None or df is None or len(X) == 0:
        try:
            ret = hist_df['Close'].pct_change(days).iloc[-1]; return float(ret), {"method":"fallback_momentum"}
        except Exception:
            return 0.0, {"error":"no_data"}
    rf = models_dict.get("rf"); et = models_dict.get("et"); gbr = models_dict.get("gbr"); hgb = models_dict.get("hgb"); meta = models_dict.get("meta"); clf = models_dict.get("clf")
    preds = []; model_info = {}
    for name,mdl in [("rf",rf),("et",et),("gbr",gbr),("hgb",hgb)]:
        try:
            if mdl is None: raise Exception("missing")
            p = float(mdl.predict(X.iloc[[-1]])[0]); preds.append(p); model_info[name] = {"pred":p}
        except Exception as e:
            model_info[name] = {"error":str(e)}; preds.append(np.nan)
    preds = np.array([p for p in preds if not (p is None or np.isnan(p))])
    try:
        if preds.size == 0:
            meta_pred = float(X['lag1'].iloc[-1] if 'lag1' in X.columns else 0.0)
        else:
            meta_input = np.array([p if not (p is None or np.isnan(p)) else np.nanmedian(preds)]).reshape(1,-1)
            if meta_input.shape[1] < 3: meta_input = np.pad(meta_input, ((0,0),(0,3-meta_input.shape[1])), 'constant', constant_values=0)
            meta_pred = float(meta.predict(meta_input)[0])
    except Exception:
        meta_pred = float(np.nanmedian(preds)) if preds.size>0 else 0.0
    weighted = float(np.nanmean(preds)) if preds.size>0 else 0.0
    final = 0.6*meta_pred + 0.4*weighted
    # cap final by horizon scaling to avoid extreme values
    horizon_scale = min(1.0, days/22.0)
    max_abs = 0.8 * horizon_scale
    final = float(max(-max_abs, min(max_abs, final)))
    # quantile estimates
    q_low = None; q_high = None
    try:
        q_low_model = GradientBoostingRegressor(loss="quantile", alpha=0.12, n_estimators=120)
        q_high_model = GradientBoostingRegressor(loss="quantile", alpha=0.88, n_estimators=120)
        q_low_model.fit(X, y); q_high_model.fit(X, y)
        q_low = float(q_low_model.predict(X.iloc[[-1]])[0]); q_high = float(q_high_model.predict(X.iloc[[-1]])[0])
    except Exception:
        q_low = None; q_high = None
    # residuals-based bootstrap interval
    low_ret = None; high_ret = None
    try:
        oof_preds = []
        for mdl in [rf, et, gbr, hgb]:
            try:
                if mdl is not None: oof_preds.append(mdl.predict(X))
            except Exception:
                pass
        if oof_preds and len(oof_preds[0]) == len(X):
            oof_stack = np.vstack(oof_preds).T
            ridge = Ridge(alpha=1.0); ridge.fit(oof_stack, y)
            residuals = np.abs(y - ridge.predict(oof_stack))
            low_ret, high_ret = bootstrap_residual_interval(residuals, final, alpha=0.88, n_boot=300)
    except Exception:
        low_ret = None; high_ret = None
    if q_low is not None and q_high is not None:
        if low_ret is None: low_ret, high_ret = q_low, q_high
        else:
            low_ret = min(low_ret, q_low); high_ret = max(high_ret, q_high)
    if low_ret is None or high_ret is None:
        vol30 = float(hist_df['vol30'].iloc[-1]) if 'vol30' in hist_df.columns else 0.02
        buffer = max(0.01, min(0.35, vol30 * 2.5 * horizon_scale))
        low_ret = final - buffer; high_ret = final + buffer
    low_ret = float(max(-0.95, min(0.95, low_ret))); high_ret = float(max(-0.95, min(0.95, high_ret)))
    # classifier probability
    dir_prob = None; brier = None
    try:
        if clf is not None:
            dir_prob = float(clf.predict_proba(X.iloc[[-1]])[0,1])
            # compute Brier score on training if available
            if hasattr(clf, 'predict_proba'):
                try:
                    y_dir = (y>0).astype(int)
                    probs = clf.predict_proba(X)[:,1]
                    brier = float(brier_score_loss(y_dir, probs))
                except Exception:
                    brier = None
    except Exception:
        dir_prob = None; brier = None
    # directional accuracy recent
    dir_acc = None
    try:
        N = min(60, max(6, len(y)//4))
        if len(y) >= N+1:
            recent_actual = np.sign(y.iloc[-N:])
            pred_signs = []
            for mdl in [rf, et, gbr, hgb]:
                try:
                    if mdl is not None:
                        p = mdl.predict(X.iloc[-N:]); pred_signs.append(np.sign(p))
                except Exception:
                    pass
            if pred_signs:
                ensemble_sign = np.sign(np.nanmean(np.vstack(pred_signs), axis=0))
                dir_acc = float((ensemble_sign == recent_actual).mean())
    except Exception:
        dir_acc = None
    diag = {"models": model_info, "final_return": final, "quantiles": (low_ret, high_ret), "dir_prob": dir_prob, "brier": brier, "dir_acc": dir_acc}
    return float(final), diag

# confidence calculation combining CV, dir_prob, dir_acc, volatility penalty
def compute_confidence(diag, recent_volatility):
    try:
        base = 60.0
        models = diag.get("models", {})
        preds = [v.get("pred") for v in models.values() if isinstance(v, dict) and v.get("pred") is not None]
        if len(preds) >= 2:
            arr = np.array(preds)
            cv = (np.std(arr) / (np.mean(np.abs(arr)) + 1e-9))
            if cv < 0.03: base += 12
            elif cv < 0.07: base += 6
            elif cv < 0.15: base += 2
            elif cv < 0.25: base -= 6
            else: base -= 12
        dir_prob = diag.get("dir_prob")
        if dir_prob is not None:
            base = base*0.6 + dir_prob*100*0.4
        dir_acc = diag.get("dir_acc")
        if dir_acc is not None:
            base += (dir_acc - 0.5) * 40.0
        if recent_volatility is not None:
            base -= min(30.0, recent_volatility * 200.0)
        raw = int(max(10, min(95, round(base))))
        return raw
    except Exception:
        return 40

# fundamentals scoring (same as previous)
def fundamentals_score(profile, yf_info=None):
    def safef(x):
        try: return float(x)
        except: return None
    roe = safef(profile.get("returnOnEquity") if isinstance(profile, dict) else None)
    debt = safef(profile.get("debtToEquity") if isinstance(profile, dict) else None)
    pe = safef(profile.get("priceEarningsRatio") if isinstance(profile, dict) else None)
    rev_growth = safef(profile.get("revenueGrowth") if isinstance(profile, dict) else None)
    if yf_info and isinstance(yf_info, dict):
        roe = roe or safef(yf_info.get("returnOnEquity") or yf_info.get("roe"))
        debt = debt or safef(yf_info.get("debtToEquity"))
        pe = pe or safef(yf_info.get("trailingPE") or yf_info.get("forwardPE"))
        rev_growth = rev_growth or safef(yf_info.get("revenueGrowth"))
    def score_positive(x, good=0.06, great=0.18):
        if x is None: return None
        if x >= great: return 100
        if x <= good: return int(50 + 50 * (x - good) / max(1e-9, (great - good)))
        return int(50 + 50 * (x - good) / max(1e-9, (great - good)))
    def score_inverse(x, bad=2.0, good=0.5):
        if x is None: return None
        if x <= good: return 100
        if x >= bad: return 0
        return int(100 * (bad - x) / max(1e-9, (bad - good)))
    scores = {}
    scores["rev_growth"] = score_positive(rev_growth, 0.03, 0.15)
    scores["roe"] = score_positive(roe, 0.06, 0.18)
    scores["debt_equity"] = score_inverse(debt, 2.0, 0.5)
    if pe is None: scores["pe"] = None
    else:
        if pe <= 0: scores["pe"] = 50
        else: scores["pe"] = int(max(0, min(100, int(100 * (1 - (pe / 30.0))))))
    weights = {"rev_growth":0.25, "roe":0.35, "debt_equity":0.25, "pe":0.15}
    total=0.0; wsum=0.0
    for k,w in weights.items():
        v = scores.get(k)
        if v is None: continue
        total += v*w; wsum += w
    if wsum == 0: return 50, {k:(scores.get(k) if scores.get(k) is not None else "NA") for k in scores}
    final = int(round(total / wsum)); final = max(0, min(100, final))
    return final, {k:(scores.get(k) if scores.get(k) is not None else "NA") for k in scores}

# screeners (including new ones)
def momentum_screener(df):
    try:
        rsi = df['rsi14'].iloc[-1]; macd = df['macd'].iloc[-1]; ret7 = df['Close'].pct_change(7).iloc[-1]
        score = int(50 + 30*np.tanh(ret7*10) + (10 if macd>0 else -10) + (5 if rsi<70 and rsi>40 else -5))
        sig = "Strong" if score >= 65 else ("Weak" if score < 45 else "Neutral")
        return {"score": max(0,min(100,score)), "signal": sig}
    except Exception:
        return {"score":50,"signal":"Neutral"}

def supertrend_screener(df):
    try:
        dir_ = df['supertrend_dir'].iloc[-1] if 'supertrend_dir' in df.columns else 0
        sig = "Bull" if dir_==1 else ("Bear" if dir_==-1 else "Neutral")
        score = 70 if dir_==1 else 30 if dir_==-1 else 50
        return {"score":score,"signal":sig,"dir":int(dir_)}
    except Exception:
        return {"score":50,"signal":"Unknown"}

def stochrsi_meanrev_screener(df):
    try:
        k = df['stoch_k'].iloc[-1]; d = df['stoch_d'].iloc[-1]
        sig = "MeanRevBuy" if k<0.2 and d<0.2 else ("Overbought" if k>0.8 and d>0.8 else "Neutral")
        score = 70 if sig=="MeanRevBuy" else 30 if sig=="Overbought" else 50
        return {"score":score,"signal":sig,"k":float(k),"d":float(d)}
    except Exception:
        return {"score":50,"signal":"Neutral"}

def cmf_liquidity_screener(df):
    try:
        cmf = df['cmf20'].iloc[-1] if 'cmf20' in df.columns else 0.0
        avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1]) if 'Volume' in df.columns else 0.0
        score = int(max(0,min(100, (0.5+cmf/2.0)*100 )))
        sig = "GoodLiquidity" if avg_vol>100000 and cmf>0 else ("Weak" if cmf<0 else "Moderate")
        return {"score":score,"signal":sig,"cmf":float(cmf),"avg_vol":avg_vol}
    except Exception:
        return {"score":50,"signal":"Unknown","cmf":None}

# mapping
hmap = {"3 days":3,"15 days":15,"1 month":22,"3 months":66,"6 months":132,"1 year":260}

# run
if run:
    if not ticker:
        st.error("Please enter a ticker in the sidebar."); st.stop()
    with st.spinner("Fetching data and computing..."):
        hist, used_symbol, err = fetch_history_try(ticker, period='5y')
        if hist.empty:
            st.error(f"Failed to fetch historical data. Error: {err}"); st.stop()
        hist = compute_indicators(hist)
        current_price = float(hist['Close'].iloc[-1])
        profile = fetch_profile_fmp(ticker, fmp_key)
        yf_info, yf_source = fetch_yf_info(ticker)
        fund_score, fund_parts = fundamentals_score(profile, yf_info)
        sent_score, sent_label = sentiment_score(profile.get("description","") if isinstance(profile, dict) else "")
        models = load_or_train_models(ticker, hist, prefer_saved=use_saved_models)
        mom = momentum_screener(hist); fund_scr = {"score":fund_score,"signal":"Good" if fund_score>=60 else ("Poor" if fund_score<45 else "Moderate")}
        vol_scr = {"score": int(max(0,min(100,100 - hist['vol30'].iloc[-1]*400))) if 'vol30' in hist.columns else 50, "signal":"Low" if hist['vol30'].iloc[-1]<0.02 else "High"}
        liq_scr = {"score":50,"signal":"Moderate"}
        super_scr = supertrend_screener(hist); stochr = stochrsi_meanrev_screener(hist); cmf_scr = cmf_liquidity_screener(hist)
        comb = {"score": int((mom["score"]*0.25 + fund_scr["score"]*0.35 + vol_scr["score"]*0.15 + liq_scr["score"]*0.10) ), "signal":"Buy" if fund_scr["score"]>=60 and mom["score"]>=55 else "Hold"}
        results = {}; diags = {}; backtests = {}
        for h in horizons:
            days = hmap.get(h,22)
            pred_ret, diag = predict_horizon(models, hist, days)
            pred_ret = float(max(-0.95, min(0.95, pred_ret)))
            pred_price = current_price * (1 + pred_ret)
            q_low, q_high = diag.get("quantiles", (None, None))
            if q_low is not None and q_high is not None:
                low_p = current_price * (1 + q_low); high_p = current_price * (1 + q_high)
            else:
                vol30 = float(hist['vol30'].iloc[-1]) if 'vol30' in hist.columns else 0.02
                buffer = max(0.01, min(0.35, vol30 * 2.5 * min(1.0, days/22.0)))
                low_p = current_price * (1 + pred_ret - buffer); high_p = current_price * (1 + pred_ret + buffer)
            low_p = float(max(0.01, low_p)); high_p = float(max(low_p*1.0001, high_p))
            conf = compute_confidence(diag, float(hist['vol30'].iloc[-1]) if 'vol30' in hist.columns else None)
            results[h] = {"pred_price":round(pred_price,4), "pred_return":pred_ret, "low_price": round(low_p,4), "high_price": round(high_p,4), "confidence": int(conf)}
            diags[h] = diag
            # backtest on recent window if possible
            X,y,df = make_X_y(hist, days)
            if X is not None and len(X)>30:
                backtests[h] = backtest_metrics(models, X, y)
        # UI outputs (keeps layout unchanged)
    st.markdown("<div style='padding:10px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04)'><h2>Predictions</h2></div>", unsafe_allow_html=True)
    left, right = st.columns([2,3])
    with left:
        st.markdown("<div style='font-size:14px;color:gray'>Current price</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:800;font-size:22px'>₹{:.4f}</div>".format(current_price), unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        for h in horizons:
            r = results[h]; predp = r["pred_price"]; conf = r["confidence"]
            implied_pct = (predp/current_price - 1) * 100
            arrow = "▲" if implied_pct > 0 else ("▼" if implied_pct < 0 else "—")
            color = "green" if implied_pct > 0 else ("red" if implied_pct < 0 else "gray")
            interval_html = "<div style='margin-top:6px;color:gray'>Range: ₹{:.2f} — ₹{:.2f}</div>".format(r["low_price"], r["high_price"])
            badge_style = "background:#e6f4ea;color:#059669;padding:6px 10px;border-radius:999px;font-weight:700;" if conf>=70 else ("background:#fff4e6;color:#b45309;padding:6px 10px;border-radius:999px;font-weight:700;" if conf>=50 else "background:#fff1f2;color:#b91c1c;padding:6px 10px;border-radius:999px;font-weight:700;")
            html = ("<div style='padding:10px;border-radius:8px;margin-bottom:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);'>"
                    "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    "<div><strong>{}</strong><div style='font-size:18px;font-weight:700;margin-top:6px;'>₹{:.4f}</div><div style='color:gray'>Predicted price</div></div>"
                    "<div style='text-align:right;'><div style='{}'>Confidence {}%</div></div></div>'"
                    "<div style='margin-top:6px;color:{};font-weight:700'>{} {:+.2f}%</div>{}</div>").format(h, predp, badge_style, conf, color, arrow, implied_pct, interval_html)
            st.markdown(html, unsafe_allow_html=True)
    with right:
        st.markdown(("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);'>"
                     "<strong>Fundamentals score</strong><div style='font-size:20px;font-weight:700;margin-top:6px'>{}/100</div>"
                     "<div style='color:gray'>ROE, Rev growth, D/E, PE</div></div>").format(fund_score), unsafe_allow_html=True)
        st.markdown(("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'>"
                     "<strong>Sentiment</strong><div style='font-size:16px;font-weight:700;margin-top:6px'>{} ({}/100)</div></div>").format(sent_label, sent_score), unsafe_allow_html=True)
        st.markdown("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'><strong>Company description</strong><div style='color:gray;margin-top:6px'>Full description (if available)</div>", unsafe_allow_html=True)
        desc = None
        if isinstance(profile, dict):
            desc = profile.get("description") or profile.get("longBusinessSummary") or None
        if not desc and isinstance(yf_info, dict):
            desc = yf_info.get("longBusinessSummary") or yf_info.get("shortBusinessSummary") or None
        if desc:
            try:
                st.write(desc)
            except Exception:
                st.write(str(desc)[:4000])
        else:
            st.write("Not available via API.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'><strong>Screeners</strong>", unsafe_allow_html=True)
        st.write(f"Momentum: Score {mom['score']} — Signal: {mom['signal']}")
        st.write(f"SuperTrend: Score {super_scr['score']} — Signal: {super_scr['signal']}")
        st.write(f"StochRSI mean-reversion: Score {stochr['score']} — Signal: {stochr['signal']}")
        st.write(f"CMF Liquidity: Score {cmf_scr['score']} — Signal: {cmf_scr['signal']}")
        st.write(f"Combined (multi-factor): Score {comb['score']} — Signal: {comb['signal']}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'><strong>Recommendation</strong>", unsafe_allow_html=True)
        for h in horizons:
            r = results[h]; conf = r["confidence"]; predp = r["pred_price"]
            implied = (predp/current_price - 1)*100; baseline = comb["signal"]
            if baseline == "Buy" and implied > 3 and conf >= 50: verdict = "Buy"
            elif baseline == "Avoid" or implied < -3: verdict = "Avoid"
            elif conf < 35: verdict = "Defer"
            else: verdict = "Hold"
            atr = hist['atr14'].iloc[-1] if 'atr14' in hist.columns else 0.0
            vol_buffer = min(0.15, max(0.01, atr / max(1e-3, current_price)))
            buy_around = current_price * (1 - vol_buffer); stop_loss = buy_around * (1 - vol_buffer * 0.8)
            reasons = []
            if fund_score >= 60: reasons.append("Fundamentals supportive.") 
            else: reasons.append("Fundamentals modest.")
            if sent_label == "Positive": reasons.append("Positive sentiment.")
            if conf >= 65: reasons.append("Models aligned.") 
            else: reasons.append("Lower confidence.")
            para = ("Horizon {h}: {verdict}. Target ₹{target:.2f} ({implied:+.2f}%). Buy-around: ₹{b:.2f} — Stop-loss: ₹{s:.2f}. Reason: {reason} Confidence: {c}%").format(
                h=h, verdict=verdict, target=predp, implied=implied, b=buy_around, s=stop_loss, reason=" ".join(reasons), c=conf
            )
            st.write(para)
        st.markdown("</div>", unsafe_allow_html=True)
        if show_debug:
            st.write("Diags:", diags)
            st.write("Backtests:", backtests)
            st.write("Models info:", models.get("info", {}))
else:
    st.info("Enter ticker and click Run Aries Predicts")
