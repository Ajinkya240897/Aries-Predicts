Aries Predicts - Advanced
-------------------------
This package contains:
- app.py : Streamlit app with advanced indicators, classifier, bootstrap calibration, backtest metrics.
- requirements.txt : recommended packages (use Python 3.10 or 3.11 on Streamlit Cloud).

Notes:
- No heavy binaries included (no xgboost/lightgbm) to keep the app deployable on Streamlit Cloud.
- For best UX, pretrain models and place joblib files under models/general/ (rf.joblib, et.joblib, gbr.joblib, hgb.joblib, meta.joblib, clf.joblib).
