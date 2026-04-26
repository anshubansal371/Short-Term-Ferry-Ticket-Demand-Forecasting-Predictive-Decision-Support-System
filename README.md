# Live Demo:- https://ferry-demand-forecasting.streamlit.app/

# Short-Term-Ferry-Ticket-Demand-Forecasting-Predictive-Decision-Support-System
End-to-end ML project forecasting Toronto Island Ferry ticket demand at 15-min to 2-hour horizons using 10 years of data. Covers EDA, preprocessing, feature engineering, and models including XGBoost (MAE: 3.955, ~92% accuracy). Deployed as an interactive Streamlit dashboard with future forecasting and operations alerts

# ⛴️ Toronto Island Ferry — Short-Term Demand Forecasting

> End-to-end machine learning project forecasting Toronto Island Ferry ticket demand at 15-minute to 2-hour horizons using 10 years of data. Covers EDA, preprocessing, feature engineering, and models including Linear Regression (MAE: 4.393, ~84% accuracy). Deployed as an interactive Streamlit dashboard with future forecasting and operations alerts.-
>
>  **Linear Regression outperformed XGBoost** on this test period — 
  suggesting ferry demand has strong linear temporal dependencies 
  that simpler models can capture effectively.

---

## 📌 Project Overview

Toronto Island Park receives millions of visitors annually, all of whom depend on ferry services. Despite having over a decade of high-frequency ticket data, ferry operations lacked short-term demand forecasts — leading to reactive scheduling, congestion, and service inefficiencies.

This project transitions ferry operations from **reactive analytics to predictive intelligence** by building a complete machine learning forecasting pipeline that predicts demand up to 2 hours ahead at 15-minute resolution.

---

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Toronto Government — Parks, Forestry & Recreation |
| Records | 261,538 |
| Interval | 15 minutes |
| Date Range | May 2015 — December 2025 |
| Columns | Timestamp, Sales Count, Redemption Count |

> **Note:** Dataset not included in this repo due to file size.
---

## 🏆 Model Results

| Rank | Model | MAE | RMSE | Accuracy |
|------|-------|-----|------|----------|
| 🥇 1 | Linear Regression | 4.393 | 6.933 | 83.96% |
| 🥈 2 | Gradient Boosting | 4.431 | 6.877 | 83.82% |
| 🥉 3 | XGBoost | 4.562 | 6.898 | 83.34% |
| 4 | Random Forest | 4.981 | 7.235 | 81.81% |
| 5 | Moving Average | 5.368 | 8.657 | 80.4% |
| 6 | Naive Forecast | 7.824 | 13.259 | 71.43% |

> Accuracy = (1 - MAE / mean_demand) × 100 | mean_demand = 49.6 tickets/interval

---


| Library | Purpose |
|---------|---------|
| Pandas | Data loading, manipulation, resampling |
| NumPy | Numerical operations, cyclical encoding |
| Matplotlib | All visualisation plots |
| Scikit-learn | Linear Regression, Random Forest, Gradient Boosting, metrics |
| XGBoost | Best performing forecasting model |
| Streamlit | Interactive dashboard deployment |
| Pickle | Model serialisation and loading |
| SciPy | Statistical analysis, Z-score spike detection |

---

## 🔑 Key Features

### 📈 Feature Engineering
- **Lag features** — t-1, t-2, t-4, t-8 (15min to 2hr lookback)
- **Rolling statistics** — mean, std, max over 1hr and 2hr windows
- **Same-time lags** — yesterday (lag-96) and last week (lag-672)
- **Cyclical encoding** — sin/cos for hour, month, day-of-week
- **Temporal features** — IsWeekend, Month, Hour, DayOfWeek

### 🎯 Uncertainty Quantification
- **Conformal prediction** for Random Forest — 93.7% coverage
- **Quantile regression** for Gradient Boosting — 98.1% coverage
- Adaptive confidence bands that widen during peak demand hours

### 🖥️ Streamlit Dashboard
- Model selector — XGBoost, RF, GB, Linear Regression
- Horizon selector — 15min, 30min, 1hr, 2hr
- Any date picker — past and future dates supported
- 🚦 Crowd level indicator — 🔴 High / 🟡 Moderate / 🟢 Low
- Operations alert center with actionable recommendations
- Download forecast and alert tables as CSV
- Hourly demand heatmap for selected week

---

## ▶️ How to Run

### Step 1 — Run Day 1–6 in Google Colab

```python
# Upload CSV in Colab
from google.colab import files
uploaded = files.upload()

# Install dependencies
!pip install xgboost statsmodels -q

Then run all the collab cells in sequential order. 

### Step 2 — Save model weights from Colab

```python
import pickle, json

with open('xgb_model.pkl', 'wb') as f: pickle.dump(xgb, f)
with open('rf_model.pkl',  'wb') as f: pickle.dump(rf,  f)
with open('gb_model.pkl',  'wb') as f: pickle.dump(gb,  f)
with open('lr_model.pkl',  'wb') as f: pickle.dump(lr,  f)
with open('gb_low.pkl',    'wb') as f: pickle.dump(gb_lower_model, f)
with open('gb_high.pkl',   'wb') as f: pickle.dump(gb_upper_model, f)
with open('feat_cols.pkl', 'wb') as f: pickle.dump(feat_cols, f)

with open('intervals.json', 'w') as f:
    json.dump({'lower_q': float(lower_q), 'upper_q': float(upper_q)}, f)

# Download all files to your PC
from google.colab import files
for f in ['xgb_model.pkl','rf_model.pkl','gb_model.pkl','lr_model.pkl',
          'gb_low.pkl','gb_high.pkl','feat_cols.pkl','intervals.json']:
    files.download(f)
```

### Step 3 — Run Streamlit Dashboard in VS Code

```bash
# Install dependencies
pip install -r streamlit_app/requirements.txt

# Place all .pkl and .json files in streamlit_app/ folder
# Place CSV in data/ folder
# Update CSV path in pipeline.py

# Run the app
streamlit run abc.py/abc1.py
```

---

## 📋 Requirements

```
pandas
numpy
matplotlib
scikit-learn
xgboost
streamlit
scipy
```

---

## 🔍 Key Observations

- **16× seasonal swing** — July/August peaks at 120 tickets/interval vs January at 7.6
- **Lag-1 dominates** — most recent value explains ~35-40% of model variance
- **Weekend premium** — Saturday/Sunday ~30% higher than weekdays
- **COVID dip** — 2020 shows ~45% demand suppression, full recovery by 2022
- **Sales → Redemption lag** — peak cross-correlation at 1-2 intervals (15-30 min)
- **MAPE unreliable** — 7.9% zero-demand intervals cause division errors; use MAE/RMSE

---
