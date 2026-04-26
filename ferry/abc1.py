import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pipeline import run_pipeline

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Toronto Island Ferry — Demand Forecasting",
    page_icon  = "⛴️",
    layout     = "wide")

st.title("⛴️ Toronto Island Ferry — Demand Forecasting Dashboard")
st.markdown("**Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support**")
st.divider()

# ══════════════════════════════════════════════════════
# LOAD SAVED MODELS (from pkl files — no retraining)
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, 'xgb_model.pkl'),  'rb') as f: xgb      = pickle.load(f)
    with open(os.path.join(base_dir, 'rf_model.pkl'),   'rb') as f: rf       = pickle.load(f)
    with open(os.path.join(base_dir, 'gb_model.pkl'),   'rb') as f: gb       = pickle.load(f)
    with open(os.path.join(base_dir, 'lr_model.pkl'),   'rb') as f: lr       = pickle.load(f)
    with open(os.path.join(base_dir, 'gb_low.pkl'),     'rb') as f: gb_low   = pickle.load(f)
    with open(os.path.join(base_dir, 'gb_high.pkl'),    'rb') as f: gb_high  = pickle.load(f)
    with open(os.path.join(base_dir, 'feat_cols.pkl'),  'rb') as f: feat_cols= pickle.load(f)
    with open(os.path.join(base_dir, 'intervals.json'), 'r')  as f: intervals= json.load(f)

    return xgb, rf, gb, lr, gb_low, gb_high, feat_cols, intervals

# ══════════════════════════════════════════════════════
# LOAD & PREPROCESS DATA (pipeline.py — no training)
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'Toronto Island Ferry Tickets.csv')
    df, train, test = run_pipeline(csv_path)
    return df, train, test

def generate_future(df, future_date):
    future_date = pd.Timestamp(future_date)
    mask = (
        (df['Timestamp'].dt.month     == future_date.month) &
        (df['Timestamp'].dt.dayofweek == future_date.dayofweek) &
        (df['Timestamp'].dt.hour      == future_date.hour)
    )
    ref = df[mask]
    if len(ref) == 0:
        ref = df[(df['Timestamp'].dt.month == future_date.month) &
                 (df['Timestamp'].dt.hour  == future_date.hour)]
    if len(ref) == 0:
        ref = df[df['Timestamp'].dt.month == future_date.month]
    if len(ref) == 0:
        ref = df

    avg_sales = ref['Sales Count'].mean()
    std_sales = ref['Sales Count'].std()
    max_sales = ref['Sales Count'].max()

    row = {
        'Sales_lag1':           avg_sales,
        'Sales_lag2':           avg_sales,
        'Sales_lag4':           avg_sales,
        'Sales_lag8':           avg_sales,
        'Sales_roll_mean_4':    avg_sales,
        'Sales_roll_mean_8':    avg_sales,
        'Sales_roll_std_4':     std_sales,
        'Sales_roll_max_4':     max_sales,
        'Sales_same_yesterday': avg_sales,
        'Sales_same_lastweek':  avg_sales,
        'Hour':       future_date.hour,
        'DayOfWeek':  future_date.dayofweek,
        'IsWeekend':  int(future_date.dayofweek >= 5),
        'Month':      future_date.month,
        'Hour_sin':   np.sin(2 * np.pi * future_date.hour  / 24),
        'Hour_cos':   np.cos(2 * np.pi * future_date.hour  / 24),
        'Month_sin':  np.sin(2 * np.pi * future_date.month / 12),
        'Month_cos':  np.cos(2 * np.pi * future_date.month / 12),
    }
    return pd.DataFrame([row])

# ── Load everything ────────────────────────────────────
with st.spinner("⚡ Loading saved models ..."):
    xgb, rf, gb, lr, gb_low, gb_high, feat_cols, intervals = load_models()
    lower_q = intervals['lower_q']
    upper_q = intervals['upper_q']

with st.spinner("📦 Preparing data ..."):
    df, train, test = load_data()

X_test      = test[feat_cols]
y_test      = test['Sales Count']
actual      = y_test.values
mean_demand = y_test.mean()

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "Gradient Boosting", "Linear Regression"]
)

horizon_choice = st.sidebar.selectbox(
    "Forecast Horizon",
    ["15 minutes (lag1)", "30 minutes (lag2)", "1 hour (lag4)", "2 hours (lag8)"]
)

show_interval = st.sidebar.checkbox("Show Confidence Interval", value=True)

start_date = st.sidebar.date_input("Start Date", value=datetime.date(2024, 1, 1))
end_date   = st.sidebar.date_input("End Date",   value=datetime.date(2024, 1, 7))

if start_date > datetime.date(2025, 12, 21):
    st.sidebar.warning("⚠️ Predictions beyond 2025 are estimates based on historical patterns.")
    st.sidebar.info("💡 Keep future range within 7 days for best performance.")

st.sidebar.divider()
st.sidebar.markdown("**Dataset Info**")
st.sidebar.write(f"Total records : {len(df):,}")
st.sidebar.write(f"Train rows    : {len(train):,}")
st.sidebar.write(f"Test rows     : {len(test):,}")
st.sidebar.write(f"Date range    : {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")

# ══════════════════════════════════════════════════════
# GET PREDICTIONS FROM LOADED MODEL
# ══════════════════════════════════════════════════════
if model_choice == "XGBoost":
    preds       = np.clip(xgb.predict(X_test), 0, None)
    p_lower     = np.clip(preds + lower_q, 0, None)
    p_upper     = preds + upper_q
    model_color = 'purple'

elif model_choice == "Random Forest":
    preds       = np.clip(rf.predict(X_test), 0, None)
    p_lower     = np.clip(preds + lower_q, 0, None)
    p_upper     = preds + upper_q
    model_color = 'tomato'

elif model_choice == "Gradient Boosting":
    preds       = np.clip(gb.predict(X_test), 0, None)
    p_lower     = np.clip(gb_low.predict(X_test), 0, None)
    p_upper     = gb_high.predict(X_test)
    model_color = 'orange'

else:
    preds       = np.clip(lr.predict(X_test), 0, None)
    p_lower     = np.clip(preds - preds.std(), 0, None)
    p_upper     = preds + preds.std()
    model_color = 'steelblue'

horizon_map = {
    "15 minutes (lag1)": "Sales_lag1",
    "30 minutes (lag2)": "Sales_lag2",
    "1 hour (lag4)":     "Sales_lag4",
    "2 hours (lag8)":    "Sales_lag8"
}
naive_col  = horizon_map[horizon_choice]
naive_pred = test[naive_col].values

# ══════════════════════════════════════════════════════
# KPI METRICS ROW
# ══════════════════════════════════════════════════════
mae          = mean_absolute_error(actual, preds)
rmse         = np.sqrt(mean_squared_error(actual, preds))
forecast_acc = round((1 - mae / mean_demand) * 100, 2)
coverage     = np.mean((actual >= p_lower) & (actual <= p_upper)) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Selected Model",    model_choice)
c2.metric("MAE",               f"{mae:.2f}")
c3.metric("RMSE",              f"{rmse:.2f}")
c4.metric("Forecast Accuracy", f"{forecast_acc}%")
c5.metric("95% CI Coverage",   f"{coverage:.1f}%")

st.divider()

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Chart",
    "🔁 Model Comparison",
    "🌡️ Feature Importance",
    "📊 EDA Overview",
    "🚨 Operations Alert"
])

# ── TAB 1: FORECAST CHART ─────────────────────────────
with tab1:
    start_dt  = pd.Timestamp(start_date)
    end_dt    = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    is_future = start_date > test['Timestamp'].max().date()

    if is_future:
        st.subheader(f"🔮 Future Forecast — {model_choice}")
        st.info("Showing predicted demand for future dates based on historical patterns.")

        max_end = start_dt + pd.Timedelta(days=7)
        if end_dt > max_end:
            end_dt = max_end
            st.warning("⚠️ Future forecast limited to 7 days max for performance.")

        horizon_freq_map = {
            "15 minutes (lag1)": "15min",
            "30 minutes (lag2)": "30min",
            "1 hour (lag4)":     "h",
            "2 hours (lag8)":    "2h"
        }
        freq = horizon_freq_map[horizon_choice]
        future_timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)

        if len(future_timestamps) == 0:
            st.error("No timestamps generated. Please make sure End Date is after Start Date.")
        else:
            future_rows = pd.concat(
                [generate_future(df, ts) for ts in future_timestamps],
                ignore_index=True
            )

            if model_choice == "XGBoost":
                future_preds = np.clip(xgb.predict(future_rows[feat_cols]), 0, None)
            elif model_choice == "Random Forest":
                future_preds = np.clip(rf.predict(future_rows[feat_cols]), 0, None)
            elif model_choice == "Gradient Boosting":
                future_preds = np.clip(gb.predict(future_rows[feat_cols]), 0, None)
            else:
                future_preds = np.clip(lr.predict(future_rows[feat_cols]), 0, None)

            future_lower = np.clip(future_preds + lower_q, 0, None)
            future_upper = future_preds + upper_q

            # Crowd Level Indicator
            peak_pred = future_preds.max()
            st.markdown("### 🚦 Crowd Level Indicator")
            ca, cb, cc = st.columns(3)
            ca.metric("Peak Predicted Demand", f"{peak_pred:.0f} tickets/interval")
            cb.metric("Avg Predicted Demand",  f"{future_preds.mean():.1f} tickets/interval")
            cc.metric("Forecast Period",        f"{(end_dt - start_dt).days} days")

            if peak_pred > 150:
                st.error("🔴 HIGH DEMAND — Deploy extra ferries & additional staff")
            elif peak_pred > 80:
                st.warning("🟡 MODERATE DEMAND — Monitor closely, standby extra ferry")
            else:
                st.success("🟢 LOW DEMAND — Normal operations sufficient")

            # Forecast Chart
            fig, ax = plt.subplots(figsize=(14, 5))
            if show_interval:
                ax.fill_between(future_timestamps, future_lower, future_upper,
                                alpha=0.25, color=model_color, label='95% Confidence Band')
            ax.plot(future_timestamps, future_preds, color=model_color,
                    lw=2, label=f'{model_choice} Forecast')
            ax.set_ylabel('Predicted Sales Count')
            ax.legend(); ax.grid(alpha=0.3)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Daily Summary Table
            st.markdown("### 📋 Daily Summary Table")
            future_df = pd.DataFrame({
                'Timestamp': future_timestamps,
                'Predicted': future_preds,
                'Lower_CI':  future_lower,
                'Upper_CI':  future_upper
            })
            future_df['Date'] = future_df['Timestamp'].dt.date
            daily_summary = future_df.groupby('Date').agg(
                Total_Predicted = ('Predicted', 'sum'),
                Avg_Predicted   = ('Predicted', 'mean'),
                Peak_Predicted  = ('Predicted', 'max'),
                Min_Predicted   = ('Predicted', 'min')
            ).round(1).reset_index()
            st.dataframe(daily_summary, use_container_width=True)

            # Download Forecast CSV
            st.download_button(
                label     = "📥 Download Forecast as CSV",
                data      = future_df.to_csv(index=False),
                file_name = f"ferry_forecast_{start_date}_{end_date}.csv",
                mime      = "text/csv"
            )

            # Weekly Heatmap
            if len(future_timestamps) >= 24:
                st.markdown("### 🗓️ Hourly Demand Heatmap")
                future_df['Hour'] = future_df['Timestamp'].dt.hour
                future_df['Day']  = future_df['Timestamp'].dt.strftime('%a %d')
                try:
                    pivot = future_df.pivot_table(values='Predicted', index='Hour',
                                                  columns='Day', aggfunc='mean')
                    fig_hm, ax_hm = plt.subplots(figsize=(14, 6))
                    im = ax_hm.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                                      interpolation='nearest')
                    ax_hm.set_yticks(range(len(pivot.index)))
                    ax_hm.set_yticklabels(pivot.index, fontsize=8)
                    ax_hm.set_xticks(range(len(pivot.columns)))
                    ax_hm.set_xticklabels(pivot.columns, fontsize=9, rotation=30)
                    ax_hm.set_ylabel('Hour of Day')
                    ax_hm.set_title('Predicted Demand Heatmap — Hour × Day', fontweight='bold')
                    plt.colorbar(im, ax=ax_hm, label='Predicted Sales')
                    plt.tight_layout()
                    st.pyplot(fig_hm)
                    plt.close()
                except:
                    pass

    else:
        st.subheader(f"{model_choice} — Forecast vs Actual")

        w     = test[(test['Timestamp'] >= start_dt) & (test['Timestamp'] < end_dt)]
        w_idx = w.index - test.index[0]

        if len(w) == 0:
            st.error("No data found for selected date range. Please select a date within 2024–2025.")
        else:
            # Crowd Level Indicator
            peak_actual = w['Sales Count'].max()
            st.markdown("### 🚦 Crowd Level Indicator")
            ca, cb, cc = st.columns(3)
            ca.metric("Peak Actual Demand", f"{peak_actual:.0f} tickets/interval")
            cb.metric("Avg Actual Demand",  f"{w['Sales Count'].mean():.1f} tickets/interval")
            cc.metric("Selected Period",     f"{(end_dt - start_dt).days} days")

            if peak_actual > 150:
                st.error("🔴 HIGH DEMAND period selected")
            elif peak_actual > 80:
                st.warning("🟡 MODERATE DEMAND period selected")
            else:
                st.success("🟢 LOW DEMAND period selected")

            fig, ax = plt.subplots(figsize=(14, 5))
            if show_interval:
                ax.fill_between(w['Timestamp'], p_lower[w_idx], p_upper[w_idx],
                                alpha=0.25, color=model_color, label='95% Confidence Band')
            ax.plot(w['Timestamp'], w['Sales Count'], color='steelblue', lw=2,   label='Actual')
            ax.plot(w['Timestamp'], preds[w_idx],     color=model_color,  lw=1.5, linestyle='--', label=model_choice)
            ax.set_ylabel('Sales Count')
            ax.legend(); ax.grid(alpha=0.3)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.subheader(f"Naive ({horizon_choice}) vs {model_choice}")
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.plot(w['Timestamp'], w['Sales Count'],  color='steelblue', lw=2,   label='Actual')
            ax2.plot(w['Timestamp'], naive_pred[w_idx], color='gray',      lw=1.2, linestyle=':', label='Naive')
            ax2.plot(w['Timestamp'], preds[w_idx],      color=model_color,  lw=1.5, linestyle='--', label=model_choice)
            ax2.set_ylabel('Sales Count')
            ax2.legend(); ax2.grid(alpha=0.3)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Download historical forecast
            st.markdown("### 📥 Download Forecast")
            hist_df = pd.DataFrame({
                'Timestamp': w['Timestamp'].values,
                'Actual':    w['Sales Count'].values,
                'Predicted': preds[w_idx],
                'Lower_CI':  p_lower[w_idx],
                'Upper_CI':  p_upper[w_idx],
            })
            st.download_button(
                label     = "📥 Download as CSV",
                data      = hist_df.to_csv(index=False),
                file_name = f"ferry_forecast_{start_date}_{end_date}.csv",
                mime      = "text/csv"
            )

# ── TAB 2: MODEL COMPARISON ───────────────────────────
with tab2:
    st.subheader("All Models — Performance Comparison")

    xgb_preds = np.clip(xgb.predict(X_test), 0, None)
    rf_preds  = np.clip(rf.predict(X_test),  0, None)
    gb_preds  = np.clip(gb.predict(X_test),  0, None)
    lr_preds  = np.clip(lr.predict(X_test),  0, None)
    naive_p   = test['Sales_lag1'].values
    ma_p      = test['Sales_roll_mean_4'].values

    all_models = {
        'Naive':             naive_p,
        'Moving Avg':        ma_p,
        'Linear Regression': lr_preds,
        'Random Forest':     rf_preds,
        'Gradient Boosting': gb_preds,
        'XGBoost':           xgb_preds,
    }

    rows = []
    for name, pred in all_models.items():
        mae_  = mean_absolute_error(actual, pred)
        rmse_ = np.sqrt(mean_squared_error(actual, pred))
        acc_  = round((1 - mae_ / mean_demand) * 100, 2)
        rows.append({'Model': name,
                     'MAE':      round(mae_,  3),
                     'RMSE':     round(rmse_, 3),
                     'Accuracy': f"{acc_}%"})

    comp_df = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
    comp_df.index += 1
    st.dataframe(comp_df, use_container_width=True)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('MAE & RMSE — All Models', fontweight='bold')
    colors = ['gray','brown','steelblue','tomato','orange','purple']

    for ax, metric in zip(axes3, ['MAE','RMSE']):
        bars = ax.bar(comp_df['Model'], comp_df[metric],
                      color=colors[:len(comp_df)], alpha=0.85, edgecolor='white')
        best = comp_df[metric].idxmin() - 1
        bars[best].set_edgecolor('green')
        bars[best].set_linewidth(2.5)
        for bar, val in zip(bars, comp_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha='center', fontsize=8, fontweight='bold')
        ax.set_title(metric)
        ax.set_xticklabels(comp_df['Model'], rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    start_dt2 = pd.Timestamp(start_date)
    end_dt2   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    w2        = test[(test['Timestamp'] >= start_dt2) & (test['Timestamp'] < end_dt2)]
    w2_idx    = w2.index - test.index[0]

    if len(w2) > 0:
        st.subheader("All Models vs Actual — Selected Date Range")
        fig4, ax4 = plt.subplots(figsize=(14, 5))
        ax4.plot(w2['Timestamp'], w2['Sales Count'],  color='black',     lw=2,   label='Actual')
        ax4.plot(w2['Timestamp'], lr_preds[w2_idx],   color='steelblue', lw=1.2, linestyle='--', label='Linear Reg')
        ax4.plot(w2['Timestamp'], rf_preds[w2_idx],   color='tomato',    lw=1.2, linestyle='--', label='Random Forest')
        ax4.plot(w2['Timestamp'], gb_preds[w2_idx],   color='orange',    lw=1.2, linestyle='--', label='Gradient Boosting')
        ax4.plot(w2['Timestamp'], xgb_preds[w2_idx],  color='purple',    lw=1.5, linestyle='--', label='XGBoost')
        ax4.set_ylabel('Sales Count')
        ax4.legend(fontsize=9); ax4.grid(alpha=0.3)
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.subheader("Actual vs Predicted — XGBoost (Best Model)")
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    ax5.scatter(actual, xgb_preds, s=2, alpha=0.2, color='purple')
    max_val = max(actual.max(), xgb_preds.max())
    ax5.plot([0, max_val], [0, max_val], color='red', lw=1.5, linestyle='--', label='Perfect fit')
    ax5.set_xlabel('Actual'); ax5.set_ylabel('Predicted')
    ax5.set_title('XGBoost — Actual vs Predicted')
    ax5.legend(); ax5.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ── TAB 3: FEATURE IMPORTANCE ─────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest")
        fi_rf = pd.DataFrame({'Feature': feat_cols, 'Importance': rf.feature_importances_})
        fi_rf = fi_rf.sort_values('Importance', ascending=True)
        fig6, ax6 = plt.subplots(figsize=(7, 7))
        ax6.barh(fi_rf['Feature'], fi_rf['Importance'], color='tomato', alpha=0.85, edgecolor='white')
        ax6.set_xlabel('Importance'); ax6.grid(axis='x', alpha=0.3)
        ax6.set_title('Random Forest', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    with col2:
        st.subheader("Gradient Boosting")
        fi_gb = pd.DataFrame({'Feature': feat_cols, 'Importance': gb.feature_importances_})
        fi_gb = fi_gb.sort_values('Importance', ascending=True)
        fig7, ax7 = plt.subplots(figsize=(7, 7))
        ax7.barh(fi_gb['Feature'], fi_gb['Importance'], color='orange', alpha=0.85, edgecolor='white')
        ax7.set_xlabel('Importance'); ax7.grid(axis='x', alpha=0.3)
        ax7.set_title('Gradient Boosting', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    st.subheader("XGBoost")
    fi_xgb = pd.DataFrame({'Feature': feat_cols, 'Importance': xgb.feature_importances_})
    fi_xgb = fi_xgb.sort_values('Importance', ascending=True)
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    ax8.barh(fi_xgb['Feature'], fi_xgb['Importance'], color='purple', alpha=0.85, edgecolor='white')
    ax8.set_xlabel('Importance'); ax8.grid(axis='x', alpha=0.3)
    ax8.set_title('XGBoost — Feature Importance', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

# ── TAB 4: EDA OVERVIEW ───────────────────────────────
with tab4:
    st.subheader("EDA Overview")

    daily = df.groupby(df['Timestamp'].dt.date)[['Sales Count','Redemption Count']].sum().reset_index()
    daily.columns = ['Date','Sales','Redemp']
    daily['Date'] = pd.to_datetime(daily['Date'])

    fig9, ax9 = plt.subplots(figsize=(14, 4))
    ax9.fill_between(daily['Date'], daily['Sales'], alpha=0.2, color='steelblue')
    ax9.plot(daily['Date'], daily['Sales'],  color='steelblue', lw=0.6, label='Sales')
    ax9.plot(daily['Date'], daily['Redemp'], color='tomato',    lw=0.6, label='Redemptions')
    ax9.set_title('Full 10-Year Daily Demand Timeline', fontweight='bold')
    ax9.set_ylabel('Tickets / Day')
    ax9.legend(); ax9.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly = df.groupby('Month')['Sales Count'].mean()
        fig10, ax10 = plt.subplots(figsize=(5, 4))
        ax10.plot(range(1,13), monthly.values, color='steelblue', lw=2, marker='o', ms=5)
        ax10.set_xticks(range(1,13))
        ax10.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax10.set_title('Monthly Seasonality', fontweight='bold')
        ax10.set_ylabel('Avg Sales / Interval')
        ax10.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig10)
        plt.close()

    with col2:
        hourly = df.groupby('Hour')['Sales Count'].mean()
        fig11, ax11 = plt.subplots(figsize=(5, 4))
        ax11.bar(hourly.index, hourly.values, color='steelblue', alpha=0.8)
        ax11.set_title('Hourly Demand Pattern', fontweight='bold')
        ax11.set_xlabel('Hour of Day')
        ax11.set_ylabel('Avg Sales / Interval')
        ax11.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig11)
        plt.close()

    with col3:
        dow = df.groupby('DayOfWeek')['Sales Count'].mean()
        fig12, ax12 = plt.subplots(figsize=(5, 4))
        colors_dow = ['#FF9F43' if i >= 5 else 'steelblue' for i in range(7)]
        ax12.bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                 dow.values, color=colors_dow, alpha=0.85)
        ax12.set_title('Day-of-Week Pattern', fontweight='bold')
        ax12.set_ylabel('Avg Sales / Interval')
        ax12.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig12)
        plt.close()

# ── TAB 5: OPERATIONS ALERT ───────────────────────────
with tab5:
    st.subheader("🚨 Operations Alert Center")
    st.markdown("Real-time demand level monitoring and operational recommendations.")

    start_dt3 = pd.Timestamp(start_date)
    end_dt3   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    is_future3= start_date > test['Timestamp'].max().date()

    if is_future3:
        st.info("📅 Showing future demand alerts based on forecast.")
        future_ts3 = pd.date_range(
            start = start_dt3,
            end   = min(end_dt3, start_dt3 + pd.Timedelta(days=7)),
            freq  = '15min'
        )
        if len(future_ts3) > 0:
            future_rows3 = pd.concat(
                [generate_future(df, ts) for ts in future_ts3], ignore_index=True
            )
            future_preds3 = np.clip(xgb.predict(future_rows3[feat_cols]), 0, None)

            peak_val  = future_preds3.max()
            avg_val   = future_preds3.mean()
            peak_hour = int(future_ts3[future_preds3.argmax()].hour)

            a1, a2, a3 = st.columns(3)
            a1.metric("🔝 Peak Demand",    f"{peak_val:.0f} tickets/15min")
            a2.metric("📊 Average Demand", f"{avg_val:.1f} tickets/15min")
            a3.metric("🕐 Peak Hour",      f"{peak_hour}:00")

            st.markdown("### 🚦 Operational Status")
            if peak_val > 150:
                st.error("🔴 **HIGH DEMAND ALERT**\n- Deploy maximum ferry capacity\n- Call in extra staff\n- Open all ticket booths\n- Activate crowd management")
            elif peak_val > 80:
                st.warning("🟡 **MODERATE DEMAND**\n- Keep standby ferry ready\n- Monitor queue lengths\n- Normal staffing + 1 extra")
            else:
                st.success("🟢 **LOW DEMAND**\n- Normal operations\n- Standard staffing sufficient\n- Single ferry schedule")

            st.markdown("### 📋 Hourly Alert Schedule")
            alert_df = pd.DataFrame({
                'Time':      future_ts3.strftime('%Y-%m-%d %H:%M'),
                'Predicted': future_preds3.round(1),
            })
            alert_df['Status'] = alert_df['Predicted'].apply(
                lambda x: '🔴 HIGH'     if x > 150 else
                          '🟡 MODERATE' if x > 80  else
                          '🟢 LOW'
            )
            alert_df['Recommendation'] = alert_df['Predicted'].apply(
                lambda x: 'Extra ferries + full staff' if x > 150 else
                          'Standby ferry ready'         if x > 80  else
                          'Normal operations'
            )
            st.dataframe(alert_df, use_container_width=True)

            st.download_button(
                label     = "📥 Download Alert Schedule",
                data      = alert_df.to_csv(index=False),
                file_name = f"ferry_alerts_{start_date}_{end_date}.csv",
                mime      = "text/csv"
            )

    else:
        w3     = test[(test['Timestamp'] >= start_dt3) & (test['Timestamp'] < end_dt3)]
        w3_idx = w3.index - test.index[0]

        if len(w3) == 0:
            st.error("No data for selected range.")
        else:
            peak_val  = w3['Sales Count'].max()
            avg_val   = w3['Sales Count'].mean()
            peak_hour = int(w3.loc[w3['Sales Count'].idxmax(), 'Timestamp'].hour)

            a1, a2, a3 = st.columns(3)
            a1.metric("🔝 Peak Demand",    f"{peak_val:.0f} tickets/15min")
            a2.metric("📊 Average Demand", f"{avg_val:.1f} tickets/15min")
            a3.metric("🕐 Peak Hour",      f"{peak_hour}:00")

            st.markdown("### 🚦 Operational Status")
            if peak_val > 150:
                st.error("🔴 **HIGH DEMAND ALERT**\n- Deploy maximum ferry capacity\n- Call in extra staff\n- Open all ticket booths\n- Activate crowd management")
            elif peak_val > 80:
                st.warning("🟡 **MODERATE DEMAND**\n- Keep standby ferry ready\n- Monitor queue lengths\n- Normal staffing + 1 extra")
            else:
                st.success("🟢 **LOW DEMAND**\n- Normal operations\n- Standard staffing sufficient\n- Single ferry schedule")

            st.markdown("### 📋 Interval-level Alert Table")
            alert_df2 = pd.DataFrame({
                'Time':      w3['Timestamp'].dt.strftime('%Y-%m-%d %H:%M').values,
                'Actual':    w3['Sales Count'].values,
                'Predicted': preds[w3_idx].round(1),
            })
            alert_df2['Status'] = alert_df2['Actual'].apply(
                lambda x: '🔴 HIGH'     if x > 150 else
                          '🟡 MODERATE' if x > 80  else
                          '🟢 LOW'
            )
            alert_df2['Recommendation'] = alert_df2['Actual'].apply(
                lambda x: 'Extra ferries + full staff' if x > 150 else
                          'Standby ferry ready'         if x > 80  else
                          'Normal operations'
            )
            st.dataframe(alert_df2, use_container_width=True)

            st.download_button(
                label     = "📥 Download Alert Table",
                data      = alert_df2.to_csv(index=False),
                file_name = f"ferry_alerts_{start_date}_{end_date}.csv",
                mime      = "text/csv"
            )

st.divider()
st.caption("Toronto Island Ferry — Demand Forecasting System | Unified Mentor Project")
