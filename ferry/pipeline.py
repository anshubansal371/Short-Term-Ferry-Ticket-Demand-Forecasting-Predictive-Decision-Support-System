# Only preprocessing & feature engineering — NO model training
# Models are loaded from saved .pkl files

import pandas as pd
import numpy as np

def run_pipeline(csv_path='Toronto_Island_Ferry_Tickets.csv'):

    # ── DAY 1: Load ──────────────────────────────────
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # ── DAY 2: Fill Missing Intervals ────────────────
    full_index = pd.date_range(start=df['Timestamp'].min(),
                               end=df['Timestamp'].max(), freq='15min')
    df = df.set_index('Timestamp').reindex(full_index)
    df.index.name = 'Timestamp'

    df['Sales Count']      = df['Sales Count'].interpolate(method='linear', limit=4).ffill().fillna(0)
    df['Redemption Count'] = df['Redemption Count'].interpolate(method='linear', limit=4).ffill().fillna(0)
    df['Sales Count']      = df['Sales Count'].clip(lower=0).round().astype(int)
    df['Redemption Count'] = df['Redemption Count'].clip(lower=0).round().astype(int)

    # IQR Capping
    Q1, Q3    = df['Sales Count'].quantile(0.25), df['Sales Count'].quantile(0.75)
    upper_cap = Q3 + 1.5 * (Q3 - Q1)
    df['Sales Count'] = df['Sales Count'].clip(upper=upper_cap)

    # Smoothing
    df['Sales_smooth_1h'] = df['Sales Count'].rolling(4, center=True, min_periods=1).mean()
    df['Sales_smooth_2h'] = df['Sales Count'].rolling(8, center=True, min_periods=1).mean()

    df = df.reset_index()

    # ── DAY 3: Temporal Features ─────────────────────
    df['Year']      = df['Timestamp'].dt.year
    df['Month']     = df['Timestamp'].dt.month
    df['Hour']      = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Hour_sin']  = np.sin(2 * np.pi * df['Hour']  / 24)
    df['Hour_cos']  = np.cos(2 * np.pi * df['Hour']  / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Lag Features
    df['Sales_lag1']           = df['Sales Count'].shift(1)
    df['Sales_lag2']           = df['Sales Count'].shift(2)
    df['Sales_lag4']           = df['Sales Count'].shift(4)
    df['Sales_lag8']           = df['Sales Count'].shift(8)
    df['Sales_roll_mean_4']    = df['Sales Count'].rolling(4).mean()
    df['Sales_roll_mean_8']    = df['Sales Count'].rolling(8).mean()
    df['Sales_roll_std_4']     = df['Sales Count'].rolling(4).std()
    df['Sales_roll_max_4']     = df['Sales Count'].rolling(4).max()
    df['Sales_same_yesterday'] = df['Sales Count'].shift(96)
    df['Sales_same_lastweek']  = df['Sales Count'].shift(672)
    df['Redemp_lag1']          = df['Redemption Count'].shift(1)
    df['Redemp_lag2']          = df['Redemption Count'].shift(2)
    df['Redemp_lag4']          = df['Redemption Count'].shift(4)
    df['Redemp_lag8']          = df['Redemption Count'].shift(8)
    df['Redemp_roll_mean_4']   = df['Redemption Count'].rolling(4).mean()
    df['Redemp_roll_std_4']    = df['Redemption Count'].rolling(4).std()

    df = df.dropna().reset_index(drop=True)

    # Train / Test Split
    train = df[df['Timestamp'] < '2024-01-01'].reset_index(drop=True)
    test  = df[df['Timestamp'] >= '2024-01-01'].reset_index(drop=True)

    return df, train, test