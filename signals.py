import pandas as pd
import numpy as np

def signal_bollinger_bands(price_df, target_col, window, std_factor, delta=1e-3, ve=1e-2):
    def kalman_filter_step(price, m_prev, R_prev, delta, ve):
        m_pred = m_prev
        R_pred = R_prev + delta

        K = R_pred / (R_pred + ve)
        m_curr = m_pred + K * (price - m_pred)
        R_curr = (1 - K) * R_pred

        return m_curr, R_curr

    df = price_df[[target_col]].copy()
    n = len(df)
    m = np.zeros(n)
    R = np.zeros(n)

    initial_prices = df[target_col].values[:window]
    m[:window] = initial_prices.mean()
    R[:window] = initial_prices.var()
    for t in range(window, n):
        m[t], R[t] = kalman_filter_step(df[target_col].values[t], m[t-1], R[t-1], delta, ve)

    df['MA'] = m
    df['SD'] = np.sqrt(R)

    df['U'] = df['MA'] + (df['SD'] * std_factor)
    df['L'] = df['MA'] - (df['SD'] * std_factor)

    return df

def signal_tsmom(prices, lookback, contra_lookback, std_threshold=2):
    def get_auto_covariance(returns, window):
        mean_returns = returns.rolling(window=window).mean()
        auto_cov = (returns - mean_returns).rolling(window=window).apply(lambda x: np.mean((x - x.mean()) * (x.shift(1) - x.shift(1).mean())))
        return auto_cov

    returns = prices.pct_change().fillna(0.)
    auto_cov = get_auto_covariance(returns, lookback).fillna(0.)

    signals = (1 * np.sign(auto_cov)).astype(int)
    contra_signal = 0
    if contra_lookback > 0:
        rolling_std = prices.rolling(window=contra_lookback).std()
        contra_signal = (((signals > 0) & (rolling_std < -std_threshold)) | ((signals < 0) & (rolling_std > std_threshold))).astype(int)

    signals_df = pd.DataFrame({
        'TSMOM': signals,
        'CONTRA': contra_signal
    }, index=returns.index)

    return signals_df