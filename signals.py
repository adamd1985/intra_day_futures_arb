import pandas as pd
import numpy as np

from pykalman import KalmanFilter
from tqdm import tqdm

from constants import YFinanceOptions

def signal_kf_bollinger_bands(price_df, target_col, volume_col, std_factor=2., kf_em_iters=5, t_max=0.1, interval=YFinanceOptions.M15):
    def get_daily_timesteps(interval):
        factor = 1
        if interval == YFinanceOptions.M15:
            factor = 4 * 24
        return factor

    daily_steps = get_daily_timesteps(interval)
    df = price_df[[target_col, volume_col]].copy()
    df['Tmax'] = df[volume_col].rolling(window=daily_steps).sum().bfill() * t_max / daily_steps
    df['V_e'] = df.apply(lambda row: row[volume_col] / row['Tmax'] if row['Tmax'] > 0 else 1, axis=1)

    kf = KalmanFilter(transition_matrices=[1],  # F, State Transition
                      observation_matrices=[1],  # H, Observation
                      initial_state_mean=df[target_col].values[0],
                      initial_state_covariance=1,
                      observation_covariance=1,  # Rt, Random Walk Noise
                      transition_covariance=0.01,  # Q, Random Walk Noise
                      em_vars=['transition_covariance', 'initial_state_mean', 'initial_state_covariance'])

    kf = kf.em(df[target_col].values, n_iter=kf_em_iters)

    state_means = []
    state_covariances = []
    state_mean = kf.initial_state_mean
    state_covariance = kf.initial_state_covariance

    for t in tqdm(range(len(df)), desc="signal_kf_bollinger_bands"):
        state_mean, state_covariance = kf.filter_update(state_mean,
                                                        state_covariance,
                                                        observation=df[target_col].values[t],
                                                        observation_covariance=df['V_e'].values[t])
        state_means.append(state_mean)
        state_covariances.append(state_covariance)

    state_means = pd.Series([x.flatten()[0] for x in state_means], index=df.index)
    state_covariances = pd.Series([x.flatten()[0] for x in state_covariances], index=df.index)

    df['MA'] = state_means
    df['SD'] = np.sqrt(state_covariances)
    df['U'] = df['MA'] + (df['SD'] * std_factor)
    df['L'] = df['MA'] - (df['SD'] * std_factor)
    df['%B'] = (df[target_col] - df['L']) / (df['U'] - df['L'])  # %B Indicator signal

    return df

def signal_bollinger_bands(price_df, target_col, window, std_factor):
    df = price_df[[target_col]].copy()
    df['MA'] = df[target_col].rolling(window=window).mean().bfill()
    df['SD'] = df[target_col].rolling(window=window).std().bfill()
    df['U'] = df['MA'] + (df['SD'] * std_factor)
    df['L'] = df['MA'] - (df['SD'] * std_factor)
    df['%B'] = (df[target_col] - df['L']) / (df['U'] - df['L']) # %B Indicator signal

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