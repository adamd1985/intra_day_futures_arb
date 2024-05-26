import pandas as pd
import numpy as np
import itertools
import math

from pykalman import KalmanFilter
from tqdm import tqdm
from scipy.stats import skew, kurtosis

from constants import YFinanceOptions
from quant_equations import calc_annualized_sharpe, deflated_sharpe_ratio, modulate_std

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import multiprocessing
NUM_CORES = multiprocessing.cpu_count()


def dynamic_support_resistance(df, cols, target_col, window_size=20, max_clusters=20):
    def find_optimal_clusters(data, max_clusters=150, min_clusters=2, max_no_improvement=5, sample_size=1000):
        def evaluate_clusters(n_clusters, data):
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=(256 * NUM_CORES), max_no_improvement=max_no_improvement)
            labels = km.fit_predict(data)
            if len(np.unique(labels)) < 2:
                return -1
            score = silhouette_score(data, labels, sample_size=sample_size)
            return score

        best_score = -1
        best_k = None
        low = min_clusters
        high = max_clusters

        while low <= high:
            mid = (low + high) // 2
            score = evaluate_clusters(mid, data)
            low_score = evaluate_clusters(mid - 1, data) if mid - 1 >= low else -1
            high_score = evaluate_clusters(mid + 1, data) if mid + 1 <= high else -1
            if score > best_score:
                best_score = score
                best_k = mid
            if low_score > score:
                high = mid - 1
            elif high_score > score:
                low = mid + 1
            else:
                break  # local maximum

        return best_k

    def sr_clusters_dtw(df, cols, max_clusters=150):
        data = df[cols].values
        optimal_clusters = find_optimal_clusters(data, max_clusters=max_clusters)
        km_model = MiniBatchKMeans(n_clusters=optimal_clusters, batch_size=(256 * NUM_CORES))
        labels = km_model.fit_predict(data)

        cluster_df = pd.DataFrame(data, columns=cols)
        cluster_df['cluster'] = labels

        return cluster_df, km_model

    def identify_support_resistance(cluster_df, km_model, target_col):
        centers = km_model.cluster_centers_
        cluster_means = cluster_df[['cluster', target_col]].groupby('cluster').mean()

        support_level = cluster_means.mean(axis=1).idxmin()
        support_value = centers[support_level].mean()
        resistance_level = cluster_means.mean(axis=1).idxmax()
        resistance_value = centers[resistance_level].mean()

        return support_value, resistance_value

    assert window_size >= max_clusters

    df = df.copy()

    dynamic_support = []
    dynamic_resistance = []
    df["Support"] = np.nan
    df["Resistance"] = np.nan
    for start in tqdm(np.arange(0, len(df), window_size)):
        window_df = df.iloc[start:start + window_size]
        try:
            cluster_df, km_model = sr_clusters_dtw(window_df, cols, max_clusters=min(len(window_df) - 1, window_size - 1))
            support, resistance = identify_support_resistance(cluster_df, km_model, target_col)
            df.iloc[start, df.columns.get_loc("Support")] = support
            df.iloc[start, df.columns.get_loc("Resistance")] = resistance
            dynamic_support.append(support)
            dynamic_resistance.append(resistance)
        except Exception as e:
            potential_supp = np.min(df.iloc[start:start + window_size][target_col])
            potential_res = np.max(df.iloc[start:start + window_size][target_col])
            df.iloc[start, df.columns.get_loc("Support")] = potential_supp
            df.iloc[start, df.columns.get_loc("Resistance")] = potential_res
            dynamic_support.append(potential_supp)
            dynamic_resistance.append(potential_res)
            print(e)
    df["Support"] = df["Support"].ffill()
    df["Resistance"] = df["Resistance"].ffill()
    return df, dynamic_support, dynamic_resistance


def tsmom_backtest(df, target_col, period, lookback=20, contra_lookback=5, std_threshold=1.5):
    df = df.copy()
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.0
    ts_mom_df = signal_tsmom(df[target_col], lookback=lookback, contra_lookback=contra_lookback, std_threshold=std_threshold)
    df['TSMOM'] = ts_mom_df['TSMOM']
    df['CONTRA'] = ts_mom_df['CONTRA']

    df['TSMOM_Shifted'] = df['TSMOM'].shift(1).fillna(0)
    df['CONTRA_Shifted'] = df['CONTRA'].shift(3).fillna(0)
    df['SB'] = (df['TSMOM_Shifted'] > 0) & (df['CONTRA_Shifted'] == 0)
    df['SS'] = (df['TSMOM_Shifted'] < 0) & (df['CONTRA_Shifted'] == 0)
    df['SBS'] = df['TSMOM_Shifted'] < 1
    df['SSB'] = df['TSMOM_Shifted'] > -1

    entry = 0
    position = 0
    for i, row in df.iterrows():
        if (row['SBS'] and position == 1) or (row['SSB'] and position == -1):
            if position == 1:
                df.at[i, 'Ret'] = (row[target_col] - entry) / entry
                df.at[i, 'Closed'] = 1
            else:
                df.at[i, 'Ret'] = (entry - row[target_col]) / entry
                df.at[i, 'Closed'] = -1
            position = 0
        if row['SB'] and position == 0:
            entry = row[target_col]
            position = 1
        elif row['SS'] and position == 0:
            entry = row[target_col]
            position = -1
        if position !=0 and row['CONTRA'] != 0:
            position = -position

        df.at[i, 'Position'] = position

    df['Ret'] = df['Position'] * df[target_col].pct_change().fillna(0)
    df['cRets'] = (1 + df['Ret']).cumprod() - 1

    variance = df['Ret'].var()
    df['Drawdown'] = (1 + df['Ret']).cumprod().div((1 + df['Ret']).cumprod().cummax()) - 1
    max_drawdown = df['Drawdown'].min()
    drawdown_length = (df['Drawdown'] < 0).astype(int).groupby(df['Drawdown'].eq(0).cumsum()).cumsum().max()
    sharpe = calc_annualized_sharpe(df['Ret'], period=period)
    trades = (df['Position'].diff().ne(0) & df['Position'].ne(0)).sum()

    stats_df = pd.DataFrame({
        "Window": [lookback],
        "Contrarian Window": [contra_lookback],
        "STD Threshold": [std_threshold],
        "Cumulative_Returns": [df['cRets'].iloc[-1]],
        "Max Ret": [df['Ret'].max()],
        "Max Loss": [df['Ret'].min()],
        "Variance": [variance],
        "STD": [np.sqrt(variance)],
        "Max_Drawdown": [max_drawdown],
        "Drawdown_Length": [drawdown_length],
        "Sharpe": [sharpe],
        "Trades_Count": [trades],
        "Trades_per_Interval": [trades / len(df)],
        "Trading_Intervals": [len(df)],
        "Rets": [df['Ret'].to_numpy()],
        "Rets_Skew": [skew(df['Ret'].to_numpy())],
        "Rets_Kurt": [kurtosis(df['Ret'].to_numpy())],
    })

    return df, stats_df

def param_search_tsmom(df, target_col, period, initial_window=20, window_factor = 1.5, window_min = 4, intial_std_threshold=1.5, hurst=0.5):
    assert initial_window > 0 and initial_window > window_min, f"initial_window: {initial_window} > window_min: {window_min}"

    num_steps = int(math.log(initial_window / window_min, window_factor)) + 1
    windows = [int(initial_window // (window_factor**i)) for i in range(num_steps)]
    contra_windows = [0, 1, 3] # in the paper they lookback 3 months (steps).
    std_thresholds = [intial_std_threshold, intial_std_threshold + 0.5]

    combinations = list(itertools.product(windows, contra_windows, std_thresholds))

    best_sharpe = -float('inf')
    best_sharpe_stats = None
    best_rets = -float('inf')
    best_rets_stats = None
    best_mdd = -float('inf')
    best_mdd_stats = None

    sharpes = []
    n_tests = len(combinations)

    for window, contra_window, std_threshold in tqdm(combinations, desc="param_search_tsmom"):
        std_factor = modulate_std (hurst, intial_std_threshold, std_threshold)
        _, stats_df = tsmom_backtest(df, target_col, period, lookback=window, contra_lookback=contra_window, std_threshold=std_factor)

        stat = stats_df['Sharpe'].iloc[0]
        sharpes.append(stat)
        if stat > best_sharpe:
            best_sharpe = stat
            best_sharpe_stats = stats_df.copy()

        stat = stats_df['Cumulative_Returns'].iloc[0]
        if stat > best_rets:
            best_rets = stat
            best_rets_stats = stats_df.copy()

        stat = stats_df['Max_Drawdown'].iloc[0]
        if stat > best_mdd:
            best_mdd = stat
            best_mdd_stats = stats_df.copy()

    # We're datamining, we need to deflated the sharpe!
    for df in [best_sharpe_stats, best_rets_stats, best_mdd_stats]:
        df['Sharpe'] = deflated_sharpe_ratio(df['Sharpe'].iloc[0],
                                            len(df['Rets'].iloc[0]),
                                            df['Rets_Skew'].iloc[0],
                                            df['Rets_Kurt'].iloc[0],
                                            sharpes,
                                            n_tests)

    results_df = pd.concat([best_sharpe_stats.assign(Metric='Sharpe'),
                            best_rets_stats.assign(Metric='Cumulative Returns'),
                            best_mdd_stats.assign(Metric='Max Drawdown')],
                           ignore_index=True)

    return results_df

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

def bollinger_band_backtest(price_df, target_col, window, period, std_factor=0.5, stoploss_pct=0.5):
    df = price_df.copy()
    bb_df = signal_bollinger_bands(df, target_col, window, std_factor)

    df['MA'] = bb_df['MA']
    df['SD'] = bb_df['SD']
    df['U'] = bb_df['U']
    df['L'] = bb_df['L']
    df['SB'] = (df[target_col] < bb_df['L']).astype(int).diff().clip(0) * +1
    df['SS'] = (df[target_col] > bb_df['U']).astype(int).diff().clip(0) * -1
    df['SBS'] = (df[target_col] > bb_df['MA']).astype(int).diff().clip(0) * -1
    df['SSB'] = (df[target_col] < bb_df['MA']).astype(int).diff().clip(0) * +1
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.
    entry = position = 0
    for i, row in df.iterrows():
        if df.index.get_loc(i) < window:
            df.loc[i, 'Position'] = 0
            continue
        if (row['SBS'] == -1 and position == 1) or \
            (row['SSB'] == 1 and position == -1) or \
            (position == 1 and row[target_col] <= row[target_col] - (stoploss_pct * entry)) or \
            (position == -1 and row[target_col] >= row[target_col] + (stoploss_pct * entry)):
            if position == 1:
                df.loc[i, 'Ret'] = (row[target_col] - entry) / entry
                df.loc[i, 'Closed'] = 1
            else:
                df.loc[i, 'Ret'] = (entry - row[target_col]) / entry
                df.loc[i, 'Closed'] = -1
            position = 0

        if (row['SB'] == 1 and position == 0) or (row['SS'] == -1 and position == 0):
            entry = row[target_col]
            position = 1 if row['SB'] == 1 else -1
        df.loc[i, 'Position'] = position
        # TODO: add unrealized returns to check for DDs.

    df['cRets'] = (1 + df['Ret']).cumprod() - 1

    variance = df['Ret'].var()
    df['Drawdown'] = (1 + df['Ret']).cumprod().div((1 + df['Ret']).cumprod().cummax()) - 1
    max_drawdown = df['Drawdown'].min()
    drawdown_length = (df['Drawdown'] < 0).astype(int).groupby(df['Drawdown'].eq(0).cumsum()).cumsum().max()
    sharpe = calc_annualized_sharpe(df['Ret'], period=period)
    trades = (df['Position'].diff().ne(0) & df['Position'].ne(0)).sum()
    stats_df = pd.DataFrame({
        "Window": [window],
        "Standard_Factor": [std_factor],
        "stoploss_pct": [stoploss_pct],
        "Cumulative_Returns": [df['cRets'].iloc[-1]],
        "Max Ret": [df['Ret'].max()],
        "Max Loss": [df['Ret'].min()],
        "Variance": [variance],
        "STD": [np.sqrt(variance)],
        "Max_Drawdown": [max_drawdown],
        "Drawdown_Length": [drawdown_length],
        "Sharpe": [sharpe],
        "Trades_Count": [trades],
        "Trades_per_Interval": [trades / len(df)],
        "Trading_Intervals": [len(df)],
        "Rets": [df['Ret'].to_numpy()],
        "Rets_Skew": [skew(df['Ret'].to_numpy())],
        "Rets_Kurt": [kurtosis(df['Ret'].to_numpy())],
    })

    return df, stats_df

def param_search_bbs(df, target_col, period, stoploss_pct=0.1, initial_window=20, window_factor = 1.5, window_min = 4, hurst=0.5):
    assert initial_window > window_min

    num_steps = int(math.log(initial_window / window_min, window_factor)) + 1
    windows = [int(initial_window // (window_factor**i)) for i in range(num_steps)]
    std_adjustments = [0.05, 0.25, 0.5]
    combinations = list(itertools.product(windows, std_adjustments))

    best_sharpe = -float('inf')
    best_sharpe_stats = None
    best_rets = -float('inf')
    best_rets_stats = None
    best_mdd = -float('inf')
    best_mdd_stats = None

    sharpes = []
    n_tests = len(combinations)

    for window, adjustment in tqdm(combinations, desc="param_search_bbs"):
        std_factor = modulate_std (hurst, adjustment=adjustment)
        _, stats_df = bollinger_band_backtest(df, target_col, window, period, std_factor=std_factor, stoploss_pct=stoploss_pct)

        stat = stats_df['Sharpe'].iloc[0]
        sharpes.append(stat)
        if stat > best_sharpe:
            best_sharpe = stat
            best_sharpe_stats = stats_df.copy()

        stat = stats_df['Cumulative_Returns'].iloc[0]
        if stat > best_rets:
            best_rets = stat
            best_rets_stats = stats_df.copy()

        stat = stats_df['Max_Drawdown'].iloc[0]
        if stat > best_mdd:
            best_mdd = stat
            best_mdd_stats = stats_df.copy()

    # We're datamining, we need to deflated the sharpe!
    for df in [best_sharpe_stats, best_rets_stats, best_mdd_stats]:
        df['Sharpe'] = deflated_sharpe_ratio(df['Sharpe'].iloc[0],
                                            len(df['Rets'].iloc[0]),
                                            df['Rets_Skew'].iloc[0],
                                            df['Rets_Kurt'].iloc[0],
                                            sharpes,
                                            n_tests)

    results_df = pd.concat([best_sharpe_stats.assign(Metric='Sharpe'),
                            best_rets_stats.assign(Metric='Cumulative Returns'),
                            best_mdd_stats.assign(Metric='Max Drawdown')],
                           ignore_index=True)

    return results_df

def kf_bollinger_band_backtest(price_df, target_col, volume_col, period, std_factor=0.5, stoploss_pct=0.9, t_max=0.1):
    df = price_df.copy()
    bb_df = signal_kf_bollinger_bands(df, target_col, volume_col, std_factor, t_max=t_max)

    df['MA'] = bb_df['MA']
    df['SD'] = bb_df['SD']
    df['U'] = bb_df['U']
    df['L'] = bb_df['L']
    df['SB'] = (df[target_col] < bb_df['L']).astype(int).diff().clip(0) * +1
    df['SS'] = (df[target_col] > bb_df['U']).astype(int).diff().clip(0) * -1
    df['SBS'] = (df[target_col] > bb_df['MA']).astype(int).diff().clip(0) * -1
    df['SSB'] = (df[target_col] < bb_df['MA']).astype(int).diff().clip(0) * +1
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.
    entry = position = 0
    for i, row in df.iterrows():
        if (row['SBS'] == -1 and position == 1) or \
            (row['SSB'] == 1 and position == -1) or \
            (position == 1 and row[target_col] <= row[target_col] - (stoploss_pct * entry)) or \
            (position == -1 and row[target_col] >= row[target_col] + (stoploss_pct * entry)):
            if position == 1:
                df.loc[i, 'Ret'] = (row[target_col] - entry) / entry
                df.loc[i, 'Closed'] = 1
            else:
                df.loc[i, 'Ret'] = (entry - row[target_col]) / entry
                df.loc[i, 'Closed'] = -1
            position = 0

        if (row['SB'] == 1 and position == 0) or (row['SS'] == -1 and position == 0):
            entry = row[target_col]
            position = 1 if row['SB'] == 1 else -1
        df.loc[i, 'Position'] = position
        # TODO: add unrealized returns to check for DDs.

    df['cRets'] = (1 + df['Ret']).cumprod() - 1

    variance = df['Ret'].var()
    df['Drawdown'] = (1 + df['Ret']).cumprod().div((1 + df['Ret']).cumprod().cummax()) - 1
    max_drawdown = df['Drawdown'].min()
    drawdown_length = (df['Drawdown'] < 0).astype(int).groupby(df['Drawdown'].eq(0).cumsum()).cumsum().max()
    sharpe = calc_annualized_sharpe(df['Ret'], period=period)
    trades = (df['Position'].diff().ne(0) & df['Position'].ne(0)).sum()
    stats_df = pd.DataFrame({
        "T_max": [t_max],
        "Standard_Factor": [std_factor],
        "stoploss_pct": [stoploss_pct],
        "Cumulative_Returns": [df['cRets'].iloc[-1]],
        "Max Ret": [df['Ret'].max()],
        "Max Loss": [df['Ret'].min()],
        "Variance": [variance],
        "STD": [np.sqrt(variance)],
        "Max_Drawdown": [max_drawdown],
        "Drawdown_Length": [drawdown_length],
        "Sharpe": [sharpe],
        "Trades_Count": [trades],
        "Trades_per_Interval": [trades / len(df)],
        "Trading_Intervals": [len(df)],
        "Rets": [df['Ret'].to_numpy()],
        "Rets_Skew": [skew(df['Ret'].to_numpy())],
        "Rets_Kurt": [kurtosis(df['Ret'].to_numpy())],
    })

    return df, stats_df

def param_search_kf_bbs(df, target_col, volume_col, period, hurst):
    std_adjustments = [0.05, 0.25, 0.5]
    t_maxs = [0.1, 0.5, 0.9]
    combinations = list(itertools.product(t_maxs, std_adjustments))

    best_sharpe = -float('inf')
    best_sharpe_stats = None
    best_rets = -float('inf')
    best_rets_stats = None
    best_mdd = -float('inf')
    best_mdd_stats = None

    sharpes = []
    n_tests = len(combinations)

    for t_max, adjustment in tqdm(combinations, desc="param_search_bbs"):
        std_factor = modulate_std(hurst, adjustment=adjustment)
        _, stats_df = kf_bollinger_band_backtest(df, target_col, volume_col, period, std_factor=std_factor, t_max=t_max)

        stat = stats_df['Sharpe'].iloc[0]
        sharpes.append(stat)
        if stat > best_sharpe:
            best_sharpe = stat
            best_sharpe_stats = stats_df.copy()

        stat = stats_df['Cumulative_Returns'].iloc[0]
        if stat > best_rets:
            best_rets = stat
            best_rets_stats = stats_df.copy()

        stat = stats_df['Max_Drawdown'].iloc[0]
        if stat > best_mdd:
            best_mdd = stat
            best_mdd_stats = stats_df.copy()

    # We're datamining, we need to deflated the sharpe!
    for df in [best_sharpe_stats, best_rets_stats, best_mdd_stats]:
        df['Sharpe'] = deflated_sharpe_ratio(df['Sharpe'].iloc[0],
                                            len(df['Rets'].iloc[0]),
                                            df['Rets_Skew'].iloc[0],
                                            df['Rets_Kurt'].iloc[0],
                                            sharpes,
                                            n_tests)

    results_df = pd.concat([best_sharpe_stats.assign(Metric='Sharpe'),
                            best_rets_stats.assign(Metric='Cumulative Returns'),
                            best_mdd_stats.assign(Metric='Max Drawdown')],
                           ignore_index=True)

    return results_df

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


def signal_kf(spread, volumes, prices, em_train_perc=0.1, em_iter=5, delta_t=1, q_t=1e-4/(1-1e-4), r_t=0.1):
    # State transition matrix
    train_size = int(len(spread) * em_train_perc)
    F = np.array([
        [1, delta_t, 0.5 * delta_t**2],
        [0, 1, delta_t],
        [0, 0, 1]
    ])
    # Observation matrix
    H = np.array([[1, 0, 0]])
    # Initial values don't have that much affect down the line.
    initial_x = np.mean(spread[:train_size])
    initial_var = np.var(spread[:train_size])
    state_mean = np.array([initial_x, 0, 0])
    # https://pykalman.github.io/
    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        initial_state_mean=state_mean,
        initial_state_covariance=np.eye(3),
        observation_covariance=np.eye(1) * r_t,  # Observation Noise.
        transition_covariance=np.eye(3) * q_t,  # Q, Process Noise.
        em_vars=['transition_covariance', 'observation_covariance',
                 'initial_state_mean', 'initial_state_covariance']
    )

    # 'Train'. EM to find the best Model Var
    kf = kf.em(spread[:train_size], n_iter=em_iter)
    filtered_state_means, filtered_state_covariances = kf.filter(spread[:train_size])
    state_mean = filtered_state_means[-1]
    state_covariance = filtered_state_covariances[-1]

    filtered_state_means = []
    hidden_1 = []
    hidden_2 = []
    filtered_state_covariances = []
    kalman_gains = []

    for i in tqdm(range(train_size, len(spread))):
        # Rt = Pt * Vt-1 / min(Vt-1, Vt)
        if volumes[i-1] != 0 and volumes[i] != 0:
            Rt = (state_covariance[0, 0] * volumes[i-1]) / min(volumes[i-1], volumes[i])
        else:
            Rt = state_covariance[0, 0]
        assert not np.isnan(Rt).any(), f"{Rt} = {state_covariance[0, 0] } * {volumes[i-1]} / {min(volumes[i-1], volumes[i])} at {i}"
        state_mean, state_covariance = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_covariance,
            observation=np.array([spread[i]]),
            observation_matrix=H,
            observation_covariance=np.array([[Rt]])
        )

        kalman_gain = state_covariance @ H.T @ np.linalg.inv(H @ state_covariance @ H.T + np.array([[Rt]]))
        kalman_gains.append(kalman_gain[:, 0])
        filtered_state_means.append(state_mean[0])
        filtered_state_covariances.append(state_covariance[0, 0])
        hidden_1.append(state_mean[1])
        hidden_2.append(state_mean[2])

    residuals = spread[train_size:] - np.array(filtered_state_means)

    results = pd.DataFrame({
        'Close': prices[train_size:],
        'X': spread[train_size:],
        'Z1': hidden_1,
        'Z2': hidden_2,
        'Filtered_X': filtered_state_means,
        'Uncertainty': filtered_state_covariances,
        'Residuals': residuals,
        'KG_X': [kg[0] for kg in kalman_gains],
        'KG_Z1': [kg[1] for kg in kalman_gains],
        'KG_Z2': [kg[2] for kg in kalman_gains]
    })

    return results

def kalman_backtest(spread, volumes, prices, period, thresholds=[0, 0.5, 1], stoploss_pct=0.9, delta_t=1, q_t=1e-4/(1-1e-4), r_t=0.1):
    results = signal_kf(spread, volumes, prices, delta_t=delta_t, q_t=q_t, r_t=r_t)
    df = results.copy()

    df['SB'] = (df['Filtered_X'] <= thresholds[0]).astype(int).diff().clip(0) * +1
    df['SS'] = (df['Filtered_X'] >= thresholds[2]).astype(int).diff().clip(0) * -1
    df['SBS'] = (df['Filtered_X'] >= thresholds[1]).astype(int).diff().clip(0) * -1
    df['SSB'] = (df['Filtered_X'] <= thresholds[1]).astype(int).diff().clip(0) * +1
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.0
    entry = position = 0
    for i, row in tqdm(df.iterrows(), desc="kalman_backtest"):
        if (row['SBS'] == -1 and position == 1) or \
           (row['SSB'] == 1 and position == -1) or \
           (position == 1 and row['Close'] <= entry * (1 - stoploss_pct)) or \
           (position == -1 and row['Close'] >= entry * (1 + stoploss_pct)):
            if position == 1:
                df.loc[i, 'Ret'] = (row['Close'] - entry) / entry
                df.loc[i, 'Closed'] = 1
            else:
                df.loc[i, 'Ret'] = (entry - row['Close']) / entry
                df.loc[i, 'Closed'] = -1
            position = 0

        if (row['SB'] == 1 and position == 0) or (row['SS'] == -1 and position == 0):
            entry = row['Close']
            position = 1 if row['SB'] == 1 else -1
        df.loc[i, 'Position'] = position

    df['cRets'] = (1 + df['Ret']).cumprod() - 1

    variance = df['Ret'].var()
    df['Drawdown'] = (1 + df['Ret']).cumprod().div((1 + df['Ret']).cumprod().cummax()) - 1
    max_drawdown = df['Drawdown'].min()
    drawdown_length = (df['Drawdown'] < 0).astype(int).groupby(df['Drawdown'].eq(0).cumsum()).cumsum().max()
    sharpe = calc_annualized_sharpe(df['Ret'], period=period)
    trades = (df['Position'].diff().ne(0) & df['Position'].ne(0)).sum()
    stats_df = pd.DataFrame({
        "Thresholds": [thresholds],
        "Stoploss_pct": [stoploss_pct],
        "Cumulative_Returns": [df['cRets'].iloc[-1]],
        "Max Ret": [df['Ret'].max()],
        "Max Loss": [df['Ret'].min()],
        "Variance": [variance],
        "STD": [np.sqrt(variance)],
        "Max_Drawdown": [max_drawdown],
        "Drawdown_Length": [drawdown_length],
        "Sharpe": [sharpe],
        "Trades_Count": [trades],
        "Trades_per_Interval": [trades / len(df)],
        "Trading_Intervals": [len(df)],
        "Rets": [df['Ret'].to_numpy()],
        "Rets_Skew": [skew(df['Ret'].to_numpy())],
        "Rets_Kurt": [kurtosis(df['Ret'].to_numpy())],
    })

    return df, stats_df