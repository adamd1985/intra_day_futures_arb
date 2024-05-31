# %% [code]
# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
import itertools
import math

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from scipy.stats import norm
from hurst import compute_Hc

from pykalman import KalmanFilter
from scipy.stats import skew, kurtosis

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from tqdm import tqdm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, fbeta_score, precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay

import matplotlib.pyplot as plt

import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

## CONSTANTS

class YFinanceOptions:
    INDEX = "Datetime"
    MIN1_RANGE = 7 - 1
    MIN15_RANGE = 60 - 1
    HOUR_RANGE = 730 - 1
    DAY_RANGE = 7300 - 1
    D1="1d"
    H1="1h"
    M15="15m"
    M1="1m"
    DATE_TIME_FORMAT = "%Y-%m-%d"
    DATE_TIME_HRS_FORMAT = '%Y-%m-%d %H:%M:%S %Z'

INTERVAL = YFinanceOptions.M15

class StockFeat:
    DATETIME = "Date"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    list = [OPEN, HIGH, LOW, CLOSE, VOLUME]

class StockFeatExt(StockFeat):
    SPREAD = "Spread"
    BARCOUNT = "Barcount"
    AVERAGE = "Average"
    list = StockFeat.list + [SPREAD, BARCOUNT, AVERAGE]

PERIOD_PD_FREQ = {
    YFinanceOptions.M1: '1T',
    YFinanceOptions.M15: '15T',
}

# Market
SNP_FUT = "ES" # E-min SnP futures
NSDQ_FUT = "NQ"
VOLATILITY_FUT= "VXM"  # CBOE Volatility Futs - for May2024 - no continous.
RUS_FUT = "RTY"
FUTURE_RATES_FUT = "10Y" # 10 Year Yield
CURRENT_RATES_FUT = "2YY" # 10 Year Yield
MARKET_FUTS = [SNP_FUT, NSDQ_FUT, VOLATILITY_FUT, RUS_FUT, CURRENT_RATES_FUT, FUTURE_RATES_FUT]

# Metals
GOLD_FUT = "GC"
SILVER_FUT = "SI"
COPPER_FUT = "HG"
PLATINUM_FUT = "PL"
PALLADIUM_FUT = "PA"

METALS_FUTS = [GOLD_FUT, SILVER_FUT, COPPER_FUT, PLATINUM_FUT, PALLADIUM_FUT]

# Energy
CRUDEOIL_FUT = "CL"
NATURALGAS_FUT = "NG"
HEATINGOIL_FUT = "HO"
RBOB_FUT = "RB"


ENERGY_FUTS = [CRUDEOIL_FUT, NATURALGAS_FUT, HEATINGOIL_FUT]

# Agri
CORN_FUT = "ZC"
SOYOIL_FUT = "ZL"
KCWHEAT_FUT = "KE"
SOYBEAN_FUT = "ZS"
SOYBEANMEAL_FUT = "ZM"
WHEAT_FUT = "ZW"
LIVECATTLE_FUT = "LE"
LEANHOG_FUT = "HE"
FEEDERCATTLE_FUT = "GF"
MILK_FUT = "DA"

AGRI_FUTS = [CORN_FUT, SOYOIL_FUT, KCWHEAT_FUT, SOYBEAN_FUT, SOYBEANMEAL_FUT, WHEAT_FUT, LIVECATTLE_FUT, LEANHOG_FUT, FEEDERCATTLE_FUT,MILK_FUT]

FUTS = AGRI_FUTS + MARKET_FUTS # + METALS_FUTS + ENERGY_FUTS

TARGET_FUT=WHEAT_FUT

KF_COLS = ['SD','Z1', 'Z2', 'Filtered_X', 'KG_X', 'KG_Z1', 'KG_Z2'] # ['Z1', 'Z2', 'Filtered_X', 'Uncertainty', 'Residuals', 'KG_X', 'KG_Z1', 'KG_Z2']
BB_COLS = ['MA', 'U','L'] # ['SB','SS','SBS','SSB', 'Unreal_Ret', 'MA','SD', 'U','L', '%B', 'X']
SR_COLS = ["Support", "Resistance"] # ["PP", "S1", "R1", "S2", "R2", "Support", "Resistance"]
MOM_COLS = ["TSMOM", "CONTRA"]
MARKET_COLS = [f"{fut}_{col}" for col in StockFeatExt.list for fut in MARKET_FUTS]
# We scale RAW column, the rest are percentages or log values.
COLS_TO_SCALE = StockFeatExt.list + BB_COLS + SR_COLS

META_LABEL = "mr_label"
ALL_FEATURES = KF_COLS + BB_COLS + SR_COLS + MOM_COLS + MARKET_COLS + StockFeatExt.list
FEATURES_SELECTED = ['10Y_Barcount', '10Y_Spread', '10Y_Volume', '2YY_Spread', '2YY_Volume',
                    'CONTRA', 'Filtered_X', 'KG_X', 'KG_Z1', 'RTY_Spread', 'SD', 'Spread',
                    'TSMOM', 'VXM_Open', 'VXM_Spread', 'Volume']

### FORMULAS

def get_ou(df, col):
    log_prices = np.log(df[col])

    h, _, _ = compute_Hc(log_prices, kind='price', simplified=True)
    spread_lag = log_prices.shift(1).bfill()

    spread_ret = (log_prices - spread_lag).bfill()
    spread_lag2 = sm.add_constant(spread_lag)

    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    hl = int(round(-np.log(2) / res.params[1], 0))

    return hl, h

def modulate_std (hurst, base_std=2.0, adjustment=0.5):
    if hurst < 0.5:
        # Reverting, increase this band
        return base_std + (0.5 - hurst) * adjustment
    elif hurst > 0.5:
        # Trending, decrease band
        return base_std - (hurst - 0.5) * adjustment
    else:
        return base_std

def var_ratio(df, col, k=100):
    # https://mingze-gao.com/posts/lomackinlay1988/#source-code
    log_prices = np.log(df[col])

    rets = np.diff(log_prices)
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1/m * np.sum(np.square(rets_k - k * mu))

    # Variance Ratio
    vr = var_k / var_1
    # Phi1
    phi1 = 2 * (2*k - 1) * (k-1) / (3*k*T)
    # Phi2

    def delta(j):
        res = 0
        for t in range(j+1, T+1):
            t -= 1  # array index is t-1 for t-th element
            res += np.square((rets[t]-mu)*(rets[t-j]-mu))
        return res / ((T-1) * var_1)**2

    phi2 = 0
    for j in range(1, k, 2):
        phi2 += (2*(k-j)/k)**2 * delta(j)

    # Test statistics
    ts1 = (vr - 1) / np.sqrt(phi1)
    ts2 = (vr - 1) / np.sqrt(phi2)

    # P-values
    p_value1 = 2 * (1 - norm.cdf(np.abs(ts1)))  # two-tailed
    p_value2 = 2 * (1 - norm.cdf(np.abs(ts2)))

    return vr, ts1, p_value1, ts2, p_value2

def get_annualized_factor(period=YFinanceOptions.M15):
    factor = 0.

    if period == YFinanceOptions.M1:
        factor = (60 * 24 * 252)
    elif period == YFinanceOptions.M15:
        factor = (4 * 24 * 252)
    elif period == YFinanceOptions.H1:
        factor = (24 * 252)
    elif period == YFinanceOptions.D1:
        factor = (252)
    else:
        raise ValueError("Unsupported period.")
    return factor

def calc_annualized_sharpe(rets, risk_free=0.035, period=YFinanceOptions.M15):
    mean_rets = rets.mean()
    std_rets = rets.std()
    factor = get_annualized_factor(period)
    sharpe_ratio = 0.
    if std_rets != 0:
        sharpe_ratio = (mean_rets - (risk_free / factor)) / std_rets
        sharpe_ratio *= np.sqrt(factor)
    return sharpe_ratio

def deflated_sharpe_ratio(SR, T, skew, kurt, SRs, N):
    if SR < 0:
        # CDF of close to 0 is 0.5, making the DSF look good.
        return SR
    mu_SR = np.mean(SRs)
    sigma_SR = np.std(SRs)

    euler_mascheroni = 0.5772156649
    max_Z = (1 - euler_mascheroni) * norm.ppf(1 - 1. / N) + euler_mascheroni * norm.ppf(1 - 1. / (N * np.e))
    SR_max = mu_SR + sigma_SR * max_Z

    PSR = (SR - SR_max) / np.sqrt((1 - skew * SR + (kurt - 1) * SR**2) / T)
    DSR = norm.cdf(PSR)
    return DSR


### SIGNALS

def dynamic_support_resistance(price_df, target_col, high_col, low_col, initial_window_size=20, max_clusters=20):
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

    df = price_df.copy()

    df["PP"] = df[[high_col, low_col, target_col]].mean(axis=1)
    df["S1"] = 2 * df["PP"] - df[high_col]
    df["R1"] = 2 * df["PP"] - df[low_col]
    df["S2"] = df["PP"] - (df[high_col] - df[low_col])
    df["R2"] = df["PP"] + (df[high_col] - df[low_col])
    cols = ["PP", "S1", "R1", "S2", "R2", target_col]

    dynamic_support = []
    dynamic_resistance = []
    df["Support"] = np.nan
    df["Resistance"] = np.nan

    start = 0
    while start < len(df):
        window_size = min(initial_window_size, len(df) - start)
        if window_size < 2:
            break
        window_df = df.iloc[start:start + window_size]
        max_clusters = min(len(window_df) - 1, initial_window_size - 1)
        try:
            cluster_df, km_model = sr_clusters_dtw(window_df, cols, max_clusters=max_clusters)
            support, resistance = identify_support_resistance(cluster_df, km_model, target_col)
            df.iloc[start:start + window_size, df.columns.get_loc("Support")] = support
            df.iloc[start:start + window_size, df.columns.get_loc("Resistance")] = resistance
            dynamic_support.append(support)
            dynamic_resistance.append(resistance)
        except Exception as e:
            potential_supp = np.min(df.iloc[start:start + window_size][target_col])
            potential_res = np.max(df.iloc[start:start + window_size][target_col])
            df.iloc[start:start + window_size, df.columns.get_loc("Support")] = potential_supp
            df.iloc[start:start + window_size, df.columns.get_loc("Resistance")] = potential_res
            dynamic_support.append(potential_supp)
            dynamic_resistance.append(potential_res)
            print(e)

        # Dynamically adjust the window size
        window_size = int(window_size * 1.5)
        start += window_size

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
    remove_signals = False
    # TODO: Vectorize this.
    for i, row in df.iterrows():
        if (row['SBS'] and position == 1) or (row['SSB'] and position == -1):
            if position == 1:
                df.at[i, 'Ret'] = (row[target_col] - entry) / entry
                df.at[i, 'Closed'] = 1
            else:
                df.at[i, 'Ret'] = (entry - row[target_col]) / entry
                df.at[i, 'Closed'] = -1
            position = 0
            remove_signals = False
        if remove_signals:
            # Just to make graphs clearer.
            df.at[i, 'SB'] = False
            df.at[i, 'SS'] = False
            df.at[i, 'SBS'] = False
            df.at[i, 'SSB'] = False
        if row['SB'] and position == 0:
            entry = row[target_col]
            position = 1
            remove_signals = True
        elif row['SS'] and position == 0:
            entry = row[target_col]
            position = -1
            remove_signals = True
        if position != 0 and row['CONTRA'] != 0:
            # THe contrarian policy, if the variance shows a breakdown of momentum.
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


def param_search_tsmom(df, target_col, period, initial_window=20, window_factor = 1.5, window_min = 10, intial_std_threshold=1.5, hurst=0.5):
    assert initial_window > 0 and initial_window > window_min, f"initial_window: {initial_window} > window_min: {window_min}"

    num_steps = int(math.log(initial_window / window_min, window_factor)) + 1
    windows = [int(initial_window // (window_factor**i)) for i in range(num_steps)]
    contra_windows = [0, 3, window_min, initial_window//2] # in the paper they lookback 3 months (steps).
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

def signal_kf_bollinger_bands(price_df, volume_df, std_factor=2., kf_em_iters=5, t_max=0.1, q_t=0.1, r_t=1, interval=YFinanceOptions.M15):
    def get_daily_timesteps(interval):
        factor = 1
        if interval == YFinanceOptions.M15:
            factor = 4 * 24
        return factor

    daily_steps = get_daily_timesteps(interval)

    df = pd.DataFrame()
    df['Close'] = price_df
    df['Volume'] = volume_df
    df['Tmax'] = df['Volume'].rolling(window=daily_steps).sum().bfill() * t_max / daily_steps
    df['V_e'] = df.apply(lambda row: row['Volume'] / row['Tmax'] if row['Tmax'] > 0 else 1, axis=1)

    kf = KalmanFilter(transition_matrices=[1],  # F, State Transition
                      observation_matrices=[1],  # H, Observation
                      initial_state_mean=df['Close'].values[0],
                      initial_state_covariance=1,
                      observation_covariance=r_t,  # Rt, Random Walk Noise
                      transition_covariance=q_t,  # Q, Random Walk Noise
                      em_vars=['transition_covariance', 'initial_state_mean', 'initial_state_covariance'])
    kf = kf.em(df['Close'].values, n_iter=kf_em_iters)

    state_means = []
    state_covariances = []
    state_mean = kf.initial_state_mean
    state_covariance = kf.initial_state_covariance

    for t in tqdm(range(len(df)), desc="signal_kf_bollinger_bands"):
        state_mean, state_covariance = kf.filter_update(state_mean,
                                                        state_covariance,
                                                        observation=df['Close'].values[t],
                                                        observation_covariance=df['V_e'].values[t])
        state_means.append(state_mean)
        state_covariances.append(state_covariance)

    stats_df = pd.DataFrame()
    stats_df["MA"] = pd.Series([x.flatten()[0] for x in state_means], index=df.index)
    stats_df["SD"] = np.sqrt(pd.Series([x.flatten()[0] for x in state_covariances], index=df.index))
    stats_df["U"] = stats_df['MA'] + (stats_df['SD'] * std_factor)
    stats_df["L"] = stats_df['MA'] - (stats_df['SD'] * std_factor)
    stats_df["%B"] = (df['Close'] - stats_df['L']) / (stats_df['U'] - stats_df['L'])  # %B Indicator signal

    return stats_df


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
    remove_signals = False
    for i, row in df.iterrows():
        if df.index.get_loc(i) < window:
            # Signal doesnt have enough data.
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
            remove_signals = False
        if remove_signals:
            # Just to make graphs clearer.
            df.at[i, 'SB'] = False
            df.at[i, 'SS'] = False
            df.at[i, 'SBS'] = False
            df.at[i, 'SSB'] = False

        if (row['SB'] == 1 and position == 0) or (row['SS'] == -1 and position == 0):
            entry = row[target_col]
            position = 1 if row['SB'] == 1 else -1
            remove_signals = True
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

def kf_bollinger_band_backtest(price_df, volume_df, period, std_factor=0.5, stoploss_pct=0.9, t_max=0.1):
    df = price_df.copy()
    bb_df = signal_kf_bollinger_bands(price_df, volume_df, std_factor, t_max=t_max)

    df = pd.DataFrame()
    df['Close'] = price_df
    df['MA'] = bb_df['MA']
    df['SD'] = bb_df['SD']
    df['U'] = bb_df['U']
    df['L'] = bb_df['L']
    df["%B"] = bb_df["%B"]
    df['SB'] = (df['Close'] < bb_df['L']).astype(int).diff().clip(0) * +1
    df['SS'] = (df['Close'] > bb_df['U']).astype(int).diff().clip(0) * -1
    df['SBS'] = (df['Close'] > bb_df['MA']).astype(int).diff().clip(0) * -1
    df['SSB'] = (df['Close'] < bb_df['MA']).astype(int).diff().clip(0) * +1
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.
    entry = position = 0
    remove_signals= False
    for i, row in df.iterrows():
        if (row['SBS'] == -1 and position == 1) or \
            (row['SSB'] == 1 and position == -1) or \
            (position == 1 and row['Close'] <= row['Close'] - (stoploss_pct * entry)) or \
            (position == -1 and row['Close'] >= row['Close'] + (stoploss_pct * entry)):
            if position == 1:
                df.loc[i, 'Ret'] = (row['Close'] - entry) / entry
                df.loc[i, 'Closed'] = 1
            else:
                df.loc[i, 'Ret'] = (entry - row['Close']) / entry
                df.loc[i, 'Closed'] = -1
            position = 0
            remove_signals= False
        if remove_signals:
            # Just to make graphs clearer.
            df.at[i, 'SB'] = False
            df.at[i, 'SS'] = False
            df.at[i, 'SBS'] = False
            df.at[i, 'SSB'] = False
        if (row['SB'] == 1 and position == 0) or (row['SS'] == -1 and position == 0):
            entry = row['Close']
            position = 1 if row['SB'] == 1 else -1
            remove_signals= True
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

def param_search_kf_bbs(price_df, volume_df, period, hurst):
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
        _, stats_df = kf_bollinger_band_backtest(price_df, volume_df, period, std_factor=std_factor, t_max=t_max)

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

def signal_tsmom(prices_df, lookback, contra_lookback, std_threshold=2):
    def get_auto_covariance(returns, window):
        mean_returns = returns.rolling(window=window).mean()
        auto_cov = (returns - mean_returns).rolling(window=window).apply(lambda x: np.mean((x - x.mean()) * (x.shift(1) - x.shift(1).mean())))
        return auto_cov

    returns = prices_df.pct_change().fillna(0.)
    auto_cov = get_auto_covariance(returns, lookback).fillna(0.)

    signals = (1 * np.sign(auto_cov)).astype(int)
    contra_signal = 0
    if contra_lookback > 0:
        rolling_std = prices_df.rolling(window=contra_lookback).std()
        contra_signal = (((signals > 0) & (rolling_std < -std_threshold)) | ((signals < 0) & (rolling_std > std_threshold))).astype(int)

    signals_df = pd.DataFrame({'TSMOM': signals, 'CONTRA': contra_signal})

    return signals_df


def signal_kf(spread_df, volumes_df, price_df, em_train_perc=0.1, em_iter=5, delta_t=1, q_t=1e-4/(1-1e-4), r_t=0.1):
    # State transition matrix
    train_size = int(len(price_df) * em_train_perc)
    F = np.array([
        [1, delta_t, 0.5 * delta_t**2],
        [0, 1, delta_t],
        [0, 0, 1]
    ])
    # Observation matrix
    H = np.array([[1, 0, 0]])
    # Initial values don't have that much affect down the line.
    initial_x = np.mean(spread_df.iloc[:train_size])
    initial_var = np.var(spread_df.iloc[:train_size])
    state_mean = np.array([initial_x, 0, 0])
    # https://pykalman.github.io/
    kf = KalmanFilter(
        transition_matrices = F,
        observation_matrices=H,
        initial_state_mean=state_mean,
        initial_state_covariance=np.eye(3),
        observation_covariance=np.eye(1) * r_t,  # Observation Noise.
        transition_covariance=np.eye(3) * q_t,  # Q, Process Noise.
        em_vars=['transition_covariance', 'observation_covariance',
                 'initial_state_mean', 'initial_state_covariance']
    )

    # 'Train'. EM to find the best Model Var
    kf = kf.em(spread_df.iloc[:train_size], n_iter=em_iter)
    filtered_state_means, filtered_state_covariances = kf.filter(spread_df.iloc[:train_size])
    state_mean = filtered_state_means[-1]
    state_covariance = filtered_state_covariances[-1]

    filtered_state_means = []
    hidden_1 = []
    hidden_2 = []
    filtered_state_covariances = []
    kalman_gains = []

    for i in tqdm(range(train_size, len(price_df))):
        # Rt = Pt * Vt-1 / min(Vt-1, Vt)
        if volumes_df.iloc[i-1] != 0 and volumes_df.iloc[i] != 0:
            Rt = (state_covariance[0, 0] * volumes_df.iloc[i-1]) / min(volumes_df.iloc[i-1], volumes_df.iloc[i])
        else:
            Rt = state_covariance[0, 0]

        state_mean, state_covariance = kf.filter_update(
            filtered_state_mean=state_mean,
            filtered_state_covariance=state_covariance,
            observation=np.array([spread_df.iloc[i]]),
            observation_matrix=H,
            observation_covariance=np.array([[Rt]])
        )

        kalman_gain = state_covariance @ H.T @ np.linalg.inv(H @ state_covariance @ H.T + np.array([[Rt]]))
        kalman_gains.append(kalman_gain[:, 0])
        filtered_state_means.append(state_mean[0])
        filtered_state_covariances.append(state_covariance[0, 0])
        hidden_1.append(state_mean[1])
        hidden_2.append(state_mean[2])

    residuals = spread_df.iloc[train_size:] - np.array(filtered_state_means)

    results = pd.DataFrame({
        'Close': price_df.iloc[train_size:],
        'X': spread_df.iloc[train_size:],
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



def kalman_backtest(spread_df, volumes_df, price_df, period, thresholds=[0, 0.5, 1], stoploss_pct=0.9, delta_t=1, q_t=1e-4/(1-1e-4), r_t=0.1):
    df = signal_kf(spread_df, volumes_df, price_df, delta_t=delta_t, q_t=q_t, r_t=r_t)

    df['SB'] = (df['Filtered_X'] <= thresholds[0]).astype(int).diff().clip(0) * +1
    df['SS'] = (df['Filtered_X'] >= thresholds[2]).astype(int).diff().clip(0) * -1
    df['SBS'] = (df['Filtered_X'] >= thresholds[1]).astype(int).diff().clip(0) * -1
    df['SSB'] = (df['Filtered_X'] <= thresholds[1]).astype(int).diff().clip(0) * +1
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.
    df['Unreal_Ret'] = 0.
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
        if position != 0:
            # Unrealized for continuous returns tracking.
            df.loc[i, 'Unreal_Ret'] = (entry - row['Close']) / entry

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


## MODELS


def print_classification_metrics(X_test, y_test, best_model):
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # ROC Curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="GBC").plot(ax=ax1)
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name="GBC").plot(ax=ax2)
    ax2.set_title('Precision-Recall Curve')
    plt.show()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = fbeta_score(y_test, y_pred, average='weighted', beta=0.5)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Beta Score: {f1:.4f}')

def param_search(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [25, 95, 125],
        'learning_rate': [0.001, 0.02, 0.3],
        'max_depth': [1, 2, 6, 12],
    }

    # CV Dataset with testfolds
    folds = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    ps = PredefinedSplit(test_fold=folds)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='precision', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best precision score: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    return best_model

def clean_corr_colinear_features(train_ts_df, test_ts_df, ALL_COLS, CORR_THRESHOLD = 0.95, VIF_THRESHOLD = 5., DROP_COL = True):
    def calculate_vif(X):
        x_const = add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = x_const.columns
        vif_data["VIF"] = [
            variance_inflation_factor(x_const.values, i)
            for i in range(x_const.shape[1])
        ]
        return vif_data

    X_train = train_ts_df[ALL_COLS]
    X_test = test_ts_df[ALL_COLS]

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
    to_drop = sorted(to_drop, key=lambda x: (upper.columns.get_loc(x), -upper[x].max()))
    print(f"These are highly corr: {to_drop}")

    if to_drop is not None and DROP_COL:
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)

    vif_data = calculate_vif(X_train)
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()
    acceptable_vif = vif_data[vif_data["VIF"] < VIF_THRESHOLD].sort_values(by="feature")
    selected_features = acceptable_vif["feature"].tolist()
    if 'const' in selected_features:
        selected_features.remove('const')
    if DROP_COL:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    print(f"Multi-Colinear: {vif_data[vif_data['VIF'] >= VIF_THRESHOLD]['feature'].values}")
    CLEAN_FEATURES = X_train.columns

    return X_train, X_test, CLEAN_FEATURES

def aug_metalabel_mr(df, metalabel = META_LABEL):
    df = df.copy()
    df[metalabel] = 0
    position = 0
    start_index = None
    df[metalabel] = 0
    for i, row in df.iterrows():
        if row['Closed'] != 0:
            # Position closed, work backwards
            metalabel = int(row['Ret'] > 0.)
            if start_index is not None and metalabel:
                df.loc[start_index:row.name, META_LABEL] = metalabel
            position = 0
            start_index = None
        if row['Position'] != 0 and position == 0:
            # New position opened
            position = row['Position']
            start_index = row.name

    return df

def augment_ts(df, target_close, target_high, target_low, target_volume, interval):
    hl, h = get_ou(df, target_close)
    window = abs(hl)
    mod_std = modulate_std(h)

    mom_df, _ = tsmom_backtest(df, target_close, interval, int(window*2), contra_lookback=window//2, std_threshold=mod_std)
    bb_df, _ = kf_bollinger_band_backtest(df[target_close], df[target_volume], interval, std_factor=mod_std)
    sr_df, _, _ = dynamic_support_resistance(df, target_close, target_high, target_low, initial_window_size=window)
    kf_df, _ = kalman_backtest(bb_df["%B"].bfill().ffill(), df[target_volume], df[target_close], period=interval)

    aug_ts_df = pd.concat([df[StockFeatExt.list], sr_df, kf_df, bb_df, mom_df], axis=1).bfill().ffill()
    aug_ts_df = aug_ts_df.loc[:, ~aug_ts_df.columns.duplicated(keep="first")]

    return aug_ts_df

def process_exog(futures, futs_df):
    futs_exog_ts = []
    for f in tqdm(futures, desc="process_exog"):
        fut_df = futs_df.filter(regex=f"{f}_.*")

        train_df = fut_df
        futs_exog_ts.append(train_df)

    futs_exog_df = pd.concat(futs_exog_ts, axis=1)

    return futs_exog_df

def process_futures(futures, futs_df, futs_exog_df, train_size, interval):
    training_ts = []
    val_ts = []
    for f in tqdm(futures, desc="process_futures"):
        fut_df = futs_df.filter(regex=f"{f}_.*")
        fut_df.columns = fut_df.columns.str.replace(f"{f}_", "", regex=False)
        fut_df = pd.concat([fut_df, futs_exog_df], axis=1)

        train_df = augment_ts(fut_df.iloc[:train_size], StockFeatExt.CLOSE, StockFeatExt.HIGH, StockFeatExt.LOW, StockFeatExt.VOLUME, interval)
        test_df = augment_ts(fut_df.iloc[train_size:], StockFeatExt.CLOSE, StockFeatExt.HIGH, StockFeatExt.LOW, StockFeatExt.VOLUME, interval)
        training_ts.append(train_df.reset_index(drop=True))
        val_ts.append(test_df.reset_index(drop=True))

    return training_ts, val_ts