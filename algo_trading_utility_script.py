import pandas as pd
import numpy as np
import itertools
from datetime import datetime

import os

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from scipy.stats import norm
from hurst import compute_Hc

from pykalman import KalmanFilter
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import GridSearchCV, PredefinedSplit
import xgboost as xgb
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, fbeta_score, precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay

import matplotlib.pyplot as plt

import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

import yfinance as yf
import pandas_market_calendars as mcal

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

class StockFeat:
    DATETIME = "Date"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    list = [OPEN, HIGH, LOW, CLOSE, VOLUME]

# the brocker gives extended info.
class StockFeatExt(StockFeat):
    SPREAD = "Spread"
    BARCOUNT = "Barcount"
    AVERAGE = "Average"
    list = StockFeat.list + [SPREAD, BARCOUNT, AVERAGE]

PERIOD_PD_FREQ = {
    YFinanceOptions.M1: '1T',
    YFinanceOptions.M15: '15T',
    YFinanceOptions.D1: '1D',
}

# Market. Yfinance is missing some futures, using their index as proxy.
SNP_DX = "^GSPC"
NSDQ_DX = "^NDX"
VOLATILITY_DX= "^VIX"
RUS_DX= "^RUT"
Y10_RATE_DX = "^TNX"
Y5_RATE_DX = "^FVX"
MARKET_DX= [Y10_RATE_DX, Y5_RATE_DX, VOLATILITY_DX, NSDQ_DX, SNP_DX, RUS_DX]

# Metals
GOLD_FUT = "GC=F"
SILVER_FUT = "SI=F"
COPPER_FUT = "HG=F"
PLATINUM_FUT = "PL=F"
PALLADIUM_FUT = "PA=F"
METALS_FUTS = [GOLD_FUT, SILVER_FUT, COPPER_FUT, PLATINUM_FUT, PALLADIUM_FUT]

# Energy
CRUDEOIL_FUT = "CL=F"
NATURALGAS_FUT = "NG=F"
HEATINGOIL_FUT = "HO=F"
RBOB_FUT = "RB=F"
ENERGY_FUTS=[CRUDEOIL_FUT, NATURALGAS_FUT, HEATINGOIL_FUT, RBOB_FUT]

FUTS = METALS_FUTS + ENERGY_FUTS

# Equities
AAPL = "AAPL"
TESLA = "TSLA"
EQUITIES=[AAPL, TESLA]

# For yahoofinance, future tickers have =F flag.
ALL_TICKERS = EQUITIES +FUTS+MARKET_DX

DATA_PATH="./data"


# See EDA
KF_COLS = ['SD','Z1', 'Z2', 'Filtered_X', 'KG_X', 'KG_Z1', 'KG_Z2'] # ['Z1', 'Z2', 'Filtered_X', 'Uncertainty', 'Residuals', 'KG_X', 'KG_Z1', 'KG_Z2']
BB_COLS = [ '%B','MA', 'U','L'] # ['SB','SS','SBS','SSB', 'Unreal_Ret', 'MA','SD', 'U','L', '%B', 'X']

MOM_COLS = ["TSMOM"]
FUTS_COLS = [f"{idx}_{col}" for col in StockFeat.list for idx in FUTS]
MARKET_COLS = [f"{idx}_{col}" for col in StockFeat.list for idx in MARKET_DX]
MARKET_COLS_EXT = [f"{idx}_{col}" for col in StockFeatExt.list for idx in MARKET_DX]
# We scale RAW column, the rest are percentages or log values.
COLS_TO_SCALE = StockFeat.list + BB_COLS  + KF_COLS + MARKET_COLS

META_LABEL_RET = "META_LABEL_RET"
META_LABEL_MR = "META_LABEL_MR"
ALL_FEATURES = StockFeat.list + KF_COLS + BB_COLS + MOM_COLS + MARKET_COLS + FUTS_COLS
FEATURES_SELECTED = ['RB=F_Volume', 'PA=F_Volume', 'HO=F_Volume', 'PL=F_Volume', 'NG=F_Volume', 'Volume', 'GC=F_Volume', '^NDX_Volume', 'Z2', 'TSMOM', 'CL=F_Volume']
START_DATE = '2017-01-01'
SPLIT_DATE = '2018-1-1' # Turning point from train to tst
END_DATE = '2019-12-31'
DATE_TIME_FORMAT = "%Y-%m-%d"
INTERVAL = YFinanceOptions.D1


## DATA FUNCTIONS

def get_yf_tickers_df(tickers_symbols, start, end, interval=INTERVAL, datadir=DATA_PATH):
    tickers = {}
    earliest_end= pd.to_datetime(datetime.strptime(end,YFinanceOptions.DATE_TIME_FORMAT)).tz_localize("UTC")
    latest_start = pd.to_datetime(datetime.strptime(start,YFinanceOptions.DATE_TIME_FORMAT)).tz_localize("UTC")
    os.makedirs(datadir, exist_ok=True)
    for symbol in tickers_symbols:
        cached_file_path = f"{datadir}/{symbol}-{start.split(' ')[0]}-{end.split(' ')[0]}-{interval}.csv"
        print(f"Checking file: {cached_file_path}")
        if os.path.exists(cached_file_path):
            print(f"loading from {cached_file_path}")
            df = pd.read_csv(cached_file_path, parse_dates= True, index_col=0)
            try:
                df.index = pd.to_datetime(df.index).tz_localize('US/Central').tz_convert('UTC')
            except Exception as e:
                df.index = pd.to_datetime(df.index).tz_convert('UTC')
            assert len(df) > 0, "Empty data"
        else:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=True,
                interval=interval
            )
            assert len(df) > 0, "No data pulled"
            try:
                df.index = pd.to_datetime(df.index).tz_localize('US/Central').tz_convert('UTC')
            except Exception as e:
                df.index = pd.to_datetime(df.index).tz_convert('UTC')
        # Use adjusted close if available.
        if 'Adj Close' in df.columns:
            assert 'Close' in df.columns
            df.drop(columns=['Adj Close'], inplace=True)
            # df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        min_date = df.index.min()
        max_date = df.index.max()
        nan_count = df["Close"].isnull().sum()
        skewness = round(skew(df["Close"].dropna()), 2)
        kurt = round(kurtosis(df["Close"].dropna()), 2)
        outliers_count = (df["Close"] > df["Close"].mean() + (3 * df["Close"].std())).sum()
        print(
            f"{symbol} => min_date: {min_date}, max_date: {max_date}, kurt:{kurt}, skewness:{skewness}, outliers_count:{outliers_count},  nan_count: {nan_count}"
        )
        tickers[symbol] = df

        if min_date > latest_start:
            latest_start = min_date
        if max_date < earliest_end:
            earliest_end = max_date

    nyse = mcal.get_calendar('CME_Agriculture')
    schedule = nyse.schedule(start_date=latest_start, end_date=earliest_end)
    all_trading_days = mcal.date_range(schedule, frequency=PERIOD_PD_FREQ[interval], tz='UTC', normalize=True)

    for symbol, df in tickers.items():
        df_filtered = df[(df.index >= latest_start) & (df.index <= earliest_end)]
        df_reindexed = df_filtered.reindex(all_trading_days, method='nearest')
        df_reindexed.index = pd.to_datetime(df_reindexed.index)
        df_reindexed = df_reindexed[~df_reindexed.index.duplicated(keep='first')]
        df_reindexed.index.name = 'Date'
        df_reindexed = df_reindexed.bfill().ffill()
        tickers[symbol] = df_reindexed

        cached_file_path = f"{datadir}/{symbol}-{start.split(' ')[0]}-{end.split(' ')[0]}-{interval}.csv"
        if not os.path.exists(cached_file_path):
            df_reindexed.to_csv(cached_file_path, index=True)

    return tickers, latest_start, earliest_end


def clean_redundant_features(train_ts_df, test_ts_df, ALL_COLS, VIF_THRESHOLD = 5., DROP_COL = True):
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

    print(f"Multi-Colinear: {vif_data[vif_data['VIF'] >= VIF_THRESHOLD]['feature'].values}. Features dropped: {DROP_COL}")

    return X_train, X_test, selected_features

def aug_metalabel_mr(df):
    df = df.copy()
    df[META_LABEL_MR] = 0
    df[META_LABEL_RET] = 0
    position = 0
    start_index = None
    df[META_LABEL_MR] = 0
    for i, row in df.iterrows():
        if row['Closed'] != 0:
            # Position closed, work backwards
            metalabel = int(row['Ret'] > 0.)
            if start_index is not None and metalabel:
                df.loc[start_index:row.name, META_LABEL_MR] = metalabel
                df.loc[start_index:row.name, META_LABEL_RET] = row['Ret']
            position = 0
            start_index = None
        if row['Position'] != 0 and position == 0:
            # New position opened
            position = row['Position']
            start_index = row.name

    return df

def augment_ts(df, interval, train_df=None):
    mom_df, _ = tsmom_backtest(df, period=interval)
    bb_df, _ = kf_bollinger_band_backtest(df[StockFeat.CLOSE], df[StockFeat.VOLUME], period=interval)
    kf_df, _ = kalman_backtest(bb_df["%B"].bfill().ffill(), df[StockFeat.VOLUME], df[StockFeat.CLOSE], period=interval)

    aug_ts_df = pd.concat([df[StockFeat.list], kf_df, bb_df, mom_df], axis=1).bfill().ffill()
    aug_ts_df = aug_ts_df.loc[:, ~aug_ts_df.columns.duplicated(keep="first")]

    return aug_ts_df

def process_exog(stocks, tickers):
    futs_exog_ts = []
    for ticker in tqdm(tickers, desc="process_exog"):
        stock_df = stocks[ticker]
        stock_df = stock_df.copy()
        stock_df = stocks[ticker].copy()
        stock_df = stock_df.rename(columns=lambda col: f"{ticker}_{col}")
        futs_exog_ts.append(stock_df)

    stock_exog_df = pd.concat(futs_exog_ts, axis=1)

    return stock_exog_df

def prepare_security_for_training(stock_df, stock_exog_df, train_size, interval):
    stock_df = pd.concat([stock_df, stock_exog_df], axis=1)
    test_df = None
    if train_size is not None:
        train_df = augment_ts(stock_df.iloc[:train_size],  interval=interval)
        test_df = augment_ts(stock_df.iloc[train_size:],  interval=interval)
    else:
        train_df = augment_ts(stock_df,  interval=interval)

    return train_df, test_df

### QUANT FORMULAS

def get_ou(df, col=StockFeat.CLOSE):
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

def var_ratio(df, col=StockFeat.CLOSE, k=100):
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


### SIGNALS, BACKTESTS AND METRICS

def get_trade_metrics(df, period, risk_free_rate=1.5, market_index=None):
    # Most of these portfolio metrics are inspired by AlphaLens
    # Sharpe
    variance = df['Ret'].var()
    sharpe = calc_annualized_sharpe(df['Ret'], period=period)

    # Drawdown
    df['Drawdown'] = (1 + df['Ret']).cumprod().div((1 + df['Ret']).cumprod().cummax()) - 1
    max_drawdown = df['Drawdown'].min()
    drawdown_length = (df['Drawdown'] < 0).astype(int).groupby(df['Drawdown'].eq(0).cumsum()).cumsum().max()

    # Trade Churn
    trades = (df['Position'].diff().ne(0) & df['Position'].ne(0)).sum()

    # Calculate Beta
    beta = None
    if market_index is not None:
        market_index['Ret'] = pd.to_numeric(market_index[StockFeat.CLOSE].pct_change().fillna(0), errors='coerce').fillna(0)
        y = pd.to_numeric(df['Ret'], errors='coerce').fillna(0)
        X = sm.add_constant(market_index['Ret'].reset_index(drop=True))
        y = y.iloc[:len(X)].reset_index(drop=True)
        X = X.iloc[:len(y)].reset_index(drop=True)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]

    # Calculate Annualized Information Ratio
    factor = get_annualized_factor(period)
    active_return = df['Ret'] - (risk_free_rate / factor)
    tracking_error = active_return.std()
    information_ratio = (active_return.mean() / tracking_error) * np.sqrt(factor)

    # Calculate Trade Churn
    trade_churn = trades / len(df)

    stats_df = pd.DataFrame({
        "Cumulative_Returns": [(np.cumprod(1 + df['Ret']) - 1).values[-1]],
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
        "Beta": [beta],
        "Information_Ratio": [information_ratio],
        "Trade_Churn": [trade_churn],
    })

    return stats_df

def signal_tsmom(prices_df, lookback=252, decay_factor=0.94):
    returns = prices_df.pct_change().fillna(0.0)

    ex_ante_volatility = returns.ewm(span=int(2/(1-decay_factor)-1), adjust=False).std()
    past_returns = prices_df.pct_change(periods=lookback).shift(1).fillna(0.0)
    signals = np.sign(past_returns) / ex_ante_volatility

    signals = signals.replace([np.inf, -np.inf], 0).fillna(0)

    return signals

def tsmom_backtest(df, period, target_col=StockFeat.CLOSE, lookback=20, risk_free_rate=1.5, market_index=None):
    df = df.copy()
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.0
    df['TSMOM'] = np.sign(signal_tsmom(df[target_col], lookback=lookback))
    df['SB'] = 0
    df['SS'] = 0
    df['SBS'] = 0
    df['SSB'] = 0
    entry = 0
    position = 0

    # TODO: Vectorize this.
    for i, row in df.iterrows():
        if (row['TSMOM'] < 0 and position == 1) \
            or (row['TSMOM'] > 0 and position == -1):
            if position == 1:
                df.at[i, 'Ret'] = (row[target_col] - entry) / entry
                df.at[i, 'Closed'] = 1
            else:
                df.at[i, 'Ret'] = (entry - row[target_col]) / entry
                df.at[i, 'Closed'] = -1
            position = 0
        elif row['TSMOM'] > 0 and position == 0:
            entry = row[target_col]
            df.at[i, 'SB'] = 1
            position = 1
        elif row['TSMOM'] < 0  and position == 0:
            entry = row[target_col]
            df.at[i, 'SS'] = 1
            position = -1
        df.at[i, 'Position'] = position
    if position != 0:
        # Close the backtest.
        assert entry != 0
        df.at[df.index[-1], 'Closed'] = np.sign(position)
        df.at[df.index[-1],  'Ret'] = (row[target_col] - entry) / entry if position == 1 else (entry - row[target_col]) / entry

    stats_df = get_trade_metrics(df, period=period, risk_free_rate=risk_free_rate, market_index=market_index)
    stats_df["Window"] = [lookback]

    return df, stats_df


def param_search_tsmom(df, period, target_col=StockFeat.CLOSE, initial_window=20, window_step = 2, window_min = 10, market_index=None):
    assert initial_window > 0 and initial_window > window_min, f"initial_window: {initial_window} > window_min: {window_min}"

    windows = list(range(initial_window, window_min - 1, -window_step))
    best_sharpe = -float('inf')
    best_sharpe_stats = None
    best_rets = -float('inf')
    best_rets_stats = None
    best_mdd = -float('inf')
    best_mdd_stats = None

    sharpes = []
    n_tests = len(windows)

    for window in tqdm(windows, desc="param_search_tsmom"):
        _, stats_df = tsmom_backtest(df, period=period, target_col=target_col, lookback=window, market_index=market_index)

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


def signal_bollinger_bands(price_df, window, std_factor=0.5, target_col=StockFeat.CLOSE):
    df = price_df[[target_col]].copy()
    df['MA'] = df[target_col].rolling(window=window).mean().bfill()
    df['SD'] = df[target_col].rolling(window=window).std().bfill()
    df['U'] = df['MA'] + (df['SD'] * std_factor)
    df['L'] = df['MA'] - (df['SD'] * std_factor)
    df['%B'] = (df[target_col] - df['L']) / (df['U'] - df['L'])
    return df

def bollinger_band_backtest(price_df, window, period, target_col=StockFeat.CLOSE, std_factor=0.5, risk_free_rate=1.5,market_index=None):
    df = price_df.copy()
    bb_df = signal_bollinger_bands(df, window=window, std_factor=std_factor, target_col=target_col)
    df['MA'] = bb_df['MA']
    df['SD'] = bb_df['SD']
    df['U'] = bb_df['U']
    df['L'] = bb_df['L']
    df['%B'] = bb_df['%B']
    df['Closed'] = 0
    df['Position'] = 0
    df['Ret'] = 0.
    df['SB'] = 0
    df['SS'] = 0.
    entry = position = 0

    for i, row in df.iterrows():
        if df.index.get_loc(i) < window:
            # Signal doesnt have enough data.
            continue
        if (row['%B'] >= 0.5 and position == 1) or \
            (row['%B'] <= 0.5 and position == -1):
            if position == 1:
                df.loc[i, 'Ret'] = (row[target_col] - entry) / entry
                df.loc[i, 'Closed'] = 1
            else:
                df.loc[i, 'Ret'] = (entry - row[target_col]) / entry
                df.loc[i, 'Closed'] = -1
            position = 0
        elif (row['%B'] >= 1. and position == 0) or\
            (row['%B'] <= 0. and position == 0):
            entry = row[target_col]
            if row['%B'] <= 0.:
                position = 1
                df.loc[i, 'SB'] = 1
            else:
                position = -1
                df.loc[i, 'SS'] = 1
        df.loc[i, 'Position'] = position
    if position != 0:
        # Close the backtest.
        df.at[df.index[-1], 'Closed'] = np.sign(position)
        df.at[df.index[-1],  'Ret'] = (row[target_col] - entry) / entry if position == 1 else (entry - row[target_col]) / entry
    stats_df = get_trade_metrics(df, period=period, risk_free_rate=risk_free_rate, market_index=market_index)
    stats_df["Window"] = [window]
    stats_df["Standard_Factor"] = [std_factor]

    return df, stats_df

def param_search_bbs(df, period, target_col=StockFeat.CLOSE, initial_window=20, window_step = 2, window_min = 4, hurst=0.5, market_index=None):
    assert initial_window > window_min

    windows = list(range(initial_window, window_min - 1, -window_step))
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
        _, stats_df = bollinger_band_backtest(df,  window, period, target_col=target_col, std_factor=std_factor, market_index=market_index)

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

def kf_bollinger_band_backtest(price_df, volume_df, period, std_factor=0.5, stoploss_pct=0.9, t_max=0.1 , risk_free_rate=1.5, market_index=None):
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
    if position != 0:
        # Close the backtest.
        df.at[df.index[-1], 'Closed'] = np.sign(position)
        df.at[df.index[-1],  'Ret'] = (row['Close'] - entry) / entry if position == 1 else (entry - row['Close']) / entry

    stats_df = get_trade_metrics(df, period=period, risk_free_rate=risk_free_rate, market_index=market_index)
    stats_df["T_max"] = [t_max]
    stats_df["Standard_Factor"] = [std_factor]

    return df, stats_df

def param_search_kf_bbs(price_df, volume_df, period, hurst, market_index=None):
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
        _, stats_df = kf_bollinger_band_backtest(price_df, volume_df, period, std_factor=std_factor, t_max=t_max, market_index=market_index)

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
        'SD': filtered_state_covariances,
        'Residuals': residuals,
        'KG_X': [kg[0] for kg in kalman_gains],
        'KG_Z1': [kg[1] for kg in kalman_gains],
        'KG_Z2': [kg[2] for kg in kalman_gains]
    })

    return results



def kalman_backtest(spread_df, volumes_df, price_df, period, thresholds=[0, 0.5, 1], delta_t=1, q_t=1e-4/(1-1e-4), r_t=0.1, risk_free_rate=1.5, market_index=None):
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
           (row['SSB'] == 1 and position == -1):
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
    if position != 0:
        # Close the backtest.
        df.at[df.index[-1], 'Closed'] = np.sign(position)
        df.at[df.index[-1],  'Ret'] = (row['Close'] - entry) / entry if position == 1 else (entry - row['Close']) / entry
    stats_df = get_trade_metrics(df, period=period, risk_free_rate=risk_free_rate, market_index=market_index)
    stats_df["Thresholds"] = [thresholds]

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

def param_search(X_train, y_train, X_test, y_test, class_weights):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10,
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=5,
        verbose_eval=25,
    )
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, class_weights[1] if class_weights is not None else None]
    }
    folds = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    ps = PredefinedSplit(test_fold=folds)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                               param_grid=param_grid, cv=ps, scoring='precision', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best precision score: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    return best_model
