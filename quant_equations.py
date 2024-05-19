import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from scipy.stats import norm
from hurst import compute_Hc

from constants import YFinanceOptions


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


def calc_annualized_sharpe(rets, risk_free=0.035, period=YFinanceOptions.M15):
    mean_rets = rets.mean()
    std_rets = rets.std()

    sharpe_ratio = (mean_rets - risk_free / 252) / std_rets
    factor = 0.

    if period == YFinanceOptions.M1:
        factor = np.sqrt(60 * 24 * 252)
    elif period == YFinanceOptions.M15:
        factor = np.sqrt(4 * 24 * 252)
    elif period == YFinanceOptions.H1:
        factor = np.sqrt(24 * 252)
    elif period == YFinanceOptions.D1:
        factor = np.sqrt(252)
    else:
        raise ValueError("Unsupported period.")
    return sharpe_ratio * factor

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
