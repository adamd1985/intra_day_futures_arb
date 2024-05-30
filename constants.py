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