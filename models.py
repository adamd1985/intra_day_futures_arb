from signals import dynamic_support_resistance, kalman_backtest, kf_bollinger_band_backtest, tsmom_backtest
from quant_equations import get_ou, modulate_std
from tqdm import tqdm
from constants import *

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, fbeta_score, precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

KF_COLS = ['SD','Z1', 'Z2', 'Filtered_X', 'KG_X', 'KG_Z1', 'KG_Z2'] # ['Z1', 'Z2', 'Filtered_X', 'Uncertainty', 'Residuals', 'KG_X', 'KG_Z1', 'KG_Z2']
BB_COLS = ['MA', 'U','L'] # ['SB','SS','SBS','SSB', 'Unreal_Ret', 'MA','SD', 'U','L', '%B', 'X']
SR_COLS = ["Support", "Resistance"] # ["PP", "S1", "R1", "S2", "R2", "Support", "Resistance"]
MOM_COLS = ["TSMOM", "CONTRA"]
MARKET_COLS = [f"{fut}_{col}" for col in StockFeatExt.list for fut in MARKET_FUTS]
# We scale RAW column, the rest are percentages or log values.
COLS_TO_SCALE = StockFeatExt.list + BB_COLS + SR_COLS

META_LABEL = "mr_label"


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
    df[META_LABEL] = 0
    for i, row in tqdm(df.iterrows(), desc="Posthoc Metalabeling"):
        if row['Closed'] != 0:
            # Position closed, work backwards
            metalabel = (row['Ret'] > 0.).astype(int)
            if start_index is not None and metalabel:
                df.loc[start_index:row.name, META_LABEL] = metalabel
            position = 0
            start_index = None
        if row['Position'] != 0 and position == 0:
            # New position opened
            position = row['Position']
            start_index = row.name

    return df

def augment_ts(df, target_close, target_high, target_low, target_volume, interval, cols_to_scale=COLS_TO_SCALE, scaler=None):
    hl, h = get_ou(df, target_close)
    window = abs(hl)
    mod_std = modulate_std(h)

    mom_df, _ = tsmom_backtest(df, target_close, interval, int(window*2), contra_lookback=window//2, std_threshold=mod_std)
    bb_df, _ = kf_bollinger_band_backtest(df[target_close], df[target_volume], interval, std_factor=mod_std)
    sr_df, _, _ = dynamic_support_resistance(df, target_close, target_high, target_low, initial_window_size=window)
    kf_df, _ = kalman_backtest(bb_df["%B"].bfill().ffill(), df[target_volume], df[target_close], period=interval)

    aug_ts_df = pd.concat([df[StockFeatExt.list], sr_df, kf_df, bb_df, mom_df], axis=1).bfill().ffill()
    aug_ts_df = aug_ts_df.loc[:, ~aug_ts_df.columns.duplicated(keep="first")]
    if cols_to_scale is not None:
        # Scale the raw values, and concat with the signals.
        aug_df_scaled = None
        if scaler is None:
            scaler = StandardScaler()
            aug_df_scaled = scaler.fit_transform(aug_ts_df[cols_to_scale])
        else:
            aug_df_scaled = scaler.transform(aug_ts_df[cols_to_scale])

        aug_df_scaled = pd.DataFrame(aug_df_scaled, columns=cols_to_scale)
        aug_ts_df = pd.concat([aug_df_scaled, aug_ts_df.drop(columns=cols_to_scale).reset_index(drop=True)], axis=1)
        aug_ts_df = aug_ts_df.loc[:, ~aug_ts_df.columns.duplicated(keep="first")]

    return aug_ts_df, scaler

def process_exog(futures, futs_df):
    futs_exog_ts = []
    for f in tqdm(futures, desc="process_exog"):
        fut_df = futs_df.filter(regex=f"{f}_.*")

        train_df = fut_df
        futs_exog_ts.append(train_df)

    futs_exog_df = pd.concat(futs_exog_ts, axis=1)

    return futs_exog_df

def process_futures(futures, futs_df, futs_exog_df, train_size, interval, cols_to_scale=COLS_TO_SCALE):
    training_ts = []
    val_ts = []
    scalers = []
    for f in tqdm(futures, desc="process_futures"):
        fut_df = futs_df.filter(regex=f"{f}_.*")
        fut_df.columns = fut_df.columns.str.replace(f"{f}_", "", regex=False)
        fut_df = pd.concat([fut_df, futs_exog_df], axis=1)

        train_df, scaler = augment_ts(fut_df.iloc[:train_size], StockFeatExt.CLOSE, StockFeatExt.HIGH, StockFeatExt.LOW, StockFeatExt.VOLUME, interval, cols_to_scale=cols_to_scale)
        test_df, _ = augment_ts(fut_df.iloc[train_size:], StockFeatExt.CLOSE, StockFeatExt.HIGH, StockFeatExt.LOW, StockFeatExt.VOLUME, interval, cols_to_scale=cols_to_scale, scaler=scaler)
        training_ts.append(train_df.reset_index(drop=True))
        val_ts.append(test_df.reset_index(drop=True))
        scalers.append(scaler) # we use these later in the validation.

    return training_ts, val_ts, scalers