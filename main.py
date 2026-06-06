import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('dataset (1).csv')
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df = df.sort_values('datetime').reset_index(drop=True)
option_cols = [c for c in df.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])
ce_strikes = np.array([int(c.replace('NIFTY27JAN26','').replace('CE','')) for c in ce_cols])
pe_strikes = np.array([int(c.replace('NIFTY27JAN26','').replace('PE','')) for c in pe_cols])
EXPIRY_DT = pd.Timestamp('2026-01-27 15:30:00')
df['is_expiry_day'] = (df['datetime'].dt.date == pd.Timestamp('2026-01-27').date()).astype(int)

atm_iv_list = []
for _, row in df.iterrows():
    spot = row['underlying_price']
    best_col, best_dist = None, np.inf
    for col, k in zip(ce_cols, ce_strikes):
        d = abs(k - spot)
        if d < best_dist and not np.isnan(row[col]):
            best_dist = d; best_col = col
    atm_iv_list.append(row[best_col] if best_col else np.nan)
atm_iv_s = pd.Series(atm_iv_list).ffill().bfill()
ce_row_mean = df[ce_cols].mean(axis=1)
pe_row_mean = df[pe_cols].mean(axis=1)

def fit_svi(strikes_obs, iv_obs, K, spot):
    try:
        lm = np.log(np.array(strikes_obs, dtype=float) / spot)
        lm_K = np.log(K / spot)
        X = np.column_stack([np.ones(len(lm)), lm, lm**2])
        coeffs, _, _, _ = np.linalg.lstsq(X, np.array(iv_obs)**2, rcond=None)
        iv2 = coeffs[0] + coeffs[1]*lm_K + coeffs[2]*lm_K**2
        return float(np.sqrt(max(iv2, 1e-8)))
    except:
        return np.nan

print("Building features...")
records = []
for i, row in df.iterrows():
    spot = row['underlying_price']
    dt = row['datetime']
    exp_hours = max((EXPIRY_DT - dt).total_seconds() / 3600, 0)
    tod = dt.hour * 60 + dt.minute
    sin_tod = np.sin(2*np.pi*tod/(6.25*60)); cos_tod = np.cos(2*np.pi*tod/(6.25*60))
    atm_iv = atm_iv_s.iloc[i]
    is_expiry = int(row['is_expiry_day'])

    for cols, strikes, otype in [(ce_cols, ce_strikes, 1), (pe_cols, pe_strikes, 0)]:
        vals_row = row[cols].values.astype(float)
        side_mean = ce_row_mean.iloc[i] if otype==1 else pe_row_mean.iloc[i]

        for j, (col, K) in enumerate(zip(cols, strikes)):
            iv_val = row[col]
            other_vals = np.delete(vals_row, j)
            other_strikes = np.delete(strikes, j)
            obs_mask = ~np.isnan(other_vals)
            n_obs = obs_mask.sum()

            left_iv  = vals_row[j-1] if j > 0 else np.nan
            right_iv = vals_row[j+1] if j < len(cols)-1 else np.nan

            svi = np.nan; cs_linear = np.nan; cs_cubic = np.nan
            if n_obs >= 3:
                svi = fit_svi(other_strikes[obs_mask], other_vals[obs_mask], K, spot)
            if n_obs >= 2:
                try: cs_linear = float(interp1d(other_strikes[obs_mask], other_vals[obs_mask], kind='linear', fill_value='extrapolate')(K))
                except: pass
            if n_obs >= 4:
                try: cs_cubic = float(interp1d(other_strikes[obs_mask], other_vals[obs_mask], kind='cubic', fill_value='extrapolate')(K))
                except: pass

            lag1 = df[col].iloc[i-1] if i > 0 else np.nan
            lag2 = df[col].iloc[i-2] if i > 1 else np.nan
            past = df[col].iloc[max(0,i-10):i].dropna().values
            roll5 = float(np.mean(past[-5:])) if len(past)>=3 else np.nan
            roll10 = float(np.mean(past)) if len(past)>=5 else np.nan
            ewm5 = float(df[col].iloc[:i].ewm(span=5).mean().iloc[-1]) if i>0 else np.nan

            records.append({
                'row_idx': i, 'contract': col, 'iv': iv_val,
                'strike': K, 'spot': spot, 'log_m': np.log(K/spot), 'log_m_sq': np.log(K/spot)**2,
                'option_type': otype, 'is_expiry': is_expiry,
                'expiry_hours': exp_hours, 'sqrt_expiry': np.sqrt(exp_hours), 'log_expiry': np.log(max(exp_hours,0.1)),
                'bar': i, 'sin_tod': sin_tod, 'cos_tod': cos_tod,
                'left_iv': left_iv, 'right_iv': right_iv,
                'svi': svi, 'cs_linear': cs_linear, 'cs_cubic': cs_cubic, 'n_obs': n_obs,
                'lag1': lag1, 'lag2': lag2, 'roll5': roll5, 'roll10': roll10, 'ewm5': ewm5,
                'atm_iv': atm_iv, 'side_mean': side_mean,
            })

feat_df = pd.DataFrame(records)
FEATURES = [
    'strike','spot','log_m','log_m_sq','option_type','is_expiry',
    'expiry_hours','sqrt_expiry','log_expiry','bar','sin_tod','cos_tod',
    'left_iv','right_iv','svi','cs_linear','cs_cubic','n_obs',
    'lag1','lag2','roll5','roll10','ewm5','atm_iv','side_mean',
]

train_df = feat_df[feat_df['iv'].notna()].copy()
X = train_df[FEATURES]; y = train_df['iv']

# CV to confirm final blend
tscv = TimeSeriesSplit(n_splits=5)
blend_mses = []
for fold, (tr, te) in enumerate(tscv.split(X), 1):
    te_ne = [t for t in te if train_df.iloc[t]['is_expiry']==0]
    if not te_ne: continue
    lgbm = LGBMRegressor(n_estimators=3000, learning_rate=0.01, max_depth=-1, num_leaves=21,
                         subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                         reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1)
    lgbm.fit(X.iloc[tr], y.iloc[tr])
    lgbm_pred = lgbm.predict(X.iloc[te_ne])
    svi_pred_val = train_df.iloc[te_ne]['svi'].fillna(train_df.iloc[te_ne]['cs_linear']).values
    blend = 0.8*svi_pred_val + 0.2*lgbm_pred
    mse = mean_squared_error(y.iloc[te_ne].values, blend)
    blend_mses.append(mse)
    print(f"  Fold {fold}: MSE={mse:.8f}")

print(f"\nFinal blend (80% SVI + 20% LGBM) non-expiry MSE: {np.mean(blend_mses):.8f}")
print("Done. Ready to write final script.") 
# ============================================================
# FINAL TRAIN ON FULL DATA
# ============================================================

print("\nTraining final model...")

final_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=21,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1
)

final_model.fit(
    train_df[FEATURES],
    train_df["iv"]
)

# ============================================================
# PREDICT MISSING VALUES
# ============================================================

missing_df = feat_df[
    feat_df["iv"].isna()
].copy()

print("Missing rows:", len(missing_df))

lgb_pred = final_model.predict(
    missing_df[FEATURES]
)

# ============================================================
# SVI PREDICTION
# ============================================================

svi_pred = (
    missing_df["svi"]
    .fillna(missing_df["cs_linear"])
    .fillna(missing_df["atm_iv"])
)

# ============================================================
# FINAL BLEND
# ============================================================

final_pred = (
    0.80 * svi_pred.values +
    0.20 * lgb_pred
)

missing_df["pred_iv"] = np.clip(
    final_pred,
    0.01,
    5.0
)

# ============================================================
# BUILD SUBMISSION
# ============================================================

submission_rows = []

for _, row in missing_df.iterrows():

    original_row = df.iloc[
        int(row["row_idx"])
    ]

    dt_str = original_row["datetime"].strftime(
        "%d-%m-%Y %H:%M"
    )

    submission_rows.append({
        "id":
            f"{dt_str}||{row['contract']}",
        "value":
            float(row["pred_iv"])
    })

submission = pd.DataFrame(
    submission_rows,
    columns=["id", "value"]
)

submission = (
    submission
    .sort_values("id")
    .reset_index(drop=True)
)

# ============================================================
# SAVE
# ============================================================

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission saved successfully.")
print("Shape:", submission.shape)
print(submission.head())

# sanity check
assert submission.shape[1] == 2
assert "id" in submission.columns
assert "value" in submission.columns